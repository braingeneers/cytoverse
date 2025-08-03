"""
Inverted File Index (IVF) implementation for vector partitioning.

This module implements the IVF component of IVFPQ for partitioning large vector datasets into manageable chunks. Each vector is assigned to one of n_partitions partitions based on proximity to the partition centroids.

The IVF index enables efficient approximate nearest neighbor search by:
1. Assigning query vectors to the most promising partitions
2. Searching only within those partitions (reducing search space)
3. Supporting asymmetric distance computation for better recall

Based on the IVFPQ approach described in:
https://towardsdatascience.com/similarity-search-with-ivfpq-9c6348fd4db3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path
import pickle
import logging
import onnxruntime as ort
import tempfile
from cytoverse.kmeans import export_kmeans_models

logger = logging.getLogger(__name__)


class InvertedFileIndex(nn.Module):
    """
    Inverted File Index for dataset partitioning.

    The IVF index partitions vectors using k-means clustering,
    creating an inverted index where each partition contains a list of vector IDs.
    This enables efficient nearest neighbor search by constraining search to
    the most relevant partition.

    Args:
        d: Input vector dimension
        n_partitions: Number of partitions

    Training Process:
    1. Run k-means clustering on the dataset to find centroids
    2. Assign each vector to its nearest centroid
    3. Build inverted lists mapping partition_id -> [vector_ids]

    Search Process:
    1. Find nearest centroids for query vector
    2. Search only within those partitions
    3. Rank results using refined distance computation
    """

    def __init__(self, d: int, n_partitions: int):
        super().__init__()

        self.d = d  # Input dimension
        self.n_partitions = n_partitions  # Number of partitions

        # Centroids: [n_partitions, d]
        self.centroids = nn.Parameter(torch.randn(n_partitions, d), requires_grad=False)

        # Inverted lists: partition_id -> list of vector indices
        # This will be populated during training
        self.inverted_lists = None

        # Vector assignments: vector_id -> partition_id
        # This helps with updates and debugging
        self.assignments = None

        # Whether the index has been trained
        self.is_trained = False

        logger.info(
            f"Initialized InvertedFileIndex: d={self.d}, n_partitions={self.n_partitions}"
        )

    def train_ivf(
        self,
        vectors: torch.Tensor,
        vector_ids: torch.Tensor,
        n_iterations: int = 50,
        verbose: bool = True,
    ) -> None:
        """
        Train the IVF index using ONNX k-means clustering.

        Args:
            vectors: Training vectors of shape [N, d]
            vector_ids: Optional vector IDs of shape [N]. If None, uses indices 0...N-1
            n_iterations: Number of k-means iterations
            verbose: Whether to print training progress
        """
        n_vectors = vectors.shape[0]

        if len(vector_ids) != n_vectors:
            raise ValueError(
                f"vector_ids length {len(vector_ids)} must match vectors length {n_vectors}"
            )

        if verbose:
            logger.info(
                f"Training IVF index on {n_vectors} vectors with {self.n_partitions} partitions using ONNX k-means"
            )

        # Export k-means ONNX models to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            export_kmeans_models(tmpdir)

            # Load ONNX k-means models
            init_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_init.onnx"))
            iter_session = ort.InferenceSession(
                str(Path(tmpdir) / "kmeans_iteration.onnx")
            )

            # Convert to numpy for ONNX inference
            vectors_np = vectors.numpy().astype(np.float32)
            k_np = np.array(self.n_partitions, dtype=np.int64)
            seed_np = np.array(42, dtype=np.int64)

            # Initialize centroids using ONNX model
            if verbose:
                logger.info("Initializing centroids with ONNX k-means++")
            centroids_np = init_session.run(
                None, {"embeddings": vectors_np, "k": k_np, "seed": seed_np}
            )[0]

            # Run k-means iterations using ONNX model
            prev_assignments = None
            converged_early = False

            for iteration in range(n_iterations):
                outputs = iter_session.run(
                    None, {"embeddings": vectors_np, "centroids": centroids_np}
                )
                centroids_np, assignments_np, converged_np = outputs

                # Check for convergence
                if prev_assignments is not None:
                    n_changed = (assignments_np != prev_assignments).sum()
                    if verbose and iteration % 10 == 0:
                        logger.info(
                            f"Iteration {iteration}: {n_changed} vectors changed assignment"
                        )

                    if n_changed == 0:
                        if verbose:
                            logger.info(f"Converged after {iteration} iterations")
                        converged_early = True
                        break

                # Check ONNX model convergence
                if converged_np > 0.5:
                    if verbose:
                        logger.info(
                            f"ONNX k-means converged after {iteration + 1} iterations"
                        )
                    converged_early = True
                    break

                prev_assignments = assignments_np.copy()

            if not converged_early and verbose:
                logger.info(f"Completed {n_iterations} iterations")

            # Convert back to torch tensors
            self.centroids.data = torch.from_numpy(centroids_np)
            assignments = torch.from_numpy(assignments_np).long()

        # Build inverted lists
        self._build_inverted_lists(vector_ids, assignments)

        self.assignments = assignments
        self.is_trained = True

        if verbose:
            self._print_partition_stats()

    def _init_centroids_kmeans_plus_plus(self, vectors: torch.Tensor) -> None:
        """Initialize centroids using k-means++ algorithm for better convergence."""
        n_vectors, d = vectors.shape

        # Choose first centroid randomly
        first_idx = torch.randint(0, n_vectors, (1,))
        centroids = vectors[first_idx].clone()

        # Choose remaining centroids with probability proportional to squared distance
        for i in range(1, self.n_partitions):
            # Compute distances to nearest existing centroids
            distances = torch.cdist(vectors, centroids).min(dim=1)[0]

            # Sample next centroid with probability proportional to squared distance
            probabilities = distances**2
            probabilities = probabilities / probabilities.sum()

            # Use multinomial sampling
            next_idx = torch.multinomial(probabilities, 1)
            centroids = torch.cat([centroids, vectors[next_idx]], dim=0)

        self.centroids.data = centroids

    def _assign_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Assign vectors to nearest centroids."""
        # Compute distances to all centroids
        distances = torch.cdist(vectors, self.centroids)  # [N, n_partitions]

        # Find nearest centroid for each vector
        assignments = torch.argmin(distances, dim=1)  # [N]

        return assignments

    def _update_centroids(
        self, vectors: torch.Tensor, assignments: torch.Tensor
    ) -> None:
        """Update centroids based on current assignments."""
        for partition_id in range(self.n_partitions):
            mask = assignments == partition_id
            if mask.sum() > 0:
                # Update centroid to mean of assigned vectors
                self.centroids.data[partition_id] = vectors[mask].mean(dim=0)
            # If no vectors assigned to this partition, keep the current centroid

    def _build_inverted_lists(
        self, vector_ids: torch.Tensor, assignments: torch.Tensor
    ) -> None:
        """Build inverted lists mapping partition_id -> [vector_ids]."""
        self.inverted_lists = {}

        for partition_id in range(self.n_partitions):
            mask = assignments == partition_id
            partition_vector_ids = vector_ids[mask].cpu().numpy().tolist()
            self.inverted_lists[partition_id] = partition_vector_ids

    def _print_partition_stats(self) -> None:
        """Print statistics about partition sizes."""
        partition_sizes = [
            len(self.inverted_lists[i]) for i in range(self.n_partitions)
        ]

        logger.info(f"Partition size statistics:")
        logger.info(f"  Min: {min(partition_sizes)}")
        logger.info(f"  Max: {max(partition_sizes)}")
        logger.info(f"  Mean: {np.mean(partition_sizes):.1f}")
        logger.info(f"  Std: {np.std(partition_sizes):.1f}")

        # Check for empty partitions
        empty_partitions = sum(1 for size in partition_sizes if size == 0)
        if empty_partitions > 0:
            logger.warning(
                f"Found {empty_partitions} empty partitions  (out of {self.n_partitions})"
            )

    def search_partitions(
        self, query_vectors: torch.Tensor, n_probe: int = 1
    ) -> List[List[int]]:
        """
        Find the most relevant partitions for query vectors.

        Args:
            query_vectors: Query vectors of shape [N_queries, d]
            n_probe: Number of partitions to search per query

        Returns:
            List of partition IDs for each query vector
        """
        if not self.is_trained:
            raise RuntimeError("IVF index must be trained before searching")

        n_queries = query_vectors.shape[0]

        # Compute distances to all centroids
        distances = torch.cdist(
            query_vectors, self.centroids
        )  # [N_queries, n_partitions]

        # Find top-k nearest centroids for each query
        _, top_partitions = torch.topk(
            distances, k=min(n_probe, self.n_partitions), dim=1, largest=False
        )

        # Convert to list of lists
        partition_lists = []
        for i in range(n_queries):
            partitions = top_partitions[i].cpu().numpy().tolist()
            partition_lists.append(partitions)

        return partition_lists

    def get_partition_vector_ids(self, partition_id: int) -> List[int]:
        """
        Get vector IDs in a specific partition.

        Args:
            partition_id: ID of the partition

        Returns:
            List of vector IDs in the partition
        """
        if not self.is_trained:
            raise RuntimeError("IVF index must be trained before accessing partitions")

        if partition_id < 0 or partition_id >= self.n_partitions:
            raise ValueError(
                f"partition_id {partition_id} out of range [0, {self.n_partitions})"
            )

        return self.inverted_lists[partition_id]

    def get_partition_stats(self) -> dict:
        """Get statistics about the partitions distribution."""
        if not self.is_trained:
            raise RuntimeError("IVF index must be trained to get statistics")

        partition_sizes = [
            len(self.inverted_lists[i]) for i in range(self.n_partitions)
        ]

        return {
            "n_partitions": self.n_partitions,
            "total_vectors": sum(partition_sizes),
            "min_size": min(partition_sizes),
            "max_size": max(partition_sizes),
            "mean_size": np.mean(partition_sizes),
            "std_size": np.std(partition_sizes),
            "empty_partitions": sum(1 for size in partition_sizes if size == 0),
            "partition_sizes": partition_sizes,
        }

    def save(self, path: Path) -> None:
        """Save the trained IVF index to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained IVF index")

        save_dict = {
            "d": self.d,
            "n_partitions": self.n_partitions,
            "centroids": self.centroids.data.cpu(),
            "inverted_lists": self.inverted_lists,
            "assignments": (
                self.assignments.cpu() if self.assignments is not None else None
            ),
            "is_trained": self.is_trained,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        logger.info(f"IVF index saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "InvertedFileIndex":
        """Load a trained IVF index from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        ivf = cls(save_dict["d"], save_dict["n_partitions"])
        ivf.centroids.data = save_dict["centroids"]
        ivf.inverted_lists = save_dict["inverted_lists"]
        ivf.assignments = save_dict["assignments"]
        ivf.is_trained = save_dict["is_trained"]

        logger.info(f"IVF index loaded from {path}")
        return ivf

    def export_metadata(self) -> dict:
        """Export metadata for browser consumption."""
        if not self.is_trained:
            raise RuntimeError("Cannot export untrained IVF index")

        stats = self.get_partition_stats()

        return {
            "d": self.d,
            "n_partitions": self.n_partitions,
            "is_trained": self.is_trained,
            "partition_stats": stats,
            "centroids_shape": list(self.centroids.shape),
        }
