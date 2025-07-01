"""
Inverted File Index (IVF) implementation for coarse quantization.

This module implements the IVF component of IVFPQ for partitioning large datasets
into manageable chunks using coarse quantization. Each vector is assigned to one
of n_clusters partitions based on proximity to coarse centroids.

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

logger = logging.getLogger(__name__)


class InvertedFileIndex(nn.Module):
    """
    Inverted File Index for coarse quantization and dataset partitioning.

    The IVF index partitions vectors into clusters using k-means clustering,
    creating an inverted index where each cluster contains a list of vector IDs.
    This enables efficient nearest neighbor search by constraining search to
    the most relevant partitions.

    Args:
        d: Input vector dimension
        n_clusters: Number of coarse clusters (partitions)

    Training Process:
    1. Run k-means clustering on the dataset to find coarse centroids
    2. Assign each vector to its nearest coarse centroid
    3. Build inverted lists mapping cluster_id -> [vector_ids]

    Search Process:
    1. Find nearest coarse centroids for query vector
    2. Search only within those partitions
    3. Rank results using refined distance computation
    """

    def __init__(self, d: int, n_clusters: int):
        super().__init__()

        self.d = d  # Input dimension
        self.n_clusters = n_clusters  # Number of coarse clusters

        # Coarse centroids: [n_clusters, d]
        self.coarse_centroids = nn.Parameter(
            torch.randn(n_clusters, d), requires_grad=False
        )

        # Inverted lists: cluster_id -> list of vector indices
        # This will be populated during training
        self.inverted_lists = None

        # Vector assignments: vector_id -> cluster_id
        # This helps with updates and debugging
        self.assignments = None

        # Whether the index has been trained
        self.is_trained = False

        logger.info(
            f"Initialized InvertedFileIndex: d={self.d}, n_clusters={self.n_clusters}"
        )

    def train_ivf(
        self,
        vectors: torch.Tensor,
        vector_ids: Optional[torch.Tensor] = None,
        n_iterations: int = 50,
        verbose: bool = True,
    ) -> None:
        """
        Train the IVF index using k-means clustering.

        Args:
            vectors: Training vectors of shape [N, d]
            vector_ids: Optional vector IDs of shape [N]. If None, uses indices 0...N-1
            n_iterations: Number of k-means iterations
            verbose: Whether to print training progress
        """
        n_vectors = vectors.shape[0]
        device = vectors.device

        if vector_ids is None:
            vector_ids = torch.arange(n_vectors, device=device)

        if len(vector_ids) != n_vectors:
            raise ValueError(
                f"vector_ids length {len(vector_ids)} must match vectors length {n_vectors}"
            )

        if verbose:
            logger.info(
                f"Training IVF index on {n_vectors} vectors with {self.n_clusters} clusters"
            )

        # Initialize centroids using k-means++
        self._init_centroids_kmeans_plus_plus(vectors)

        # K-means iterations
        prev_assignments = None

        for iteration in range(n_iterations):
            # Assign vectors to nearest centroids
            assignments = self._assign_vectors(vectors)

            # Check for convergence
            if prev_assignments is not None:
                n_changed = (assignments != prev_assignments).sum().item()
                if verbose and iteration % 10 == 0:
                    logger.info(
                        f"Iteration {iteration}: {n_changed} vectors changed assignment"
                    )

                if n_changed == 0:
                    if verbose:
                        logger.info(f"Converged after {iteration} iterations")
                    break

            # Update centroids
            self._update_centroids(vectors, assignments)
            prev_assignments = assignments

        # Build inverted lists
        self._build_inverted_lists(vector_ids, assignments)

        self.assignments = assignments
        self.is_trained = True

        if verbose:
            self._print_cluster_stats()

    def _init_centroids_kmeans_plus_plus(self, vectors: torch.Tensor) -> None:
        """Initialize centroids using k-means++ algorithm for better convergence."""
        n_vectors, d = vectors.shape
        device = vectors.device

        # Choose first centroid randomly
        first_idx = torch.randint(0, n_vectors, (1,), device=device)
        centroids = vectors[first_idx].clone()

        # Choose remaining centroids with probability proportional to squared distance
        for i in range(1, self.n_clusters):
            # Compute distances to nearest existing centroids
            distances = torch.cdist(vectors, centroids).min(dim=1)[0]

            # Sample next centroid with probability proportional to squared distance
            probabilities = distances**2
            probabilities = probabilities / probabilities.sum()

            # Use multinomial sampling
            next_idx = torch.multinomial(probabilities, 1)
            centroids = torch.cat([centroids, vectors[next_idx]], dim=0)

        self.coarse_centroids.data = centroids

    def _assign_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Assign vectors to nearest coarse centroids."""
        # Compute distances to all coarse centroids
        distances = torch.cdist(vectors, self.coarse_centroids)  # [N, n_clusters]

        # Find nearest centroid for each vector
        assignments = torch.argmin(distances, dim=1)  # [N]

        return assignments

    def _update_centroids(
        self, vectors: torch.Tensor, assignments: torch.Tensor
    ) -> None:
        """Update coarse centroids based on current assignments."""
        for cluster_id in range(self.n_clusters):
            mask = assignments == cluster_id
            if mask.sum() > 0:
                # Update centroid to mean of assigned vectors
                self.coarse_centroids.data[cluster_id] = vectors[mask].mean(dim=0)
            # If no vectors assigned to this cluster, keep the current centroid

    def _build_inverted_lists(
        self, vector_ids: torch.Tensor, assignments: torch.Tensor
    ) -> None:
        """Build inverted lists mapping cluster_id -> [vector_ids]."""
        self.inverted_lists = {}

        for cluster_id in range(self.n_clusters):
            mask = assignments == cluster_id
            cluster_vector_ids = vector_ids[mask].cpu().numpy().tolist()
            self.inverted_lists[cluster_id] = cluster_vector_ids

    def _print_cluster_stats(self) -> None:
        """Print statistics about cluster sizes."""
        cluster_sizes = [len(self.inverted_lists[i]) for i in range(self.n_clusters)]

        logger.info(f"Cluster size statistics:")
        logger.info(f"  Min: {min(cluster_sizes)}")
        logger.info(f"  Max: {max(cluster_sizes)}")
        logger.info(f"  Mean: {np.mean(cluster_sizes):.1f}")
        logger.info(f"  Std: {np.std(cluster_sizes):.1f}")

        # Check for empty clusters
        empty_clusters = sum(1 for size in cluster_sizes if size == 0)
        if empty_clusters > 0:
            logger.warning(f"Found {empty_clusters} empty clusters")

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

        # Compute distances to all coarse centroids
        distances = torch.cdist(
            query_vectors, self.coarse_centroids
        )  # [N_queries, n_clusters]

        # Find top-k nearest centroids for each query
        _, top_clusters = torch.topk(
            distances, k=min(n_probe, self.n_clusters), dim=1, largest=False
        )

        # Convert to list of lists
        partition_lists = []
        for i in range(n_queries):
            partitions = top_clusters[i].cpu().numpy().tolist()
            partition_lists.append(partitions)

        return partition_lists

    def get_partition_vectors(self, partition_id: int) -> List[int]:
        """
        Get vector IDs in a specific partition.

        Args:
            partition_id: ID of the partition

        Returns:
            List of vector IDs in the partition
        """
        if not self.is_trained:
            raise RuntimeError("IVF index must be trained before accessing partitions")

        if partition_id < 0 or partition_id >= self.n_clusters:
            raise ValueError(
                f"partition_id {partition_id} out of range [0, {self.n_clusters})"
            )

        return self.inverted_lists[partition_id]

    def get_cluster_stats(self) -> dict:
        """Get statistics about the cluster distribution."""
        if not self.is_trained:
            raise RuntimeError("IVF index must be trained to get statistics")

        cluster_sizes = [len(self.inverted_lists[i]) for i in range(self.n_clusters)]

        return {
            "n_clusters": self.n_clusters,
            "total_vectors": sum(cluster_sizes),
            "min_size": min(cluster_sizes),
            "max_size": max(cluster_sizes),
            "mean_size": np.mean(cluster_sizes),
            "std_size": np.std(cluster_sizes),
            "empty_clusters": sum(1 for size in cluster_sizes if size == 0),
            "cluster_sizes": cluster_sizes,
        }

    def save(self, path: Path) -> None:
        """Save the trained IVF index to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained IVF index")

        save_dict = {
            "d": self.d,
            "n_clusters": self.n_clusters,
            "coarse_centroids": self.coarse_centroids.data.cpu(),
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

        ivf = cls(save_dict["d"], save_dict["n_clusters"])
        ivf.coarse_centroids.data = save_dict["coarse_centroids"]
        ivf.inverted_lists = save_dict["inverted_lists"]
        ivf.assignments = save_dict["assignments"]
        ivf.is_trained = save_dict["is_trained"]

        logger.info(f"IVF index loaded from {path}")
        return ivf

    def export_metadata(self) -> dict:
        """Export metadata for browser consumption."""
        if not self.is_trained:
            raise RuntimeError("Cannot export untrained IVF index")

        stats = self.get_cluster_stats()

        return {
            "d": self.d,
            "n_clusters": self.n_clusters,
            "is_trained": self.is_trained,
            "cluster_stats": stats,
            "coarse_centroids_shape": list(self.coarse_centroids.shape),
        }
