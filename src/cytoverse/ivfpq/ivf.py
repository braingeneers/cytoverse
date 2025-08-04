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
from typing import Optional, Tuple, List, Union, Dict
from pathlib import Path
import pickle
import logging
import json
import onnxruntime as ort
import tempfile
from cytoverse.kmeans import export_kmeans_models, run_python_kmeans
import pyarrow as pa
from tqdm import tqdm
from .pq import ProductQuantizer

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
        use_onnx_kmeans: bool = True,
    ) -> None:
        """
        Train the IVF index using either Python or ONNX k-means clustering.

        Args:
            vectors: Training vectors of shape [N, d]
            vector_ids: Optional vector IDs of shape [N]. If None, uses indices 0...N-1
            n_iterations: Number of k-means iterations
            verbose: Whether to print training progress
            use_onnx_kmeans: If True, use ONNX k-means. If False, use Python k-means.
        """
        n_vectors = vectors.shape[0]

        if len(vector_ids) != n_vectors:
            raise ValueError(
                f"vector_ids length {len(vector_ids)} must match vectors length {n_vectors}"
            )

        kmeans_type = "ONNX" if use_onnx_kmeans else "Python"
        if verbose:
            logger.info(
                f"Training IVF index on {n_vectors} vectors with {self.n_partitions} partitions using {kmeans_type} k-means"
            )
            logger.info(f"Vector dimensions: {vectors.shape}")
            logger.info(
                f"Memory usage: ~{vectors.numel() * 4 / 1e6:.1f} MB for input vectors"
            )

        if use_onnx_kmeans:
            # Export k-means ONNX models to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                export_kmeans_models(tmpdir)

                # Load ONNX k-means models
                init_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_init.onnx"))
                iter_session = ort.InferenceSession(
                    str(Path(tmpdir) / "kmeans_iteration.onnx")
                )

                # Convert to numpy for ONNX inference - minimize memory usage
                vectors_np = vectors.detach().cpu().numpy().astype(np.float32)
                k_np = np.array(self.n_partitions, dtype=np.int64)
                seed_np = np.array(42, dtype=np.int64)

                # Initialize centroids using ONNX model
                if verbose:
                    logger.info("Initializing centroids with ONNX k-means++")
                centroids_np = init_session.run(
                    None, {"embeddings": vectors_np, "k": k_np, "seed": seed_np}
                )[0]

                if verbose:
                    logger.info(
                        f"Initialized {centroids_np.shape[0]} centroids with shape {centroids_np.shape}"
                    )
                    logger.info(f"Starting {n_iterations} k-means iterations...")

                # Run k-means iterations using ONNX model
                prev_assignments = None
                converged_early = False

                # Progress bar for k-means iterations
                iteration_pbar = tqdm(
                    range(n_iterations),
                    desc="IVF k-means clustering",
                    leave=True,
                )

                for iteration in iteration_pbar:
                    # Force garbage collection before each iteration to free ONNX runtime memory
                    import gc

                    gc.collect()

                    if verbose and iteration == 0:
                        logger.info(
                            f"Starting k-means iteration {iteration + 1}/{n_iterations}"
                        )

                    outputs = iter_session.run(
                        None, {"embeddings": vectors_np, "centroids": centroids_np}
                    )
                    centroids_np, assignments_np, converged_np = outputs

                    if verbose and iteration == 0:
                        logger.info(
                            f"First iteration completed, got {len(np.unique(assignments_np))} active clusters"
                        )

                    # Check for convergence
                    if prev_assignments is not None:
                        n_changed = (assignments_np != prev_assignments).sum()

                        # Update progress bar with convergence info
                        iteration_pbar.set_postfix(
                            vectors_changed=n_changed,
                            iteration=f"{iteration + 1}/{n_iterations}",
                            active_clusters=len(np.unique(assignments_np)),
                        )

                        if (
                            verbose and iteration % 5 == 0
                        ):  # Log every 5 iterations instead of 10
                            logger.info(
                                f"Iteration {iteration + 1}: {n_changed} vectors changed assignment, {len(np.unique(assignments_np))} active clusters"
                            )

                        if n_changed == 0:
                            iteration_pbar.set_description(
                                "IVF k-means clustering (converged)"
                            )
                            iteration_pbar.close()
                            if verbose:
                                logger.info(f"Converged after {iteration} iterations")
                            converged_early = True
                            break

                    # Check ONNX model convergence
                    if converged_np > 0.5:
                        iteration_pbar.set_description(
                            "IVF k-means clustering (ONNX converged)"
                        )
                        iteration_pbar.close()
                        if verbose:
                            logger.info(
                                f"ONNX k-means converged after {iteration + 1} iterations"
                            )
                        converged_early = True
                        break

                    # Avoid memory leak: reuse array instead of copying
                    if prev_assignments is None:
                        prev_assignments = assignments_np.copy()
                    else:
                        prev_assignments[:] = assignments_np

                if not iteration_pbar.disable:
                    iteration_pbar.close()

                if not converged_early and verbose:
                    logger.info(f"Completed {n_iterations} iterations")

                # Convert back to torch tensors
                self.centroids.data = torch.from_numpy(centroids_np)
                assignments = torch.from_numpy(assignments_np).long()

                # Clean up large numpy arrays to free memory
                del vectors_np
                del centroids_np
                del assignments_np
                if prev_assignments is not None:
                    del prev_assignments

                # Force garbage collection to free ONNX runtime memory
                import gc

                gc.collect()
        else:
            # Use Python k-means implementation
            if verbose:
                logger.info("Running Python k-means clustering")
            
            centroids, assignments = run_python_kmeans(
                vectors,
                k=self.n_partitions,
                max_iterations=n_iterations,
                seed=42,
                verbose=verbose
            )
            
            self.centroids.data = centroids

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


class IVFPQ:
    """
    Combined IVFPQ implementation for approximate nearest neighbor search.

    This class loads pre-trained IVF (Inverted File Index) and PQ (Product Quantization)
    models to provide efficient similarity search on large-scale embedding datasets.

    The IVFPQ process:
    1. Load pre-trained IVF index from model_path/ivf/model.pkl
    2. Load pre-trained PQ model from model_path/pq/model.pkl
    3. Encode all vectors using IVF assignments and PQ codes
    4. Enable fast search by selecting relevant partitions first

    Args:
        model_path: Path to directory containing pre-trained models
    """

    def __init__(self, model_path: Path):
        model_path = Path(model_path)

        # Load pre-trained IVF and PQ components
        ivf_path = model_path / "ivf" / "model.pkl"
        pq_path = model_path / "pq" / "model.pkl"

        if not ivf_path.exists():
            raise FileNotFoundError(f"IVF model not found at {ivf_path}")
        if not pq_path.exists():
            raise FileNotFoundError(f"PQ model not found at {pq_path}")

        self.ivf = InvertedFileIndex.load(ivf_path)
        self.pq = ProductQuantizer.load(pq_path)

        # Extract dimensions from loaded models
        self.d = self.ivf.d
        self.m = self.pq.m
        self.k = self.pq.k
        self.n_partitions = self.ivf.n_partitions

        # Partition data (populated during encode_vectors)
        self.partition_data = (
            {}
        )  # partition_id -> {"vector_ids": [...], "pq_codes": [...]}

        logger.info(
            f"Loaded IVFPQ from {model_path}: d={self.d}, m={self.m}, k={self.k}, n_partitions={self.n_partitions}"
        )

    def encode_vectors(
        self,
        vectors: torch.Tensor,
        vector_ids: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> None:
        """
        Encode vectors using the pre-trained IVFPQ model.

        Args:
            vectors: Input vectors of shape [N, d]
            vector_ids: Optional vector IDs of shape [N]. If None, uses indices 0...N-1
            verbose: Whether to print encoding progress
        """
        if verbose:
            logger.info("Encoding vectors with IVFPQ...")

        n_vectors = vectors.shape[0]
        device = vectors.device

        if vector_ids is None:
            vector_ids = torch.arange(n_vectors, device=device)

        # Encode vectors and organize by partition
        self._encode_and_partition_vectors(vectors, vector_ids, verbose)

        if verbose:
            logger.info("Vector encoding completed!")
            self._print_encoding_summary()

    def _encode_and_partition_vectors(
        self, vectors: torch.Tensor, vector_ids: torch.Tensor, verbose: bool = True
    ) -> None:
        """
        Encode all vectors with PQ and organize by IVF partition.

        Args:
            vectors: Input vectors of shape [N, d]
            vector_ids: Vector IDs of shape [N]
            verbose: Whether to print progress
        """
        # Get IVF assignments for all vectors
        assignments = self.ivf._assign_vectors(vectors)

        # Encode all vectors with PQ
        pq_codes = self.pq(vectors)

        # Organize by partition
        self.partition_data = {}

        for partition_id in range(self.n_partitions):
            # Find vectors assigned to this partition
            mask = assignments == partition_id

            if mask.sum() > 0:
                partition_vector_ids = vector_ids[mask].cpu().numpy()
                partition_pq_codes = pq_codes[mask].cpu().numpy()

                self.partition_data[partition_id] = {
                    "vector_ids": partition_vector_ids.tolist(),
                    "pq_codes": partition_pq_codes.tolist(),
                    "size": len(partition_vector_ids),
                }
            else:
                # Empty partition
                self.partition_data[partition_id] = {
                    "vector_ids": [],
                    "pq_codes": [],
                    "size": 0,
                }

        if verbose:
            non_empty_partitions = sum(
                1 for p in self.partition_data.values() if p["size"] > 0
            )
            total_vectors = sum(p["size"] for p in self.partition_data.values())
            logger.info(
                f"Organized {total_vectors} vectors into {non_empty_partitions}/{self.n_partitions} partitions"
            )

    def _print_encoding_summary(self) -> None:
        """Print summary statistics after encoding."""
        total_vectors = sum(p["size"] for p in self.partition_data.values())
        non_empty_partitions = sum(
            1 for p in self.partition_data.values() if p["size"] > 0
        )

        partition_sizes = [
            p["size"] for p in self.partition_data.values() if p["size"] > 0
        ]
        avg_partition_size = np.mean(partition_sizes) if partition_sizes else 0

        # Calculate compression ratio
        original_size_bits = total_vectors * self.d * 32  # 32-bit floats
        compressed_size_bits = total_vectors * self.m * 8  # 8-bit codes
        compression_ratio = (
            original_size_bits / compressed_size_bits if compressed_size_bits > 0 else 0
        )

        logger.info("=== IVFPQ Encoding Summary ===")
        logger.info(f"Total vectors: {total_vectors:,}")
        logger.info(f"Active partitions: {non_empty_partitions}/{self.n_partitions}")
        logger.info(f"Average partition size: {avg_partition_size:.1f}")
        logger.info(f"Compression ratio: {compression_ratio:.1f}x")
        logger.info(f"Memory per vector: {self.m} bytes (PQ codes)")

    def search_partitions(
        self, query_vector: torch.Tensor, n_probe: int = 4
    ) -> List[int]:
        """
        Select the most promising partitions for a query vector.

        Args:
            query_vector: Query vector of shape [d]
            n_probe: Number of partitions to select

        Returns:
            List of partition IDs sorted by relevance
        """
        # Ensure query_vector is 2D for IVF search_partitions method
        if query_vector.dim() == 1:
            query_vector = query_vector.unsqueeze(0)  # Add batch dimension

        # IVF search_partitions returns results for each query vector
        # Since we only have one query, take the first result
        partition_lists = self.ivf.search_partitions(query_vector, n_probe)
        return partition_lists[0]  # Return list for the single query

    def get_partition_data(self, partition_id: int) -> Dict:
        """
        Get the data for a specific partition.

        Args:
            partition_id: Partition ID

        Returns:
            Dictionary with vector_ids, pq_codes, and size
        """
        if not self.partition_data:
            raise RuntimeError(
                "IVFPQ must encode vectors before accessing partition data"
            )

        if partition_id not in self.partition_data:
            raise ValueError(f"Invalid partition ID: {partition_id}")

        return self.partition_data[partition_id]

    def get_all_partition_ids(self) -> List[int]:
        """
        Get all partition IDs.

        Returns:
            List of all partition IDs
        """
        return list(range(self.n_partitions))

    def get_partition_stats(self) -> Dict:
        """
        Get statistics about all partitions.

        Returns:
            Dictionary with partition statistics
        """
        if not self.partition_data:
            raise RuntimeError(
                "IVFPQ must encode vectors before accessing partition stats"
            )

        partition_sizes = [p["size"] for p in self.partition_data.values()]
        non_empty_partitions = sum(1 for size in partition_sizes if size > 0)
        total_vectors = sum(partition_sizes)

        return {
            "total_partitions": self.n_partitions,
            "non_empty_partitions": non_empty_partitions,
            "total_vectors": total_vectors,
            "partition_sizes": partition_sizes,
            "avg_partition_size": (
                np.mean([s for s in partition_sizes if s > 0])
                if non_empty_partitions > 0
                else 0
            ),
            "min_partition_size": (
                min([s for s in partition_sizes if s > 0])
                if non_empty_partitions > 0
                else 0
            ),
            "max_partition_size": max(partition_sizes) if partition_sizes else 0,
        }

    def export_metadata(self) -> Dict:
        """
        Export complete metadata for browser consumption.

        Returns:
            Dictionary with all metadata needed for browser-side search
        """
        if not self.partition_data:
            raise RuntimeError("IVFPQ must encode vectors before exporting metadata")

        return {
            "ivfpq": {
                "d": self.d,
                "m": self.m,
                "k": self.k,
                "n_partitions": self.n_partitions,
                "stats": self.get_partition_stats(),
            },
            "ivf": self.ivf.export_metadata(),
            "pq": {
                "d": self.d,
                "m": self.m,
                "k": self.k,
                "d_sub": self.pq.d_sub,
                "is_trained": self.pq.is_trained,
            },
        }

    def export(self, output_dir: Path) -> None:
        """
        Export complete IVFPQ model for browser consumption using Arrow format.

        This creates:
        1. Individual partition files (partitions/partition_*.arrow) with PQ codes and vector IDs
        2. Centroid index file (centroids.arrow) with partition metadata
        3. Browser-compatible metadata files

        Args:
            output_dir: Directory to save browser assets
        """
        if not self.partition_data:
            raise RuntimeError("IVFPQ must encode vectors before exporting")

        # Create ivfpq subdirectory
        ivfpq_dir = output_dir / "ivfpq"
        ivfpq_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting IVFPQ to {ivfpq_dir}")

        # 1. Export individual partition files
        self._export_partition_files(ivfpq_dir)

        # 2. Export centroid index with partition metadata
        self._export_centroid_index(ivfpq_dir)

        # 4. Export complete metadata
        metadata_path = ivfpq_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.export_metadata(), f, indent=2)

        logger.info(f"IVFPQ export completed successfully to {ivfpq_dir}")

    def _export_partition_files(self, output_dir: Path) -> None:
        """Export each partition as a custom binary file optimized for browser loading."""
        partitions_dir = output_dir / "partitions"
        partitions_dir.mkdir(exist_ok=True)

        for partition_id, partition_data in self.partition_data.items():
            if partition_data["size"] > 0:  # Only export non-empty partitions
                # Convert lists back to numpy arrays for processing
                vector_ids = np.array(partition_data["vector_ids"], dtype=np.uint32)
                pq_codes = np.array(partition_data["pq_codes"], dtype=np.uint8)
                
                num_vectors = len(vector_ids)
                
                # Use zero-padded naming convention for proper sorting
                partition_file = partitions_dir / f"partition_{partition_id:04d}.bin"
                
                # Write custom binary format:
                # Header: [num_vectors:uint32, m:uint32]
                # Data: [vector_id:uint32, pq_codes:uint8[m]] × num_vectors
                with open(partition_file, 'wb') as f:
                    # Write header
                    f.write(np.uint32(num_vectors).tobytes())
                    f.write(np.uint32(self.m).tobytes())
                    
                    # Write data in interleaved format for optimal access
                    for i in range(num_vectors):
                        f.write(vector_ids[i].tobytes())
                        f.write(pq_codes[i].tobytes())

        logger.info(
            f"Exported {len([p for p in self.partition_data.values() if p['size'] > 0])} partition files to {partitions_dir}"
        )

    def _export_centroid_index(self, output_dir: Path) -> None:
        """Export centroid index in custom binary format."""
        # Prepare centroid data
        centroids = self.ivf.centroids.detach().cpu().numpy().astype(np.float32)
        n_partitions, d = centroids.shape

        # Export centroids as binary file
        centroids_file = output_dir / "centroids.bin"
        with open(centroids_file, 'wb') as f:
            # Write header: [n_partitions:uint32, d:uint32]
            f.write(np.uint32(n_partitions).tobytes())
            f.write(np.uint32(d).tobytes())
            
            # Write centroids data: [centroid:float32[d]] × n_partitions
            centroids.tobytes() # This is already in row-major order
            f.write(centroids.tobytes())

        logger.info(f"Exported centroid index to {centroids_file}")

    def search(
        self,
        query_vector: torch.Tensor,
        k: int = 4,
        n_probe: int = 4,
        labels: Optional[List[str]] = None,
        model_path: Optional[Path] = None,
    ) -> Tuple[List[int], List[int], List[float], List[str]]:
        """
        Search for the k nearest neighbors to a query vector.

        This method can work with either in-memory partition data or on-disk partitions.
        If model_path is provided, it will load partitions from disk dynamically.

        Args:
            query_vector: Query vector of shape [d]
            k: Number of nearest neighbors to return
            n_probe: Number of partitions to search
            labels: Optional list of labels corresponding to vector IDs
            model_path: Optional path to load partitions from disk

        Returns:
            Tuple of (vector_ids, partition_ids, distances, labels_found)
        """
        if query_vector.dim() == 1:
            query_vector = query_vector.unsqueeze(0)  # Add batch dimension

        # Get the most promising partitions
        selected_partitions = self.search_partitions(query_vector.squeeze(0), n_probe)

        candidates = []  # List of (distance, vector_id, partition_id)

        for partition_id in selected_partitions:
            if model_path is not None:
                # Load partition from disk
                partition_data = self._load_partition_from_disk(
                    partition_id, model_path
                )
            else:
                # Use in-memory partition data
                partition_data = self.get_partition_data(partition_id)

            if partition_data["size"] == 0:
                continue

            # Convert PQ codes back to tensors
            pq_codes = torch.tensor(partition_data["pq_codes"], dtype=torch.long)
            vector_ids = partition_data["vector_ids"]

            # Decode PQ codes to approximate vectors
            reconstructed_vectors = self.pq.decode(pq_codes)

            # Compute distances to query
            query_expanded = query_vector.expand(reconstructed_vectors.shape[0], -1)
            distances = torch.norm(reconstructed_vectors - query_expanded, dim=1)

            # Add candidates from this partition
            for i, (dist, vec_id) in enumerate(zip(distances, vector_ids)):
                candidates.append((dist.item(), vec_id, partition_id))

        # Sort by distance and take top k
        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:k]

        # Extract results
        vector_ids = [c[1] for c in top_candidates]
        partition_ids = [c[2] for c in top_candidates]
        distances = [c[0] for c in top_candidates]

        # Add labels if provided
        labels_found = []
        if labels is not None:
            for vec_id in vector_ids:
                if 0 <= vec_id < len(labels):
                    labels_found.append(labels[vec_id])
                else:
                    labels_found.append("Unknown")
        else:
            labels_found = [""] * len(vector_ids)

        return vector_ids, partition_ids, distances, labels_found

    def _load_partition_from_disk(self, partition_id: int, model_path: Path) -> Dict:
        """
        Load a partition from disk using custom binary format.

        Args:
            partition_id: Partition ID to load
            model_path: Path to the model directory

        Returns:
            Dictionary with vector_ids, pq_codes, and size
        """
        partitions_dir = model_path / "ivfpq" / "partitions"
        partition_file = partitions_dir / f"partition_{partition_id:04d}.bin"

        if not partition_file.exists():
            # Return empty partition
            return {"vector_ids": [], "pq_codes": [], "size": 0}

        # Load custom binary format
        with open(partition_file, 'rb') as f:
            # Read header
            num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            m = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            
            # Read interleaved data
            vector_ids = []
            pq_codes = []
            
            for i in range(num_vectors):
                vector_id = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                codes = np.frombuffer(f.read(m), dtype=np.uint8).tolist()
                
                vector_ids.append(int(vector_id))
                pq_codes.append(codes)

        return {
            "vector_ids": vector_ids,
            "pq_codes": pq_codes,
            "size": len(vector_ids),
        }
