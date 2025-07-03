"""
Combined IVFPQ (Inverted File Index Product Quantization) implementation.

This module combines the IVF and PQ components to provide a complete IVFPQ system
for approximate nearest neighbor search on large-scale embedding datasets.

The IVFPQ approach:
1. Uses IVF to partition the dataset into coarse clusters
2. Applies PQ encoding to vectors within each partition
3. Enables efficient search by constraining queries to relevant partitions
4. Supports export to Arrow/Parquet format for browser consumption

Based on the approach described in:
https://towardsdatascience.com/similarity-search-with-ivfpq-9c6348fd4db3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
import pickle
import logging
import json

from .pq import ProductQuantizer
from .ivf import InvertedFileIndex

logger = logging.getLogger(__name__)


class IVFPQ:
    """
    Combined IVFPQ implementation for approximate nearest neighbor search.

    This class integrates IVF (Inverted File Index) and PQ (Product Quantization)
    to provide efficient similarity search on large-scale embedding datasets.

    The IVFPQ process:
    1. Train IVF index to partition dataset into coarse clusters
    2. Train PQ model to compress vectors within each partition
    3. Encode all vectors using IVF assignments and PQ codes
    4. Enable fast search by selecting relevant partitions first

    Args:
        d: Input vector dimension
        m: Number of PQ subquantizers
        k: Number of centroids per PQ subquantizer
        n_clusters: Number of IVF coarse clusters
    """

    def __init__(self, d: int, m: int, k: int, n_clusters: int):
        self.d = d
        self.m = m
        self.k = k
        self.n_clusters = n_clusters

        # Initialize IVF and PQ components
        self.ivf = InvertedFileIndex(d=d, n_clusters=n_clusters)
        self.pq = ProductQuantizer(d=d, m=m, k=k)

        # Training state
        self.is_trained = False

        # Partition data (populated during training)
        self.partition_data = (
            {}
        )  # partition_id -> {"vector_ids": [...], "pq_codes": [...]}

        logger.info(f"Initialized IVFPQ: d={d}, m={m}, k={k}, n_clusters={n_clusters}")

    def train(
        self,
        vectors: torch.Tensor,
        vector_ids: Optional[torch.Tensor] = None,
        ivf_iterations: int = 50,
        pq_iterations: int = 50,
        verbose: bool = True,
    ) -> None:
        """
        Train the complete IVFPQ model.

        Args:
            vectors: Training vectors of shape [N, d]
            vector_ids: Optional vector IDs of shape [N]. If None, uses indices 0...N-1
            ivf_iterations: Number of k-means iterations for IVF training
            pq_iterations: Number of k-means iterations for PQ training
            verbose: Whether to print training progress
        """
        if verbose:
            logger.info("Starting IVFPQ training...")

        n_vectors = vectors.shape[0]
        device = vectors.device

        if vector_ids is None:
            vector_ids = torch.arange(n_vectors, device=device)

        # Step 1: Train IVF index
        if verbose:
            logger.info("Step 1/3: Training IVF index...")
        self.ivf.train_ivf(
            vectors=vectors,
            vector_ids=vector_ids,
            n_iterations=ivf_iterations,
            verbose=verbose,
        )

        # Step 2: Train PQ model on all vectors
        if verbose:
            logger.info("Step 2/3: Training PQ model...")
        self.pq.train_pq(vectors=vectors, n_iterations=pq_iterations, verbose=verbose)

        # Step 3: Encode all vectors and organize by partition
        if verbose:
            logger.info("Step 3/3: Encoding vectors and organizing partitions...")
        self._encode_and_partition_vectors(vectors, vector_ids, verbose)

        self.is_trained = True

        if verbose:
            logger.info("IVFPQ training completed!")
            self._print_training_summary()

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

        for partition_id in range(self.n_clusters):
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
                f"Organized {total_vectors} vectors into {non_empty_partitions}/{self.n_clusters} partitions"
            )

    def _print_training_summary(self) -> None:
        """Print summary statistics after training."""
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

        logger.info("=== IVFPQ Training Summary ===")
        logger.info(f"Total vectors: {total_vectors:,}")
        logger.info(f"Active partitions: {non_empty_partitions}/{self.n_clusters}")
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
        if not self.is_trained:
            raise RuntimeError("IVFPQ must be trained before searching")

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
        if not self.is_trained:
            raise RuntimeError("IVFPQ must be trained before accessing partition data")

        if partition_id not in self.partition_data:
            raise ValueError(f"Invalid partition ID: {partition_id}")

        return self.partition_data[partition_id]

    def get_all_partition_ids(self) -> List[int]:
        """
        Get all partition IDs.

        Returns:
            List of all partition IDs
        """
        return list(range(self.n_clusters))

    def get_partition_stats(self) -> Dict:
        """
        Get statistics about all partitions.

        Returns:
            Dictionary with partition statistics
        """
        if not self.is_trained:
            raise RuntimeError("IVFPQ must be trained before accessing partition stats")

        partition_sizes = [p["size"] for p in self.partition_data.values()]
        non_empty_partitions = sum(1 for size in partition_sizes if size > 0)
        total_vectors = sum(partition_sizes)

        return {
            "total_partitions": self.n_clusters,
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

    def save(self, output_dir: Path) -> None:
        """
        Save the complete IVFPQ model to disk.

        Args:
            output_dir: Directory to save the model files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual components
        self.ivf.save(output_dir / "ivf_index.pkl")
        self.pq.save(output_dir / "pq_model.pkl")

        # Save partition data
        partition_data_path = output_dir / "partition_data.pkl"
        with open(partition_data_path, "wb") as f:
            pickle.dump(self.partition_data, f)

        # Save metadata
        metadata = {
            "d": self.d,
            "m": self.m,
            "k": self.k,
            "n_clusters": self.n_clusters,
            "is_trained": self.is_trained,
            "stats": self.get_partition_stats() if self.is_trained else None,
        }

        metadata_path = output_dir / "ivfpq_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved IVFPQ model to {output_dir}")

    @classmethod
    def load(cls, model_dir: Path) -> "IVFPQ":
        """
        Load a trained IVFPQ model from disk.

        Args:
            model_dir: Directory containing the saved model files

        Returns:
            Loaded IVFPQ instance
        """
        # Load metadata
        metadata_path = model_dir / "ivfpq_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create instance
        ivfpq = cls(
            d=metadata["d"],
            m=metadata["m"],
            k=metadata["k"],
            n_clusters=metadata["n_clusters"],
        )

        # Load components
        ivfpq.ivf = InvertedFileIndex.load(model_dir / "ivf_index.pkl")
        ivfpq.pq = ProductQuantizer.load(model_dir / "pq_model.pkl")

        # Load partition data
        partition_data_path = model_dir / "partition_data.pkl"
        with open(partition_data_path, "rb") as f:
            ivfpq.partition_data = pickle.load(f)

        ivfpq.is_trained = metadata["is_trained"]

        logger.info(f"Loaded IVFPQ model from {model_dir}")
        return ivfpq

    def export_metadata(self) -> Dict:
        """
        Export complete metadata for browser consumption.

        Returns:
            Dictionary with all metadata needed for browser-side search
        """
        if not self.is_trained:
            raise RuntimeError("IVFPQ must be trained before exporting metadata")

        return {
            "ivfpq": {
                "d": self.d,
                "m": self.m,
                "k": self.k,
                "n_clusters": self.n_clusters,
                "is_trained": self.is_trained,
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
