"""
Combined IVFPQ (Inverted File Index Product Quantization) implementation.

This module loads pre-trained IVF and PQ components to provide a complete IVFPQ system
for approximate nearest neighbor search on large-scale embedding datasets.

The IVFPQ approach:
1. Loads pre-trained IVF index to partition the dataset into partitions
2. Loads pre-trained PQ model to encode vectors within each partition
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
import logging
import json
import pyarrow as pa
import pyarrow.parquet as pq

from .pq import ProductQuantizer
from .ivf import InvertedFileIndex

logger = logging.getLogger(__name__)


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
        if not self.partition_data:
            raise RuntimeError("IVFPQ must encode vectors before searching")

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
        metadata_path = ivfpq_dir / "ivfpq_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.export_metadata(), f, indent=2)

        logger.info(f"IVFPQ export completed successfully to {ivfpq_dir}")

    def _export_partition_files(self, output_dir: Path) -> None:
        """Export each partition as a separate Arrow file."""
        partitions_dir = output_dir / "partitions"
        partitions_dir.mkdir(exist_ok=True)

        for partition_id, partition_data in self.partition_data.items():
            if partition_data["size"] > 0:  # Only export non-empty partitions
                # Convert lists back to numpy arrays for processing
                vector_ids = np.array(partition_data["vector_ids"])
                pq_codes = np.array(partition_data["pq_codes"])

                # Store PQ codes as individual columns (code_0, code_1, ..., code_m-1)
                table_data = {"vector_id": vector_ids}

                # Add each PQ code as a separate column
                for i in range(self.m):
                    table_data[f"code_{i}"] = pq_codes[:, i]

                # Create Arrow table with partition data
                table = pa.table(table_data)

                # Use zero-padded naming convention for proper sorting
                partition_file = partitions_dir / f"partition_{partition_id:04d}.arrow"

                # Write Arrow file
                with pa.OSFile(str(partition_file), "wb") as sink:
                    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                        writer.write_table(table)

        logger.info(
            f"Exported {len([p for p in self.partition_data.values() if p['size'] > 0])} partition files to {partitions_dir}"
        )

    def _export_centroid_index(self, output_dir: Path) -> None:
        """Export centroid index with partition metadata."""
        # Prepare centroid data with metadata
        centroids = self.ivf.centroids.detach().cpu().numpy()

        # Create partition metadata
        partition_ids = []
        partition_sizes = []
        partition_files = []

        for partition_id in range(self.n_partitions):
            partition_ids.append(partition_id)

            if partition_id in self.partition_data:
                partition_sizes.append(self.partition_data[partition_id]["size"])
                partition_files.append(f"partitions/partition_{partition_id:04d}.arrow")
            else:
                # Empty partition
                partition_sizes.append(0)
                partition_files.append(None)

        # Create Arrow table
        # Store centroids as a nested array (list of coordinates)
        centroid_lists = [centroid.tolist() for centroid in centroids]

        table = pa.table(
            {
                "centroid_id": partition_ids,
                "centroid_coords": centroid_lists,
                "partition_size": partition_sizes,
                "partition_file": partition_files,
            }
        )

        # Export centroid index as Arrow file
        centroids_file = output_dir / "centroids.arrow"
        with pa.OSFile(str(centroids_file), "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        logger.info(f"Exported centroid index to {centroids_file}")
