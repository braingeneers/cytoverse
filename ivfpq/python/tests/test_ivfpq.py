#!/usr/bin/env python3

"""
Test suite for Inverted File Index (IVF) with Product Quantized (PQ) residual vectors


This module tests the core functionality of the IVF index including:
- Training with scikit-learn on synthetic and real data
- Residual vector computation and storage
- Binary file export/import functionality
- Search with residual-based PQ distance computation
"""

from unittest import result
import pytest
import torch
import numpy as np
from pathlib import Path
import json

from ivfpq.ivfpq import (
    IVFPQ,
    _load_centroids_binary,
    _export_centroids_binary,
    _load_partition_binary,
    _export_partition_binary,
)
from ivfpq.pq import PQ


class TestIVFResidualBasic:
    """Test basic IVF functionality with residual vectors and synthetic data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.d = 128
        self.n_vectors = 1000
        self.n_partitions = 16

        # Create synthetic vectors with some structure
        torch.manual_seed(42)
        np.random.seed(42)

        # Create partitioned data for more realistic testing
        partition_centers = torch.randn(8, self.d) * 2
        vectors = []

        for i in range(self.n_vectors):
            center_idx = i % 8
            noise = torch.randn(self.d) * 0.5
            vector = partition_centers[center_idx] + noise
            vectors.append(vector)

        self.vectors = torch.stack(vectors)

    def test_train_ivfpq(self, tmp_path):
        """Test IVF training with residual vectors."""
        output_dir = tmp_path

        # Train IVF with residual vectors
        ivf = IVFPQ.build(
            vectors=self.vectors,
            output_dir=output_dir,
            max_iterations=20,
        )

        # Verify centroids shape
        assert ivf.centroids.shape == (ivf.n_partitions, self.d)

        # Check files were created
        assert (output_dir / "ivf_centroids.bin").exists()
        assert (output_dir / "ivf_metadata.json").exists()
        assert (output_dir / "pq_metadata.json").exists()
        assert (output_dir / "pq_codebooks.bin").exists()
        assert (output_dir / "pq_encode.onnx").exists()
        assert (output_dir / "pq_distance.onnx").exists()

        # Check partition files
        partitions_dir = output_dir / "partitions"
        assert partitions_dir.exists()

        # Count non-empty partitions
        non_empty_partitions = 0
        for partition_id in range(self.n_partitions):
            partition_file = partitions_dir / f"partition_{partition_id:04d}.bin"
            if partition_file.exists():
                non_empty_partitions += 1

        assert non_empty_partitions > 0

    def test_centroids_binary_io(self, tmp_path):
        """Test binary export/import of centroids."""
        centroids = torch.randn(self.n_partitions, self.d)

        centroids_file = tmp_path / "test_centroids.bin"

        # Export
        _export_centroids_binary(centroids_file, centroids)
        assert centroids_file.exists()

        # Import
        loaded_centroids = _load_centroids_binary(centroids_file)

        # Check they match
        assert torch.allclose(centroids, loaded_centroids, atol=1e-6)

    def test_partition_binary_io(self, tmp_path):
        """Test binary export/import of partition data."""
        # Create test data
        n_vectors = 100
        m = 16
        vector_ids = torch.randint(0, self.n_vectors, (n_vectors,))
        pq_codes = torch.randint(0, 256, (n_vectors, m), dtype=torch.long)

        partition_file = tmp_path / "test_partition.bin"

        # Export
        _export_partition_binary(partition_file, vector_ids, pq_codes)
        assert partition_file.exists()

        # Import
        loaded_vector_ids, loaded_pq_codes = _load_partition_binary(partition_file, m)

        # Check they match
        assert torch.equal(vector_ids, loaded_vector_ids)
        assert torch.equal(pq_codes, loaded_pq_codes)

    def test_forward_method(self):
        """Test the forward method for coarse centroid search."""
        ivf = IVFPQ(self.d, self.n_partitions)
        query_vector = torch.randn(self.d)
        centroids = torch.randn(self.n_partitions, self.d)

        k = 4
        top_k_indices, top_k_distances = ivf.forward(query_vector, centroids, k)

        assert len(top_k_indices) == k
        assert len(top_k_distances) == k
        assert torch.all(
            top_k_distances[:-1] <= top_k_distances[1:]
        )  # Distances should be sorted

    def test_ivf_search(self, tmp_path):
        """Test IVF search with residual vectors using the refactored search method."""
        output_dir = tmp_path

        # Train IVF
        ivf = IVFPQ.build(
            vectors=self.vectors,
            output_dir=output_dir,
            max_iterations=10,
        )

        # Test search with exact vector
        test_idx = 42
        query_vector = self.vectors[test_idx]

        vector_ids, distances = ivf.search(
            query_vector=query_vector,
            model_path=output_dir,
            n_probe=4,
            k_per_partition=10,
        )

        # Should find the exact vector as closest
        assert test_idx in vector_ids
        exact_match_idx = vector_ids.index(test_idx)

        print(f"Distance to exact match (idx {test_idx}): {distances[exact_match_idx]}")

        # The exact match should be reasonably close (allowing for PQ reconstruction error)
        assert (
            distances[exact_match_idx] < 10.0
        ), f"Distance to exact match: {distances[exact_match_idx]}"

        # Test with small perturbation
        noise = torch.randn_like(query_vector) * 0.05
        perturbed_query = query_vector + noise

        perturbed_ids, perturbed_distances = ivf.search(
            query_vector=perturbed_query,
            model_path=output_dir,
            n_probe=4,
            k_per_partition=10,
        )

        # Original vector should still be in top results
        assert (
            test_idx in perturbed_ids
        ), f"Failed to find perturbed match for index {test_idx}"

    def test_save_and_load(self, tmp_path):
        """Test saving and loading the IVFPQ."""
        output_dir = tmp_path

        # Train and save
        ivf = IVFPQ.build(
            vectors=self.vectors,
            output_dir=output_dir,
            max_iterations=10,
        )

        # Load and verify
        loaded_ivf = IVFPQ.load(output_dir)
        assert loaded_ivf.is_trained
        assert torch.allclose(ivf.centroids, loaded_ivf.centroids)
        assert ivf.d == loaded_ivf.d
        assert ivf.n_partitions == loaded_ivf.n_partitions


class TestIVFResidualIntegration:
    """Integration tests for IVF with residual vectors."""

    def test_different_vector_dimensions(self, tmp_path):
        """Test with different vector dimensions."""
        for d in [64, 128, 256]:
            n_vectors = 200

            torch.manual_seed(42)
            vectors = torch.randn(n_vectors, d)

            test_output_dir = tmp_path / f"test_dim_{d}"
            test_output_dir.mkdir(exist_ok=True)

            ivf = IVFPQ.build(
                vectors=vectors,
                output_dir=test_output_dir,
                max_iterations=5,
            )

            assert ivf.centroids.shape == (ivf.n_partitions, d)

    def test_different_partition_counts(self, tmp_path):
        """Test with different numbers of partitions."""
        d = 64
        n_vectors = 300
        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        for n_partitions in [4, 8, 16]:
            test_output_dir = tmp_path / f"test_partitions_{n_partitions}"
            test_output_dir.mkdir(exist_ok=True)

            ivf = IVFPQ.build(
                vectors=vectors,
                output_dir=test_output_dir,
                n_partitions=n_partitions,
                max_iterations=5,
            )

            assert ivf.centroids.shape == (n_partitions, d)

            # Test metadata
            metadata_file = test_output_dir / "ivf_metadata.json"
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert metadata["n_partitions"] == n_partitions
            assert metadata["d"] == d
            assert metadata["total_vectors"] == n_vectors

    def test_search_accuracy(self, tmp_path):
        """Test search accuracy with perturbed queries."""
        d = 64
        n_vectors = 500
        n_partitions = 8

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        output_dir = tmp_path

        # Train
        ivf = IVFPQ.build(
            vectors=vectors,
            output_dir=output_dir,
            max_iterations=15,
        )

        # Test with multiple queries
        test_indices = [10, 50, 100, 200, 300]

        for test_idx in test_indices:
            # Test exact query
            query_vector = vectors[test_idx]

            vector_ids, distances = ivf.search(
                query_vector=query_vector,
                model_path=output_dir,
                n_probe=4,
                k_per_partition=10,
            )

            # Should find the exact vector
            assert (
                test_idx in vector_ids
            ), f"Failed to find exact match for index {test_idx}"

            # Test with small perturbation
            noise = torch.randn_like(query_vector) * 0.05
            perturbed_query = query_vector + noise

            perturbed_ids, perturbed_distances = ivf.search(
                query_vector=perturbed_query,
                model_path=output_dir,
                n_probe=4,
                k_per_partition=10,
            )

            # Original vector should still be in top results
            assert (
                test_idx in perturbed_ids
            ), f"Failed to find perturbed match for index {test_idx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
