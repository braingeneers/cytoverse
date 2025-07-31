#!/usr/bin/env python3

"""
Test suite for Inverted File Index (IVF) implementation.

This module tests the core functionality of the IVF index including:
- Training on synthetic and real data
- Partition assignment and search
- Save/load functionality
- Integration with existing PQ module
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from cytoverse.ivfpq.ivf import InvertedFileIndex


class TestIVFBasic:
    """Test basic IVF functionality with synthetic data."""

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
        self.vector_ids = torch.arange(self.n_vectors)

    def test_initialization(self):
        """Test IVF index initialization."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)

        assert ivf.d == self.d
        assert ivf.n_partitions == self.n_partitions
        assert ivf.centroids.shape == (self.n_partitions, self.d)
        assert not ivf.is_trained
        assert ivf.inverted_lists is None

    def test_training(self):
        """Test IVF training process."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)

        # Train the index
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=20)

        # Check training completed
        assert ivf.is_trained
        assert ivf.inverted_lists is not None
        assert ivf.assignments is not None

        # Check all vectors are assigned
        assert len(ivf.assignments) == self.n_vectors
        assert torch.all(ivf.assignments >= 0)
        assert torch.all(ivf.assignments < self.n_partitions)

        # Check inverted lists contain all vectors
        total_assigned = sum(
            len(ivf.inverted_lists[i]) for i in range(self.n_partitions)
        )
        assert total_assigned == self.n_vectors

    def test_get_partition_vector_ids(self):
        """Test getting vectors from specific partitions."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10)

        # Test getting vectors from each partition
        for partition_id in range(self.n_partitions):
            vector_ids = ivf.get_partition_vector_ids(partition_id)
            assert isinstance(vector_ids, list)
            assert all(isinstance(vid, int) for vid in vector_ids)
            assert all(0 <= vid < self.n_vectors for vid in vector_ids)

    def test_partitions_stats(self):
        """Test partition statistics."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10)

        stats = ivf.get_partition_stats()

        assert stats["n_partitions"] == self.n_partitions
        assert stats["total_vectors"] == self.n_vectors
        assert stats["min_size"] >= 0
        assert stats["max_size"] <= self.n_vectors
        assert stats["mean_size"] == self.n_vectors / self.n_partitions
        assert len(stats["partition_sizes"]) == self.n_partitions

    def test_save_load(self):
        """Test saving and loading IVF index."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_ivf.pkl"

            # Save
            ivf.save(save_path)
            assert save_path.exists()

            # Load
            ivf_loaded = InvertedFileIndex.load(save_path)

            # Check loaded index matches original
            assert ivf_loaded.d == ivf.d
            assert ivf_loaded.n_partitions == ivf.n_partitions
            assert ivf_loaded.is_trained == ivf.is_trained
            assert torch.allclose(ivf_loaded.centroids, ivf.centroids)
            assert ivf_loaded.inverted_lists == ivf.inverted_lists

    def test_export_metadata(self):
        """Test metadata export for browser consumption."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10)

        metadata = ivf.export_metadata()

        assert metadata["d"] == self.d
        assert metadata["n_partitions"] == self.n_partitions
        assert metadata["is_trained"] == True
        assert "partition_stats" in metadata
        assert metadata["centroids_shape"] == [self.n_partitions, self.d]

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)

        # Test operations on untrained index
        with pytest.raises(RuntimeError):
            ivf.search_partitions(self.vectors[:1])

        with pytest.raises(RuntimeError):
            ivf.get_partition_vector_ids(0)

        with pytest.raises(RuntimeError):
            ivf.get_partition_stats()

        with pytest.raises(RuntimeError):
            ivf.save(Path("test.pkl"))

        with pytest.raises(RuntimeError):
            ivf.export_metadata()

        # Train and test invalid partition access
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=5)

        with pytest.raises(ValueError):
            ivf.get_partition_vector_ids(-1)

        with pytest.raises(ValueError):
            ivf.get_partition_vector_ids(self.n_partitions)

    def test_mismatched_vector_ids(self):
        """Test error handling for mismatched vector IDs."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)

        # Wrong number of vector IDs
        wrong_ids = torch.arange(self.n_vectors // 2)

        with pytest.raises(ValueError):
            ivf.train_ivf(self.vectors, wrong_ids)


class TestIVFIntegration:
    """Test IVF integration with real-world scenarios."""

    def test_different_partition_counts(self):
        """Test with different numbers of partitions."""
        d = 64
        n_vectors = 500
        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)
        vector_ids = torch.arange(n_vectors)

        for n_partitions in [4, 8, 16, 32]:
            ivf = InvertedFileIndex(d, n_partitions)
            ivf.train_ivf(vectors, vector_ids, n_iterations=10)

            assert ivf.is_trained
            stats = ivf.get_partition_stats()
            assert stats["total_vectors"] == n_vectors

    def test_high_dimensional_data(self):
        """Test with high-dimensional data similar to SCimilarity."""
        d = 512  # SCimilarity embedding dimension
        n_vectors = 200
        n_partitions = 8

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)
        vector_ids = torch.arange(n_vectors)

        ivf = InvertedFileIndex(d, n_partitions)
        ivf.train_ivf(vectors, vector_ids, n_iterations=15)

        assert ivf.is_trained
        assert ivf.centroids.shape == (n_partitions, d)

        # Test search
        query = torch.randn(1, d)
        partitions = ivf.search_partitions(query, n_probe=2)
        assert len(partitions[0]) == 2

    def test_kmeans_plus_plus_initialization(self):
        """Test that k-means++ initialization produces reasonable results."""
        d = 32
        n_vectors = 100
        n_partitions = 8

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        ivf = InvertedFileIndex(d, n_partitions)

        # Initialize centroids
        ivf._init_centroids_kmeans_plus_plus(vectors)

        # Check that centroids are not identical
        centroids = ivf.centroids.data
        for i in range(n_partitions):
            for j in range(i + 1, n_partitions):
                distance = torch.norm(centroids[i] - centroids[j])
                assert distance > 0, f"Centroids {i} and {j} are identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
