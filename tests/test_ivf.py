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
import shutil

import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cellspace.ivfpq.ivf import InvertedFileIndex


class TestIVFBasic:
    """Test basic IVF functionality with synthetic data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.d = 128
        self.n_vectors = 1000
        self.n_clusters = 16

        # Create synthetic vectors with some structure
        torch.manual_seed(42)
        np.random.seed(42)

        # Create clustered data for more realistic testing
        cluster_centers = torch.randn(8, self.d) * 2
        vectors = []

        for i in range(self.n_vectors):
            center_idx = i % 8
            noise = torch.randn(self.d) * 0.5
            vector = cluster_centers[center_idx] + noise
            vectors.append(vector)

        self.vectors = torch.stack(vectors)
        self.vector_ids = torch.arange(self.n_vectors)

    def test_initialization(self):
        """Test IVF index initialization."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)

        assert ivf.d == self.d
        assert ivf.n_clusters == self.n_clusters
        assert ivf.coarse_centroids.shape == (self.n_clusters, self.d)
        assert not ivf.is_trained
        assert ivf.inverted_lists is None

    def test_training(self):
        """Test IVF training process."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)

        # Train the index
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=20, verbose=False)

        # Check training completed
        assert ivf.is_trained
        assert ivf.inverted_lists is not None
        assert ivf.assignments is not None

        # Check all vectors are assigned
        assert len(ivf.assignments) == self.n_vectors
        assert torch.all(ivf.assignments >= 0)
        assert torch.all(ivf.assignments < self.n_clusters)

        # Check inverted lists contain all vectors
        total_assigned = sum(len(ivf.inverted_lists[i]) for i in range(self.n_clusters))
        assert total_assigned == self.n_vectors

    def test_training_without_vector_ids(self):
        """Test training without explicit vector IDs."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)

        # Train without vector_ids (should use 0...N-1)
        ivf.train_ivf(self.vectors, n_iterations=10, verbose=False)

        assert ivf.is_trained

        # Check that vector IDs are 0...N-1
        all_vector_ids = set()
        for cluster_id in range(self.n_clusters):
            all_vector_ids.update(ivf.inverted_lists[cluster_id])

        expected_ids = set(range(self.n_vectors))
        assert all_vector_ids == expected_ids

    def test_search_partitions(self):
        """Test partition search functionality."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10, verbose=False)

        # Test single query
        query = self.vectors[:1]  # Use first training vector as query
        partitions = ivf.search_partitions(query, n_probe=1)

        assert len(partitions) == 1
        assert len(partitions[0]) == 1
        assert 0 <= partitions[0][0] < self.n_clusters

        # Test multiple queries with multiple probes
        queries = self.vectors[:5]
        partitions = ivf.search_partitions(queries, n_probe=3)

        assert len(partitions) == 5
        for partition_list in partitions:
            assert len(partition_list) == 3
            assert all(0 <= p < self.n_clusters for p in partition_list)

    def test_get_partition_vectors(self):
        """Test getting vectors from specific partitions."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10, verbose=False)

        # Test getting vectors from each partition
        for cluster_id in range(self.n_clusters):
            vector_ids = ivf.get_partition_vectors(cluster_id)
            assert isinstance(vector_ids, list)
            assert all(isinstance(vid, int) for vid in vector_ids)
            assert all(0 <= vid < self.n_vectors for vid in vector_ids)

    def test_cluster_stats(self):
        """Test cluster statistics."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10, verbose=False)

        stats = ivf.get_cluster_stats()

        assert stats["n_clusters"] == self.n_clusters
        assert stats["total_vectors"] == self.n_vectors
        assert stats["min_size"] >= 0
        assert stats["max_size"] <= self.n_vectors
        assert stats["mean_size"] == self.n_vectors / self.n_clusters
        assert len(stats["cluster_sizes"]) == self.n_clusters

    def test_save_load(self):
        """Test saving and loading IVF index."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10, verbose=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_ivf.pkl"

            # Save
            ivf.save(save_path)
            assert save_path.exists()

            # Load
            ivf_loaded = InvertedFileIndex.load(save_path)

            # Check loaded index matches original
            assert ivf_loaded.d == ivf.d
            assert ivf_loaded.n_clusters == ivf.n_clusters
            assert ivf_loaded.is_trained == ivf.is_trained
            assert torch.allclose(ivf_loaded.coarse_centroids, ivf.coarse_centroids)
            assert ivf_loaded.inverted_lists == ivf.inverted_lists

    def test_export_metadata(self):
        """Test metadata export for browser consumption."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=10, verbose=False)

        metadata = ivf.export_metadata()

        assert metadata["d"] == self.d
        assert metadata["n_clusters"] == self.n_clusters
        assert metadata["is_trained"] == True
        assert "cluster_stats" in metadata
        assert metadata["coarse_centroids_shape"] == [self.n_clusters, self.d]

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)

        # Test operations on untrained index
        with pytest.raises(RuntimeError):
            ivf.search_partitions(self.vectors[:1])

        with pytest.raises(RuntimeError):
            ivf.get_partition_vectors(0)

        with pytest.raises(RuntimeError):
            ivf.get_cluster_stats()

        with pytest.raises(RuntimeError):
            ivf.save(Path("test.pkl"))

        with pytest.raises(RuntimeError):
            ivf.export_metadata()

        # Train and test invalid partition access
        ivf.train_ivf(self.vectors, self.vector_ids, n_iterations=5, verbose=False)

        with pytest.raises(ValueError):
            ivf.get_partition_vectors(-1)

        with pytest.raises(ValueError):
            ivf.get_partition_vectors(self.n_clusters)

    def test_mismatched_vector_ids(self):
        """Test error handling for mismatched vector IDs."""
        ivf = InvertedFileIndex(self.d, self.n_clusters)

        # Wrong number of vector IDs
        wrong_ids = torch.arange(self.n_vectors // 2)

        with pytest.raises(ValueError):
            ivf.train_ivf(self.vectors, wrong_ids)


class TestIVFIntegration:
    """Test IVF integration with real-world scenarios."""

    def test_different_cluster_counts(self):
        """Test with different numbers of clusters."""
        d = 64
        n_vectors = 500
        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        for n_clusters in [4, 8, 16, 32]:
            ivf = InvertedFileIndex(d, n_clusters)
            ivf.train_ivf(vectors, n_iterations=10, verbose=False)

            assert ivf.is_trained
            stats = ivf.get_cluster_stats()
            assert stats["total_vectors"] == n_vectors

    def test_high_dimensional_data(self):
        """Test with high-dimensional data similar to SCimilarity."""
        d = 512  # SCimilarity embedding dimension
        n_vectors = 200
        n_clusters = 8

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        ivf = InvertedFileIndex(d, n_clusters)
        ivf.train_ivf(vectors, n_iterations=15, verbose=False)

        assert ivf.is_trained
        assert ivf.coarse_centroids.shape == (n_clusters, d)

        # Test search
        query = torch.randn(1, d)
        partitions = ivf.search_partitions(query, n_probe=2)
        assert len(partitions[0]) == 2

    def test_kmeans_plus_plus_initialization(self):
        """Test that k-means++ initialization produces reasonable results."""
        d = 32
        n_vectors = 100
        n_clusters = 8

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        ivf = InvertedFileIndex(d, n_clusters)

        # Initialize centroids
        ivf._init_centroids_kmeans_plus_plus(vectors)

        # Check that centroids are not identical
        centroids = ivf.coarse_centroids.data
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                distance = torch.norm(centroids[i] - centroids[j])
                assert distance > 0, f"Centroids {i} and {j} are identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
