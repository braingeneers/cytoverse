#!/usr/bin/env python3

"""
Test suite for Inverted File Index (IVF) implementation with residual vectors.

This module tests the core functionality of the IVF index including:
- Training with scikit-learn on synthetic and real data
- Residual vector computation and storage
- Binary file export/import functionality
- Search with residual-based PQ distance computation
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json

from ivfpq.ivf import (
    InvertedFileIndex,
    train_ivf_residual,
    search_ivf_residual,
    load_centroids_binary,
    _export_centroids_binary,
    _export_partition_binary,
    _load_partition_binary,
)
from ivfpq.pq import ProductQuantizer, train_pq_codebooks


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

    def test_train_ivf_residual(self, tmp_path):
        """Test IVF training with residual vectors."""
        output_dir = tmp_path
        
        # Train IVF with residual vectors
        result = train_ivf_residual(
            vectors=self.vectors,
            n_partitions=self.n_partitions,
            output_dir=output_dir,
            max_iterations=20,
            verbose=True
        )
        
        # Check training results
        assert "centroids" in result
        assert "assignments" in result
        assert "residuals_by_partition" in result
        assert "vector_ids_by_partition" in result
        
        centroids = result["centroids"]
        assignments = result["assignments"]
        
        # Verify centroids shape
        assert centroids.shape == (self.n_partitions, self.d)
        
        # Verify all vectors are assigned
        assert len(assignments) == self.n_vectors
        assert np.all(assignments >= 0)
        assert np.all(assignments < self.n_partitions)
        
        # Check files were created
        assert (output_dir / "centroids.bin").exists()
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "pq_metadata.json").exists()
        assert (output_dir / "pq_codebooks.bin").exists()
        assert (output_dir / "pq_codebooks.npy").exists()
        assert (output_dir / "pq_encode.onnx").exists()
        assert (output_dir / "pq_decode.onnx").exists()
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
        loaded_centroids = load_centroids_binary(centroids_file)
        
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

    def test_ivf_search_residual(self, tmp_path):
        """Test IVF search with residual vectors."""
        output_dir = tmp_path
        
        # Train IVF
        result = train_ivf_residual(
            vectors=self.vectors,
            n_partitions=self.n_partitions,
            output_dir=output_dir,
            max_iterations=10,
            verbose=False
        )
        
        centroids = result["centroids"]
        
        # Test search with exact vector
        test_idx = 42
        query_vector = self.vectors[test_idx]
        
        vector_ids, distances = search_ivf_residual(
            query_vector=query_vector,
            centroids=centroids,
            n_probe=4,
            model_path=output_dir,
            k=10,
            verbose=False
        )
        
        # Should find the exact vector as closest
        assert test_idx in vector_ids
        exact_match_idx = vector_ids.index(test_idx)
        
        # The exact match should be reasonably close (allowing for PQ reconstruction error)
        assert distances[exact_match_idx] < 5.0, f"Distance to exact match: {distances[exact_match_idx]}"

    def test_inverted_file_index_class(self, tmp_path):
        """Test the InvertedFileIndex class interface."""
        ivf = InvertedFileIndex(self.d, self.n_partitions)
        
        assert ivf.d == self.d
        assert ivf.n_partitions == self.n_partitions
        assert not ivf.is_trained
        
        output_dir = tmp_path
        
        # Train
        result = ivf.train_ivf(
            vectors=self.vectors,
            output_dir=output_dir,
            max_iterations=10,
            verbose=False
        )
        
        assert ivf.is_trained
        assert "centroids" in result
        
        # Test search
        query_vector = self.vectors[0]
        vector_ids, distances = ivf.search(
            query_vector=query_vector,
            model_path=output_dir,
            n_probe=2,
            k=5,
            verbose=False
        )
        
        assert len(vector_ids) <= 5
        assert len(distances) == len(vector_ids)
        assert all(isinstance(vid, int) for vid in vector_ids)
        assert all(isinstance(dist, float) for dist in distances)


class TestIVFResidualIntegration:
    """Integration tests for IVF with residual vectors."""

    def test_different_vector_dimensions(self, tmp_path):
        """Test with different vector dimensions."""
        for d in [64, 128, 256]:
            n_vectors = 200
            n_partitions = 8
            
            torch.manual_seed(42)
            vectors = torch.randn(n_vectors, d)
            
            test_output_dir = tmp_path / f"test_dim_{d}"
            test_output_dir.mkdir(exist_ok=True)
            
            result = train_ivf_residual(
                vectors=vectors,
                n_partitions=n_partitions,
                output_dir=test_output_dir,
                max_iterations=5,
                verbose=False
            )
            
            assert result["centroids"].shape == (n_partitions, d)
            assert len(result["assignments"]) == n_vectors

    def test_different_partition_counts(self, tmp_path):
        """Test with different numbers of partitions."""
        d = 64
        n_vectors = 300
        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)
        
        for n_partitions in [4, 8, 16]:
            test_output_dir = tmp_path / f"test_partitions_{n_partitions}"
            test_output_dir.mkdir(exist_ok=True)
            
            result = train_ivf_residual(
                vectors=vectors,
                n_partitions=n_partitions,
                output_dir=test_output_dir,
                max_iterations=5,
                verbose=False
            )
            
            assert result["centroids"].shape == (n_partitions, d)
            
            # Test metadata
            metadata_file = test_output_dir / "metadata.json"
            with open(metadata_file, 'r') as f:
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
        result = train_ivf_residual(
            vectors=vectors,
            n_partitions=n_partitions,
            output_dir=output_dir,
            max_iterations=15,
            verbose=False
        )
        
        centroids = result["centroids"]
        
        # Test with multiple queries
        test_indices = [10, 50, 100, 200, 300]
        
        for test_idx in test_indices:
            # Test exact query
            query_vector = vectors[test_idx]
            
            vector_ids, distances = search_ivf_residual(
                query_vector=query_vector,
                centroids=centroids,
                n_probe=4,
                model_path=output_dir,
                k=10,
                verbose=False
            )
            
            # Should find the exact vector
            assert test_idx in vector_ids, f"Failed to find exact match for index {test_idx}"
            
            # Test with small perturbation
            noise = torch.randn_like(query_vector) * 0.05
            perturbed_query = query_vector + noise
            
            perturbed_ids, perturbed_distances = search_ivf_residual(
                query_vector=perturbed_query,
                centroids=centroids,
                n_probe=4,
                model_path=output_dir,
                k=10,
                verbose=False
            )
            
            # Original vector should still be in top results
            assert test_idx in perturbed_ids, f"Failed to find perturbed match for index {test_idx}"

    def test_empty_partitions_handling(self, tmp_path):
        """Test handling of empty partitions."""
        d = 32
        n_vectors = 20  # Small number to potentially create empty partitions
        n_partitions = 50  # Large number to ensure some empty partitions
        
        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)
        
        output_dir = tmp_path
        
        result = train_ivf_residual(
            vectors=vectors,
            n_partitions=n_partitions,
            output_dir=output_dir,
            max_iterations=5,
            verbose=False
        )
        
        # Check that empty partitions are handled
        assert result["centroids"].shape == (n_partitions, d)
        
        # Count non-empty partitions
        partition_sizes = result["metadata"]["partition_sizes"]
        non_empty_count = sum(1 for size in partition_sizes.values() if size > 0)
        
        # Should have fewer non-empty partitions than total partitions
        assert non_empty_count < n_partitions
        assert non_empty_count > 0
        
        # Search should still work
        query_vector = vectors[0]
        vector_ids, distances = search_ivf_residual(
            query_vector=query_vector,
            centroids=result["centroids"],
            n_probe=2,
            model_path=output_dir,
            k=5,
            verbose=False
        )
        
        assert len(vector_ids) > 0
        assert 0 in vector_ids  # Should find the exact query


class TestIVFResidualCompatibility:
    """Test compatibility with different data types and edge cases."""

    def test_save_load_compatibility(self, tmp_path):
        """Test save/load compatibility with residual-based IVF."""
        d = 64
        n_partitions = 8
        
        ivf = InvertedFileIndex(d, n_partitions)
        
        # Create synthetic centroids
        centroids = torch.randn(n_partitions, d)
        ivf.centroids.data = centroids
        ivf.is_trained = True
        
        save_path = tmp_path / "ivf_residual.pkl"
        
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

    def test_error_handling(self, tmp_path):
        """Test error handling for invalid operations."""
        d = 64
        n_partitions = 8
        ivf = InvertedFileIndex(d, n_partitions)
        
        # Test operations on untrained index
        with pytest.raises(RuntimeError):
            ivf.save(Path("test.pkl"))
        
        # Test invalid dimensions
        wrong_vector = torch.randn(32)  # Wrong dimension
        output_dir = tmp_path
        
        # Train with correct vectors first (use more vectors for k=256)
        correct_vectors = torch.randn(1000, d)
        ivf.train_ivf(correct_vectors, output_dir, verbose=False)
        
        # Now test search with wrong dimension - should handle gracefully
        # (The search function should validate dimensions)
        with pytest.raises((ValueError, RuntimeError)):
            ivf.search(wrong_vector, output_dir, n_probe=1, k=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])