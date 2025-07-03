"""
Tests for the combined IVFPQ implementation.

This module tests the integration of IVF and PQ components in the complete
IVFPQ system, including training, encoding, partitioning, and save/load functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cellspace.ivfpq import IVFPQ


class TestIVFPQ:
    """Test suite for IVFPQ class."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic embedding data for testing."""
        torch.manual_seed(42)
        np.random.seed(42)

        n_vectors = 1000
        d = 128
        vectors = torch.randn(n_vectors, d)
        vector_ids = torch.arange(n_vectors)

        return vectors, vector_ids, d

    @pytest.fixture
    def ivfpq_params(self):
        """Standard IVFPQ parameters for testing."""
        return {"d": 128, "m": 16, "k": 256, "n_clusters": 32}

    def test_initialization(self, ivfpq_params):
        """Test IVFPQ initialization."""
        ivfpq = IVFPQ(**ivfpq_params)

        assert ivfpq.d == ivfpq_params["d"]
        assert ivfpq.m == ivfpq_params["m"]
        assert ivfpq.k == ivfpq_params["k"]
        assert ivfpq.n_clusters == ivfpq_params["n_clusters"]
        assert not ivfpq.is_trained
        assert len(ivfpq.partition_data) == 0

    def test_training(self, synthetic_data, ivfpq_params):
        """Test complete IVFPQ training pipeline."""
        vectors, vector_ids, d = synthetic_data
        ivfpq = IVFPQ(**ivfpq_params)

        # Train the model
        ivfpq.train(
            vectors=vectors,
            vector_ids=vector_ids,
            ivf_iterations=10,  # Reduced for testing speed
            pq_iterations=10,  # Reduced for testing speed
            verbose=False,
        )

        # Verify training state
        assert ivfpq.is_trained
        assert ivfpq.ivf.is_trained
        assert ivfpq.pq.is_trained

        # Verify partition data structure
        assert len(ivfpq.partition_data) == ivfpq_params["n_clusters"]

        # Verify all vectors are accounted for
        total_vectors = sum(p["size"] for p in ivfpq.partition_data.values())
        assert total_vectors == len(vectors)

    def test_partition_operations(self, synthetic_data, ivfpq_params):
        """Test partition-related operations."""
        vectors, vector_ids, d = synthetic_data
        ivfpq = IVFPQ(**ivfpq_params)
        ivfpq.train(
            vectors, vector_ids, ivf_iterations=5, pq_iterations=5, verbose=False
        )

        # Test partition ID retrieval
        partition_ids = ivfpq.get_all_partition_ids()
        assert len(partition_ids) == ivfpq_params["n_clusters"]
        assert partition_ids == list(range(ivfpq_params["n_clusters"]))

        # Test partition data access
        for partition_id in partition_ids:
            partition_data = ivfpq.get_partition_data(partition_id)
            assert "vector_ids" in partition_data
            assert "pq_codes" in partition_data
            assert "size" in partition_data
            assert len(partition_data["vector_ids"]) == partition_data["size"]
            assert len(partition_data["pq_codes"]) == partition_data["size"]

    def test_search_partitions(self, synthetic_data, ivfpq_params):
        """Test partition selection for search queries."""
        vectors, vector_ids, d = synthetic_data
        ivfpq = IVFPQ(**ivfpq_params)
        ivfpq.train(
            vectors, vector_ids, ivf_iterations=5, pq_iterations=5, verbose=False
        )

        # Test search partition selection
        query_vector = torch.randn(d)

        for n_probe in [1, 4, 8]:
            selected_partitions = ivfpq.search_partitions(query_vector, n_probe=n_probe)
            assert len(selected_partitions) == min(n_probe, ivfpq_params["n_clusters"])
            assert all(0 <= p < ivfpq_params["n_clusters"] for p in selected_partitions)

    def test_partition_stats(self, synthetic_data, ivfpq_params):
        """Test partition statistics computation."""
        vectors, vector_ids, d = synthetic_data
        ivfpq = IVFPQ(**ivfpq_params)
        ivfpq.train(
            vectors, vector_ids, ivf_iterations=5, pq_iterations=5, verbose=False
        )

        stats = ivfpq.get_partition_stats()

        assert "total_partitions" in stats
        assert "non_empty_partitions" in stats
        assert "total_vectors" in stats
        assert "partition_sizes" in stats
        assert "avg_partition_size" in stats
        assert "min_partition_size" in stats
        assert "max_partition_size" in stats

        assert stats["total_partitions"] == ivfpq_params["n_clusters"]
        assert stats["total_vectors"] == len(vectors)
        assert len(stats["partition_sizes"]) == ivfpq_params["n_clusters"]

    def test_save_and_load(self, synthetic_data, ivfpq_params):
        """Test save and load functionality."""
        vectors, vector_ids, d = synthetic_data

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Train and save model
            ivfpq_original = IVFPQ(**ivfpq_params)
            ivfpq_original.train(
                vectors, vector_ids, ivf_iterations=5, pq_iterations=5, verbose=False
            )
            ivfpq_original.save(temp_path)

            # Load model
            ivfpq_loaded = IVFPQ.load(temp_path)

            # Verify loaded model matches original
            assert ivfpq_loaded.d == ivfpq_original.d
            assert ivfpq_loaded.m == ivfpq_original.m
            assert ivfpq_loaded.k == ivfpq_original.k
            assert ivfpq_loaded.n_clusters == ivfpq_original.n_clusters
            assert ivfpq_loaded.is_trained == ivfpq_original.is_trained

            # Verify partition data is preserved
            assert len(ivfpq_loaded.partition_data) == len(
                ivfpq_original.partition_data
            )

            # Test that loaded model works for search
            query_vector = torch.randn(d)
            original_partitions = ivfpq_original.search_partitions(
                query_vector, n_probe=4
            )
            loaded_partitions = ivfpq_loaded.search_partitions(query_vector, n_probe=4)
            assert original_partitions == loaded_partitions

    def test_metadata_export(self, synthetic_data, ivfpq_params):
        """Test metadata export functionality."""
        vectors, vector_ids, d = synthetic_data
        ivfpq = IVFPQ(**ivfpq_params)
        ivfpq.train(
            vectors, vector_ids, ivf_iterations=5, pq_iterations=5, verbose=False
        )

        metadata = ivfpq.export_metadata()

        # Verify structure
        assert "ivfpq" in metadata
        assert "ivf" in metadata
        assert "pq" in metadata

        # Verify IVFPQ metadata
        ivfpq_meta = metadata["ivfpq"]
        assert ivfpq_meta["d"] == ivfpq_params["d"]
        assert ivfpq_meta["m"] == ivfpq_params["m"]
        assert ivfpq_meta["k"] == ivfpq_params["k"]
        assert ivfpq_meta["n_clusters"] == ivfpq_params["n_clusters"]
        assert ivfpq_meta["is_trained"] == True
        assert "stats" in ivfpq_meta

    def test_error_handling(self, ivfpq_params):
        """Test error handling for untrained models."""
        ivfpq = IVFPQ(**ivfpq_params)

        # Test operations that require training
        with pytest.raises(RuntimeError, match="IVFPQ must be trained"):
            ivfpq.search_partitions(torch.randn(ivfpq_params["d"]))

        with pytest.raises(RuntimeError, match="IVFPQ must be trained"):
            ivfpq.get_partition_data(0)

        with pytest.raises(RuntimeError, match="IVFPQ must be trained"):
            ivfpq.get_partition_stats()

        with pytest.raises(RuntimeError, match="IVFPQ must be trained"):
            ivfpq.export_metadata()

    def test_invalid_partition_id(self, synthetic_data, ivfpq_params):
        """Test handling of invalid partition IDs."""
        vectors, vector_ids, d = synthetic_data
        ivfpq = IVFPQ(**ivfpq_params)
        ivfpq.train(
            vectors, vector_ids, ivf_iterations=5, pq_iterations=5, verbose=False
        )

        # Test invalid partition ID
        with pytest.raises(ValueError, match="Invalid partition ID"):
            ivfpq.get_partition_data(ivfpq_params["n_clusters"])  # Out of range

        with pytest.raises(ValueError, match="Invalid partition ID"):
            ivfpq.get_partition_data(-1)  # Negative

    def test_compression_ratio(self, synthetic_data, ivfpq_params):
        """Test that the compression ratio is as expected."""
        vectors, vector_ids, d = synthetic_data
        ivfpq = IVFPQ(**ivfpq_params)
        ivfpq.train(
            vectors, vector_ids, ivf_iterations=5, pq_iterations=5, verbose=False
        )

        # Calculate expected compression ratio
        original_bits_per_vector = d * 32  # 32-bit floats
        compressed_bits_per_vector = ivfpq_params["m"] * 8  # 8-bit codes
        expected_compression = original_bits_per_vector / compressed_bits_per_vector

        # Verify the compression ratio is reasonable
        assert expected_compression == 32.0  # For our test parameters

        # Verify each partition contains PQ codes of the right size
        for partition_id in range(ivfpq_params["n_clusters"]):
            partition_data = ivfpq.get_partition_data(partition_id)
            if partition_data["size"] > 0:
                # Each vector should have m codes
                for pq_codes in partition_data["pq_codes"]:
                    assert len(pq_codes) == ivfpq_params["m"]


if __name__ == "__main__":
    pytest.main([__file__])
