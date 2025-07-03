"""
Comprehensive tests for the IVFPQ implementation.

This module tests the complete IVFPQ pipeline including:
- Training separate IVF and PQ models
- Exporting models to disk
- Creating IVFPQ partitions
- Searching with both exact and perturbed queries
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import random
import json
from typing import List, Tuple

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cellspace.ivfpq import ProductQuantizer, InvertedFileIndex, IVFPQ


class TestIVFPQComplete:
    """Comprehensive test suite for IVFPQ system."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic embedding data for testing with labels."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        n_vectors = 1000
        d = 128
        l = 8  # Number of unique labels

        # Generate random vectors
        vectors = torch.randn(n_vectors, d)
        vector_ids = torch.arange(n_vectors)

        # Generate text labels
        label_names = [f"CellType_{i:02d}" for i in range(l)]

        # Randomly assign labels to vectors
        labels = [random.choice(label_names) for _ in range(n_vectors)]

        return vectors, vector_ids, labels, d

    @pytest.fixture
    def model_params(self):
        """Standard model parameters for testing."""
        return {
            "d": 128,
            "m": 16,  # PQ parameters
            "k": 256,
            "n_partitions": 32,  # IVF parameters
        }

    def test_complete_pipeline(self, synthetic_data, model_params):
        """Test the complete IVFPQ pipeline from training to search."""
        vectors, vector_ids, labels, d = synthetic_data

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)

            # Step 1: Train and save PQ model
            print("Training PQ model...")
            pq = ProductQuantizer(
                d=model_params["d"], m=model_params["m"], k=model_params["k"]
            )
            pq.train_pq(vectors, n_iterations=20)

            # Save PQ model
            pq_dir = model_path / "pq"
            pq_dir.mkdir(parents=True)
            pq.save(pq_dir / "model.pkl")

            # Step 2: Train and save IVF model
            print("Training IVF model...")
            ivf = InvertedFileIndex(
                d=model_params["d"], n_partitions=model_params["n_partitions"]
            )
            ivf.train_ivf(vectors, vector_ids, n_iterations=20)

            # Save IVF model
            ivf_dir = model_path / "ivf"
            ivf_dir.mkdir(parents=True)
            ivf.save(ivf_dir / "model.pkl")

            # Step 3: Create IVFPQ instance and encode vectors
            print("Creating IVFPQ index...")
            ivfpq = IVFPQ(model_path)
            ivfpq.encode_vectors(vectors, vector_ids)

            # Verify encoding
            assert len(ivfpq.partition_data) == model_params["n_partitions"]
            total_encoded = sum(p["size"] for p in ivfpq.partition_data.values())
            assert total_encoded == len(vectors)

            # Step 4: Export to disk
            print("Exporting IVFPQ index...")
            ivfpq.export(model_path)

            # Verify export files exist
            ivfpq_dir = model_path / "ivfpq"
            assert (ivfpq_dir / "metadata.json").exists()
            assert (ivfpq_dir / "centroids.arrow").exists()
            partitions_dir = ivfpq_dir / "partitions"
            assert partitions_dir.exists()

            # Step 5: Test search with exact vector
            print("Testing exact vector search...")
            test_idx = 42
            query_vector = vectors[test_idx]
            expected_id = vector_ids[test_idx].item()
            expected_label = labels[test_idx]

            # Search using in-memory data
            result_ids, result_partitions, distances, result_labels = ivfpq.search(
                query_vector, k=4, n_probe=8, labels=labels
            )

            # Verify exact match is found and is the closest
            assert (
                expected_id in result_ids
            ), f"Expected ID {expected_id} not found in results {result_ids}"
            exact_match_idx = result_ids.index(expected_id)
            assert result_labels[exact_match_idx] == expected_label
            # The exact query should be the closest match (index 0), allowing for PQ reconstruction error
            assert (
                exact_match_idx == 0
            ), f"Expected exact match to be closest (index 0), but found at index {exact_match_idx}"

            # Step 6: Test search with perturbed vector
            print("Testing perturbed vector search...")
            noise_scale = 0.1
            noise = torch.randn_like(query_vector) * noise_scale
            perturbed_query = query_vector + noise

            (
                perturbed_ids,
                perturbed_partitions,
                perturbed_distances,
                perturbed_labels,
            ) = ivfpq.search(perturbed_query, k=4, n_probe=8, labels=labels)

            # Verify original vector is still among top k=4 neighbors
            assert (
                expected_id in perturbed_ids
            ), f"Expected ID {expected_id} not found in perturbed results {perturbed_ids}"
            perturbed_match_idx = perturbed_ids.index(expected_id)
            assert perturbed_labels[perturbed_match_idx] == expected_label

            # Step 7: Test search using on-disk partitions
            print("Testing on-disk partition search...")
            # Create new IVFPQ instance that will load partitions from disk
            ivfpq_disk = IVFPQ(model_path)

            disk_ids, disk_partitions, disk_distances, disk_labels = ivfpq_disk.search(
                query_vector, k=4, n_probe=8, labels=labels, model_path=model_path
            )

            # Results should be similar to in-memory search
            assert (
                expected_id in disk_ids
            ), f"Expected ID {expected_id} not found in disk results {disk_ids}"

            print("All tests passed!")

    def test_search_functionality_detailed(self, synthetic_data, model_params):
        """Test search functionality in detail."""
        vectors, vector_ids, labels, d = synthetic_data

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)

            # Train models
            pq = ProductQuantizer(
                d=model_params["d"], m=model_params["m"], k=model_params["k"]
            )
            pq.train_pq(vectors, n_iterations=15)
            pq_dir = model_path / "pq"
            pq_dir.mkdir(parents=True)
            pq.save(pq_dir / "model.pkl")

            ivf = InvertedFileIndex(
                d=model_params["d"], n_partitions=model_params["n_partitions"]
            )
            ivf.train_ivf(vectors, vector_ids, n_iterations=15)
            ivf_dir = model_path / "ivf"
            ivf_dir.mkdir(parents=True)
            ivf.save(ivf_dir / "model.pkl")

            # Create IVFPQ and encode
            ivfpq = IVFPQ(model_path)
            ivfpq.encode_vectors(vectors, vector_ids)

            # Test multiple exact queries
            for test_idx in [10, 50, 100, 200]:
                query_vector = vectors[test_idx]
                expected_id = vector_ids[test_idx].item()

                result_ids, _, distances, _ = ivfpq.search(
                    query_vector, k=10, n_probe=8
                )

                # Exact match should be first or very close
                assert (
                    expected_id in result_ids[:5]
                ), f"Expected ID {expected_id} not in top 5 for query {test_idx}"

            # Test that different n_probe values affect results
            query_vector = vectors[0]

            results_probe_1, _, _, _ = ivfpq.search(query_vector, k=5, n_probe=1)
            results_probe_8, _, _, _ = ivfpq.search(query_vector, k=5, n_probe=8)

            # More probes should potentially find different (possibly better) results
            # At minimum, they should return valid results
            assert len(results_probe_1) <= 5
            assert len(results_probe_8) <= 5

    def test_model_loading_and_export(self, synthetic_data, model_params):
        """Test model saving/loading and export functionality."""
        vectors, vector_ids, labels, d = synthetic_data

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)

            # Train and save models
            pq = ProductQuantizer(
                d=model_params["d"], m=model_params["m"], k=model_params["k"]
            )
            pq.train_pq(vectors, n_iterations=10)
            pq_dir = model_path / "pq"
            pq_dir.mkdir(parents=True)
            pq.save(pq_dir / "model.pkl")

            ivf = InvertedFileIndex(
                d=model_params["d"], n_partitions=model_params["n_partitions"]
            )
            ivf.train_ivf(vectors, vector_ids, n_iterations=10)
            ivf_dir = model_path / "ivf"
            ivf_dir.mkdir(parents=True)
            ivf.save(ivf_dir / "model.pkl")

            # Test IVFPQ loading
            ivfpq = IVFPQ(model_path)
            assert ivfpq.d == model_params["d"]
            assert ivfpq.m == model_params["m"]
            assert ivfpq.k == model_params["k"]
            assert ivfpq.n_partitions == model_params["n_partitions"]

            # Encode and export
            ivfpq.encode_vectors(vectors, vector_ids)
            ivfpq.export(model_path)

            # Verify export structure
            ivfpq_dir = model_path / "ivfpq"
            assert ivfpq_dir.exists()
            assert (ivfpq_dir / "metadata.json").exists()
            assert (ivfpq_dir / "centroids.arrow").exists()

            # Check metadata content
            with open(ivfpq_dir / "metadata.json") as f:
                metadata = json.load(f)

            assert "ivfpq" in metadata
            assert "ivf" in metadata
            assert "pq" in metadata
            assert metadata["ivfpq"]["d"] == model_params["d"]
            assert metadata["ivfpq"]["n_partitions"] == model_params["n_partitions"]

    def test_label_retrieval(self, synthetic_data, model_params):
        """Test that labels are correctly retrieved in search results."""
        vectors, vector_ids, labels, d = synthetic_data

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)

            # Setup models
            pq = ProductQuantizer(
                d=model_params["d"], m=model_params["m"], k=model_params["k"]
            )
            pq.train_pq(vectors, n_iterations=10)
            pq_dir = model_path / "pq"
            pq_dir.mkdir(parents=True)
            pq.save(pq_dir / "model.pkl")

            ivf = InvertedFileIndex(
                d=model_params["d"], n_partitions=model_params["n_partitions"]
            )
            ivf.train_ivf(vectors, vector_ids, n_iterations=10)
            ivf_dir = model_path / "ivf"
            ivf_dir.mkdir(parents=True)
            ivf.save(ivf_dir / "model.pkl")

            ivfpq = IVFPQ(model_path)
            ivfpq.encode_vectors(vectors, vector_ids)

            # Test with labels
            test_idx = 123
            query_vector = vectors[test_idx]
            expected_label = labels[test_idx]

            result_ids, _, _, result_labels = ivfpq.search(
                query_vector, k=5, n_probe=8, labels=labels
            )

            # Find the exact match and verify its label
            expected_id = vector_ids[test_idx].item()
            if expected_id in result_ids:
                match_idx = result_ids.index(expected_id)
                assert result_labels[match_idx] == expected_label

            # Test without labels
            result_ids_no_labels, _, _, result_labels_no_labels = ivfpq.search(
                query_vector, k=5, n_probe=8
            )

            # Should return empty strings for labels
            assert all(label == "" for label in result_labels_no_labels)

    def test_error_handling(self, model_params):
        """Test error handling for various edge cases."""
        # Test loading non-existent models
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)

            with pytest.raises(FileNotFoundError, match="IVF model not found"):
                IVFPQ(model_path)

            # Create IVF but not PQ
            ivf_dir = model_path / "ivf"
            ivf_dir.mkdir(parents=True)
            ivf = InvertedFileIndex(
                d=model_params["d"], n_partitions=model_params["n_partitions"]
            )
            # Don't train, just save a dummy file
            (ivf_dir / "model.pkl").touch()

            with pytest.raises(FileNotFoundError, match="PQ model not found"):
                IVFPQ(model_path)

    def test_partition_distribution(self, synthetic_data, model_params):
        """Test that vectors are distributed across partitions reasonably."""
        vectors, vector_ids, labels, d = synthetic_data

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)

            # Setup models
            pq = ProductQuantizer(
                d=model_params["d"], m=model_params["m"], k=model_params["k"]
            )
            pq.train_pq(vectors, n_iterations=10)
            pq_dir = model_path / "pq"
            pq_dir.mkdir(parents=True)
            pq.save(pq_dir / "model.pkl")

            ivf = InvertedFileIndex(
                d=model_params["d"], n_partitions=model_params["n_partitions"]
            )
            ivf.train_ivf(vectors, vector_ids, n_iterations=10)
            ivf_dir = model_path / "ivf"
            ivf_dir.mkdir(parents=True)
            ivf.save(ivf_dir / "model.pkl")

            ivfpq = IVFPQ(model_path)
            ivfpq.encode_vectors(vectors, vector_ids)

            # Check partition distribution
            stats = ivfpq.get_partition_stats()

            assert stats["total_vectors"] == len(vectors)
            assert stats["total_partitions"] == model_params["n_partitions"]
            assert stats["non_empty_partitions"] > 0
            assert stats["avg_partition_size"] > 0

            # Check that most partitions have some vectors (not all concentrated in one)
            non_empty_count = sum(1 for size in stats["partition_sizes"] if size > 0)
            assert (
                non_empty_count >= model_params["n_partitions"] // 4
            )  # At least 25% should be non-empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
