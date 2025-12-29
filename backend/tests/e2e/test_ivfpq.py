#!/usr/bin/env python3

"""
Test suite for Inverted File Index (IVF) with Product Quantized (PQ) residual vectors


This module tests the core functionality of the IVF index including:
- Training with scikit-learn on synthetic and real data
- Residual vector computation and storage
- Binary file export/import functionality
- Search with residual-based PQ distance computation
- Consensus computation from merged partition results

The new IVFPQ implementation returns ALL vectors from ALL probed partitions,
merged and sorted by distance. The 'k' parameter in search limits how many of
these merged results are returned, but internally all vectors from all probed
partitions are considered and sorted globally.
"""

import sys
from pathlib import Path

# Add scripts directory to path for ivfpq import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from unittest import result
import pytest
import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple
from collections import Counter

from ivfpq import (
    IVFPQ,
    _load_centroids_binary,
    _export_centroids_binary,
    _load_partition_binary,
    _export_partition_binary,
)
from pq import PQ


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
        IVFPQ.build(
            vectors=self.vectors,
            output_dir=output_dir,
            num_training_vectors=self.vectors.shape[0],
            max_iterations=20,
        )
        ivf = IVFPQ.load(output_dir)

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

    def test_forward_onnx(self, tmp_path):
        """Test the exported ONNX model for coarse centroid search produces concordant results."""
        import onnxruntime as ort

        # First build an IVFPQ to generate the ONNX model
        output_dir = tmp_path
        IVFPQ.build(
            vectors=self.vectors,
            output_dir=output_dir,
            num_training_vectors=self.vectors.shape[0],
            max_iterations=10,
        )

        # Load the IVF model
        ivf = IVFPQ.load(output_dir)

        # Verify ONNX file was created
        onnx_file = output_dir / "ivf_coarse.onnx"
        assert onnx_file.exists(), "ivf_coarse.onnx not found"

        # Create test inputs
        query_vector = torch.randn(self.d)
        centroids = ivf.centroids.data
        k = 4

        # Get PyTorch results
        pytorch_indices, pytorch_distances = ivf.forward(query_vector, centroids, k)

        # Create ONNX runtime session
        ort_session = ort.InferenceSession(str(onnx_file))

        # Prepare inputs for ONNX (need to convert to numpy and add batch dimension if needed)
        ort_inputs = {
            "query_vector": query_vector.numpy(),
            "partition_centroids": centroids.numpy(),
            "k": np.array(k, dtype=np.int32),
        }

        # Run ONNX inference
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_indices = torch.from_numpy(ort_outputs[0])
        onnx_distances = torch.from_numpy(ort_outputs[1])

        # Validate results are concordant
        assert torch.equal(
            pytorch_indices, onnx_indices
        ), f"ONNX indices {onnx_indices} don't match PyTorch indices {pytorch_indices}"

        # For distances, allow small numerical differences due to floating point precision
        assert torch.allclose(
            pytorch_distances, onnx_distances, rtol=1e-5, atol=1e-6
        ), f"ONNX distances differ from PyTorch distances beyond tolerance"

        # Verify both are properly sorted
        assert torch.all(
            onnx_distances[:-1] <= onnx_distances[1:]
        ), "ONNX distances should be sorted"

        print(f"✓ ONNX forward method produces concordant results with PyTorch")
        print(f"  PyTorch indices: {pytorch_indices.tolist()}")
        print(f"  ONNX indices: {onnx_indices.tolist()}")
        print(
            f"  Distance difference: {torch.abs(pytorch_distances - onnx_distances).max().item():.2e}"
        )

    def test_ivf_search(self, tmp_path):
        """Test IVF search with residual vectors using the refactored search method."""
        output_dir = tmp_path

        # Train IVF
        IVFPQ.build(
            vectors=self.vectors,
            output_dir=output_dir,
            num_training_vectors=self.vectors.shape[0],
            max_iterations=10,
        )
        ivf = IVFPQ.load(output_dir)

        # Test search with exact vector
        test_idx = 42
        query_vector = self.vectors[test_idx]
        k_results = 50  # How many of the globally sorted results to return

        # Search merges ALL vectors from ALL probed partitions, sorts them globally,
        # then returns the top k from this complete sorted list
        vector_ids, distances, _ = ivf.search(
            query_vector=query_vector,
            model_path=output_dir,
            n_probe=4,
            k=k_results,  # Return top 50 from the globally sorted merged results
        )

        # The returned results are the top-k from ALL vectors in the probed partitions
        assert len(vector_ids) <= k_results
        assert len(distances) == len(vector_ids)

        # Should find the exact vector in top results
        assert (
            test_idx in vector_ids
        ), f"Exact match {test_idx} not found in top {k_results} results"
        exact_match_idx = vector_ids.index(test_idx)

        print(f"Distance to exact match (idx {test_idx}): {distances[exact_match_idx]}")
        print(
            f"Found at position {exact_match_idx} in top {len(vector_ids)} globally sorted results"
        )
        print(
            f"Note: These are from ALL vectors in {4} probed partitions, globally sorted"
        )

        # The exact match should be reasonably close (allowing for PQ reconstruction error)
        assert (
            distances[exact_match_idx] < 10.0
        ), f"Distance to exact match: {distances[exact_match_idx]}"

        # Test with small perturbation
        noise = torch.randn_like(query_vector) * 0.05
        perturbed_query = query_vector + noise

        perturbed_ids, perturbed_distances, _ = ivf.search(
            query_vector=perturbed_query,
            model_path=output_dir,
            n_probe=4,
            k=k_results,  # Return top 50 from globally sorted merged results
        )

        # Original vector should still be in the globally sorted top results
        assert (
            test_idx in perturbed_ids
        ), f"Failed to find perturbed match for index {test_idx} in top {k_results} global results"

    def test_save_and_load(self, tmp_path):
        """Test saving and loading the IVFPQ."""
        output_dir = tmp_path

        # Train and save
        IVFPQ.build(
            vectors=self.vectors,
            output_dir=output_dir,
            num_training_vectors=self.vectors.shape[0],
            max_iterations=10,
        )

        # Load and verify
        ivf = IVFPQ.load(output_dir)
        assert ivf.is_trained


class TestIVFResidualIntegration:
    """Integration tests for IVF with residual vectors."""

    def compute_consensus_label(self, vector_ids: List[int], k: int = 1) -> List[int]:
        """
        Compute consensus label(s) from a list of candidate vector IDs.

        Args:
            vector_ids: List of vector IDs from merged partition results
            k: Number of consensus labels to return (default 1 for single nearest neighbor)

        Returns:
            List of k consensus labels based on frequency in top results
        """
        if not vector_ids:
            return []

        # For simple nearest neighbor, just return the top k
        # In a real scenario with labels, you might want to do voting
        return vector_ids[:k]

    def test_consensus_from_merged_partitions(self, tmp_path):
        """Test that search returns globally sorted results from all probed partitions."""
        d = 64
        n_vectors = 500

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        output_dir = tmp_path

        # Train IVF
        IVFPQ.build(
            vectors=vectors,
            output_dir=output_dir,
            num_training_vectors=vectors.shape[0],
            max_iterations=10,
        )
        ivf = IVFPQ.load(output_dir)

        # Test with different n_probe values
        test_idx = 100
        query_vector = vectors[test_idx]
        k_results = 50  # How many results to return from the global sort

        print(f"\nTesting global sorting from all partitions for vector {test_idx}")

        for n_probe in [1, 2, 4, 8]:
            # Search merges ALL vectors from ALL n_probe partitions,
            # sorts them globally by distance, then returns top k
            vector_ids, distances, _ = ivf.search(
                query_vector=query_vector,
                model_path=output_dir,
                n_probe=n_probe,
                k=k_results,  # Return top 50 from global sort
            )

            # Compute consensus from the globally sorted results
            consensus_labels = self.compute_consensus_label(vector_ids, k=5)

            print(f"  n_probe={n_probe}:")
            print(
                f"    Returned top {len(vector_ids)} from ALL vectors in {n_probe} partitions"
            )
            print(f"    Top 5 consensus: {consensus_labels}")

            # Check if exact match is in the global results
            if test_idx in vector_ids:
                pos = vector_ids.index(test_idx)
                print(f"    Exact match found at global position {pos}")
            else:
                print(f"    Exact match not in top {k_results} global results")

            # Verify distances are globally sorted
            for i in range(1, len(distances)):
                assert (
                    distances[i - 1] <= distances[i]
                ), f"Distances not globally sorted at index {i}"

        print(f"✓ Global sorting from all probed partitions working correctly")

    def test_search_returns_all_partition_vectors(self, tmp_path):
        """Test that search returns ALL vectors from ALL probed partitions, globally sorted."""
        d = 64
        n_vectors = 256  # Small dataset to make counting easier
        n_partitions = 16  # sqrt(n_vectors) as rule of thumb in ivfpq.build

        torch.manual_seed(42)
        # Create vectors with clear partition structure
        vectors = []
        for i in range(n_vectors):
            # Create vectors that will cluster into roughly equal partitions
            partition_id = i % 16
            base = torch.zeros(d)
            base[partition_id * 16 : (partition_id + 1) * 16] = (
                5.0  # Strong signal in different dimensions
            )
            noise = torch.randn(d) * 0.1
            vectors.append(base + noise)
        vectors = torch.stack(vectors)

        output_dir = tmp_path

        # Train IVF
        IVFPQ.build(
            vectors=vectors,
            output_dir=output_dir,
            num_training_vectors=vectors.shape[0],
            max_iterations=20,
        )
        ivf = IVFPQ.load(output_dir)
        assert ivf.n_partitions == n_partitions

        # Query with a vector that should match partition 0
        query_vector = torch.zeros(d)
        query_vector[:16] = 5.0

        print(f"\nTesting that search returns ALL vectors from probed partitions")

        # Test with n_probe=1: should get ALL vectors from 1 partition
        vector_ids_1, distances_1, _ = ivf.search(
            query_vector=query_vector,
            model_path=output_dir,
            n_probe=1,
            k=1000,  # Set k very high to get all results
        )

        print(f"  n_probe=1: Got {len(vector_ids_1)} vectors from 1 partition")

        # Test with n_probe=2: should get ALL vectors from 2 partitions
        vector_ids_2, distances_2, _ = ivf.search(
            query_vector=query_vector,
            model_path=output_dir,
            n_probe=2,
            k=1000,  # Set k very high to get all results
        )

        print(f"  n_probe=2: Got {len(vector_ids_2)} vectors from 2 partitions")

        # With n_probe=2, we should get more vectors than n_probe=1
        assert len(vector_ids_2) > len(
            vector_ids_1
        ), f"n_probe=2 should return more vectors than n_probe=1"

        # Test with n_probe=4 (all partitions): should get ALL vectors
        vector_ids_all, distances_all, _ = ivf.search(
            query_vector=query_vector,
            model_path=output_dir,
            n_probe=n_partitions,  # Search all partitions
            k=1000,  # Set k very high to get all results
        )

        print(
            f"  n_probe={n_partitions}: Got {len(vector_ids_all)} vectors from ALL {n_partitions} partitions"
        )
        print(f"  Total vectors in dataset: {n_vectors}")

        # When searching all partitions, we should get all vectors
        # (may be slightly less if some partitions are empty)
        assert (
            len(vector_ids_all) >= n_vectors * 0.9
        ), f"Should get most vectors when searching all partitions"

        # All results should be globally sorted by distance
        for results, distances in [
            (vector_ids_1, distances_1),
            (vector_ids_2, distances_2),
            (vector_ids_all, distances_all),
        ]:
            for i in range(1, len(distances)):
                assert (
                    distances[i - 1] <= distances[i]
                ), f"Distances not globally sorted at index {i}"

        # Test that limiting k works correctly
        vector_ids_limited, distances_limited, _ = ivf.search(
            query_vector=query_vector,
            model_path=output_dir,
            n_probe=n_partitions,  # Search all partitions
            k=10,  # Limit to top 10
        )

        assert len(vector_ids_limited) == 10, f"Should return exactly k=10 results"
        assert (
            vector_ids_limited == vector_ids_all[:10]
        ), "Limited results should be first 10 of unlimited results"

        print(
            f"  With k=10: Got exactly {len(vector_ids_limited)} top results from global sort"
        )
        print(
            f"✓ Search correctly returns ALL vectors from probed partitions, globally sorted"
        )

    def test_consensus_with_voting(self, tmp_path):
        """Test consensus computation with voting for classification scenarios."""
        d = 64
        n_vectors = 500
        n_classes = 10  # Simulate classification with 10 classes

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        # Create synthetic labels (simulating a classification dataset)
        labels = [i % n_classes for i in range(n_vectors)]

        output_dir = tmp_path

        # Train IVF
        IVFPQ.build(
            vectors=vectors,
            output_dir=output_dir,
            num_training_vectors=vectors.shape[0],
            max_iterations=10,
        )
        ivf = IVFPQ.load(output_dir)

        # Test with a query that's a slight perturbation of an existing vector
        test_idx = 150
        true_label = labels[test_idx]

        # Add noise to create a realistic query
        query_vector = vectors[test_idx] + torch.randn(d) * 0.1

        k_results = 50  # How many results to return from global sort
        n_probe = 4

        # Search merges ALL vectors from ALL probed partitions,
        # sorts them globally, and returns the top k_results
        vector_ids, distances, _ = ivf.search(
            query_vector=query_vector,
            model_path=output_dir,
            n_probe=n_probe,
            k=k_results,  # Return top 50 from the global sort
        )

        print(f"\nConsensus voting example:")
        print(f"  Searched {n_probe} partitions, merged ALL their vectors")
        print(f"  Returned top {len(vector_ids)} from global sort")

        # Compute consensus label using voting from top-k neighbors
        k_neighbors = 5
        neighbor_labels = [labels[vid] for vid in vector_ids[:k_neighbors]]

        # Vote for the most common label
        label_counts = Counter(neighbor_labels)
        consensus_label = label_counts.most_common(1)[0][0]

        print(
            f"  Query: perturbed version of vector {test_idx} (true label: {true_label})"
        )
        print(
            f"  Top {k_neighbors} neighbors from global sort: {vector_ids[:k_neighbors]}"
        )
        print(f"  Their labels: {neighbor_labels}")
        print(f"  Label votes: {dict(label_counts)}")
        print(f"  Consensus label: {consensus_label}")
        print(f"  Correct: {consensus_label == true_label}")

        # Verify the results are globally sorted
        assert len(vector_ids) <= k_results
        assert all(
            distances[i - 1] <= distances[i] for i in range(1, len(distances))
        ), "Results must be globally sorted by distance"

    def test_different_vector_dimensions(self, tmp_path):
        """Test with different vector dimensions."""
        for d in [64, 128, 256]:
            n_vectors = 200

            torch.manual_seed(42)
            vectors = torch.randn(n_vectors, d)

            test_output_dir = tmp_path / f"test_dim_{d}"
            test_output_dir.mkdir(exist_ok=True)

            IVFPQ.build(
                vectors=vectors,
                output_dir=test_output_dir,
                num_training_vectors=vectors.shape[0],
                max_iterations=5,
            )
            ivf = IVFPQ.load(test_output_dir)

            assert ivf.centroids.shape == (ivf.n_partitions, d)

    def test_search_accuracy(self, tmp_path):
        """Test search accuracy with perturbed queries using globally sorted results."""
        d = 64
        n_vectors = 500
        n_partitions = 8
        k_results = 50  # How many results to return from global sort

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        output_dir = tmp_path

        # Train
        IVFPQ.build(
            vectors=vectors,
            output_dir=output_dir,
            num_training_vectors=vectors.shape[0],
            max_iterations=15,
        )
        ivf = IVFPQ.load(output_dir)

        # Test with multiple queries
        test_indices = [10, 50, 100, 200, 300]
        exact_match_positions = []
        perturbed_match_positions = []

        for test_idx in test_indices:
            # Test exact query
            query_vector = vectors[test_idx]

            # Search merges ALL vectors from ALL probed partitions,
            # sorts them globally, and returns top k_results
            vector_ids, distances, _ = ivf.search(
                query_vector=query_vector,
                model_path=output_dir,
                n_probe=4,
                k=k_results,  # Return top 50 from global sort
            )

            # Verify results are globally sorted
            for i in range(1, len(distances)):
                assert distances[i - 1] <= distances[i], "Results not globally sorted"

            # Check if exact match is in the global top-k
            assert (
                test_idx in vector_ids
            ), f"Failed to find exact match for index {test_idx} in top {k_results} global results"

            exact_pos = vector_ids.index(test_idx)
            exact_match_positions.append(exact_pos)

            # For exact matches, should typically be in top 10 of global results
            assert (
                exact_pos < 10
            ), f"Exact match at global position {exact_pos}, expected in top 10"

            # Test with small perturbation
            noise = torch.randn_like(query_vector) * 0.05
            perturbed_query = query_vector + noise

            perturbed_ids, perturbed_distances, _ = ivf.search(
                query_vector=perturbed_query,
                model_path=output_dir,
                n_probe=4,
                k=k_results,  # Return top 50 from global sort
            )

            # Original vector should still be in global top-k results
            assert (
                test_idx in perturbed_ids
            ), f"Failed to find perturbed match for index {test_idx} in top {k_results} global results"

            perturbed_pos = perturbed_ids.index(test_idx)
            perturbed_match_positions.append(perturbed_pos)

            # For slightly perturbed queries, should still be in top 20 of global results
            assert (
                perturbed_pos < 20
            ), f"Perturbed match at global position {perturbed_pos}, expected in top 20"

        # Print summary statistics
        avg_exact_pos = sum(exact_match_positions) / len(exact_match_positions)
        avg_perturbed_pos = sum(perturbed_match_positions) / len(
            perturbed_match_positions
        )

        print(f"\nSearch accuracy summary:")
        print(f"  Average global position for exact matches: {avg_exact_pos:.1f}")
        print(
            f"  Average global position for perturbed matches: {avg_perturbed_pos:.1f}"
        )
        print(f"  All positions from top {k_results} of globally sorted results")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
