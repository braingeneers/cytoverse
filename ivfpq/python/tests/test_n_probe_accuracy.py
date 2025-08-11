#!/usr/bin/env python3

"""
Test to verify that search accuracy improves or remains stable as n_probe increases.

This test creates a sophisticated distribution of vectors with clear cluster structure,
then verifies that self-recall (finding exact matches) improves monotonically with n_probe.

The new IVFPQ implementation returns distances and vector IDs from all partitions probed,
which are then merged and sorted. A consensus is computed from the top-k (default 50) 
results from this merged list to determine the final nearest neighbors.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple

from ivfpq.ivfpq import IVFPQ


class TestNProbeAccuracy:
    """Test accuracy vs n_probe relationship."""

    def setup_method(self):
        """Set up test fixtures with sophisticated vector distribution."""
        self.d = 128
        self.n_vectors = 10_000
        self.n_partitions = int(
            np.sqrt(self.n_vectors)
        )  # Square root for balanced partitions
        self.n_clusters = 16  # Natural clusters in data

        # Create sophisticated vector distribution with clear structure
        torch.manual_seed(42)
        np.random.seed(42)

        # Create well-separated cluster centers
        cluster_centers = []
        for i in range(self.n_clusters):
            # Create centers that are well-separated in space
            center = torch.randn(self.d)
            center = center / torch.norm(center) * (10 + i * 2)  # Different radii
            # Add some structured offset
            if i % 2 == 0:
                center[: self.d // 2] += 3.0
            else:
                center[self.d // 2 :] += 3.0
            cluster_centers.append(center)

        # Generate vectors around these centers with varying spread
        vectors = []
        cluster_assignments = []
        for i in range(self.n_vectors):
            cluster_id = i % self.n_clusters
            cluster_assignments.append(cluster_id)

            # Vary the spread based on cluster ID
            spread = 0.3 + (cluster_id / self.n_clusters) * 0.7
            noise = torch.randn(self.d) * spread

            # Add some vectors that are between clusters (harder cases)
            if i % 20 == 0 and cluster_id < self.n_clusters - 1:
                # Interpolate between two clusters
                alpha = 0.3 + np.random.random() * 0.4
                vector = (
                    alpha * cluster_centers[cluster_id]
                    + (1 - alpha) * cluster_centers[cluster_id + 1]
                    + noise * 0.5
                )
            else:
                vector = cluster_centers[cluster_id] + noise

            vectors.append(vector)

        self.vectors = torch.stack(vectors)
        self.cluster_assignments = cluster_assignments
        self.cluster_centers = cluster_centers

    def compute_recall_at_k(
        self,
        query_indices: List[int],
        n_probe: int,
        centroids: torch.Tensor,
        model_path: Path,
        k: int = 10,
        topk_consensus: int = 50,
        verbose: bool = False,
    ) -> float:
        """
        Compute recall@k for a set of query indices.

        The new IVFPQ returns distances and vector IDs from all partitions probed,
        which are then merged and sorted to compute consensus from the topk results.

        Recall@k = fraction of queries where the exact match appears in top-k results.
        """
        found_count = 0

        for i, query_idx in enumerate(query_indices):
            query_vector = self.vectors[query_idx]

            ivf = IVFPQ.load(model_path)
            # Search returns merged results from all probed partitions
            # The topk_consensus parameter determines how many results per partition to consider
            vector_ids, distances = ivf.search(
                query_vector=query_vector,
                model_path=model_path,
                n_probe=n_probe,
                k=topk_consensus,  # Get more results for consensus
            )

            if verbose and i < 3:
                print(f"    Query {query_idx}: returned {len(vector_ids)} results from {n_probe} partitions")

            # Check if exact match is in the final top-k results after consensus
            if query_idx in vector_ids[:k]:
                found_count += 1
                if verbose and i < 5:  # Show first few examples
                    idx_pos = vector_ids.index(query_idx)
                    print(
                        f"    Query {query_idx}: found at position {idx_pos}, distance={distances[idx_pos]:.3f}"
                    )
            elif verbose and i < 5:
                print(f"    Query {query_idx}: NOT FOUND in top {k} (consensus from top {topk_consensus})")

        recall = found_count / len(query_indices)
        return recall

    def test_n_probe_monotonic_accuracy(self, tmp_path):
        """Test that accuracy improves monotonically with n_probe."""
        output_dir = tmp_path

        # Train IVF with our sophisticated distribution
        print(
            f"\nTraining IVF with {self.n_vectors} vectors, {self.n_partitions} partitions"
        )
        ivf = IVFPQ.build(
            vectors=self.vectors,
            n_partitions=self.n_partitions,
            output_dir=output_dir,
            max_iterations=30,  # More iterations for better convergence
            pq_m=16,
            pq_k=256,
        )

        # Create APPROXIMATE queries by perturbing existing vectors
        # This simulates real-world queries that are similar but not exact matches
        query_vectors = []
        ground_truth_indices = []

        np.random.seed(123)  # Different seed for queries

        # Select diverse base vectors for perturbation
        base_indices = list(range(0, self.n_vectors, 50))[:100]  # 100 queries

        for base_idx in base_indices:
            base_vector = self.vectors[base_idx]

            # Add significant noise to create approximate query
            # The noise should be large enough to potentially change partition assignment
            noise_level = 1.5  # Significant perturbation
            noise = torch.randn_like(base_vector) * noise_level
            query_vector = base_vector + noise

            query_vectors.append(query_vector)
            ground_truth_indices.append(
                base_idx
            )  # The original vector is the "ground truth"

        print(f"\nTesting with {len(query_vectors)} approximate query vectors")

        # Test with increasing n_probe values
        n_probe_values = [1, 2, 4, 8, 16, 24, 32]
        recalls = []
        topk_consensus = 50  # Default consensus size

        for n_probe in n_probe_values:
            if n_probe > self.n_partitions:
                n_probe = self.n_partitions

            # Compute recall for approximate queries
            found_count = 0
            for i, (query_vector, gt_idx) in enumerate(
                zip(query_vectors, ground_truth_indices)
            ):
                # Search returns merged and sorted results from all probed partitions
                vector_ids, distances = ivf.search(
                    query_vector=query_vector,
                    model_path=output_dir,
                    n_probe=n_probe,
                    k=topk_consensus,  # Get top 50 for consensus computation
                )

                # Check if ground truth is in top 5 results after consensus
                if gt_idx in vector_ids[:5]:
                    found_count += 1
                    if (
                        n_probe in [1, 32] and i < 3
                    ):  # Show first few for min/max n_probe
                        idx_pos = vector_ids.index(gt_idx)
                        print(
                            f"    Query {i}: ground truth {gt_idx} found at position {idx_pos} (from {len(vector_ids)} consensus results), distance={distances[idx_pos]:.3f}"
                        )
                elif n_probe in [1, 32] and i < 3:
                    print(f"    Query {i}: ground truth {gt_idx} NOT FOUND in top 5 (from {len(vector_ids)} consensus results)")

            recall = found_count / len(query_vectors)

            recalls.append(recall)
            print(f"n_probe={n_probe:3d}: recall@5 = {recall:.3f} (consensus from top {topk_consensus})")

        # Verify monotonic improvement (allowing for small fluctuations due to ties)
        for i in range(1, len(recalls)):
            # Allow tiny decrease due to numerical issues or ties
            assert recalls[i] >= recalls[i - 1] - 0.01, (
                f"Recall decreased from {recalls[i-1]:.3f} to {recalls[i]:.3f} "
                f"when n_probe increased from {n_probe_values[i-1]} to {n_probe_values[i]}"
            )

        # Verify reasonable absolute recall values
        assert recalls[0] > 0.2, f"Recall with n_probe=1 too low: {recalls[0]:.3f}"
        assert (
            recalls[-1] >= 0.7
        ), f"Recall with n_probe={n_probe_values[-1]} too low: {recalls[-1]:.3f}"

        # Verify substantial improvement from min to max n_probe
        improvement = recalls[-1] - recalls[0]
        assert improvement > 0.05, (
            f"Insufficient improvement from n_probe=1 to n_probe={n_probe_values[-1]}: "
            f"{improvement:.3f}"
        )

        print(
            f"\n✓ Recall improves monotonically from {recalls[0]:.3f} to {recalls[-1]:.3f}"
        )
        print(f"✓ Total improvement: {improvement:.3f}")

    def test_distance_consistency_across_partitions(self, tmp_path):
        """
        Test that the same vector found in different partitions
        has consistent distance estimates when using consensus from merged results.
        """
        output_dir = tmp_path

        # Train IVF
        ivf = IVFPQ.build(
            vectors=self.vectors,
            n_partitions=self.n_partitions,
            output_dir=output_dir,
            max_iterations=20,
            pq_m=16,
            pq_k=256,
        )

        centroids = ivf.centroids

        # Find a query vector and track its distance across different n_probe values
        test_idx = 42
        query_vector = self.vectors[test_idx]
        topk_consensus = 50  # Default consensus size

        # Search with increasing n_probe and track when we find it
        distances_when_found = []
        n_probe_when_found = []

        for n_probe in range(1, min(self.n_partitions + 1, 33)):
            # Search returns merged results from all probed partitions
            vector_ids, distances = ivf.search(
                query_vector=query_vector,
                model_path=output_dir,
                n_probe=n_probe,
                k=topk_consensus,  # Get consensus from top 50
            )

            if test_idx in vector_ids:
                idx_in_results = vector_ids.index(test_idx)
                distance = distances[idx_in_results]
                distances_when_found.append(distance)
                n_probe_when_found.append(n_probe)

                if len(distances_when_found) == 1:
                    print(
                        f"  First found with n_probe={n_probe}, distance={distance:.3f} (from {len(vector_ids)} consensus results)"
                    )

        # All distances should be very similar (within PQ quantization noise)
        if len(distances_when_found) > 1:
            min_dist = min(distances_when_found)
            max_dist = max(distances_when_found)
            dist_range = max_dist - min_dist

            print(
                f"  Distance range across different n_probe: [{min_dist:.3f}, {max_dist:.3f}]"
            )
            print(f"  Range: {dist_range:.3f}")

            # Distance should be consistent (small variation due to numerical precision)
            assert (
                dist_range < 0.1
            ), f"Distance varies too much across partitions: range={dist_range:.3f}"

            print(f"✓ Distance estimates are consistent across partitions")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
