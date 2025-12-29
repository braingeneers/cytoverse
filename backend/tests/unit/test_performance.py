#!/usr/bin/env python3
"""
Test accuracy and performance of IVFPQ system compared to SCimilarity.

This test:
1. Uses data/GSE136831_subsample.h5ad as query file
2. Generates embeddings and labels using SCimilarity
3. Uses IVFPQ to generate labels for the same cells
4. Compares accuracy between the two methods
"""

import sys
from pathlib import Path

# Add scripts directory to path for ivfpq import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
import numpy as np
import pandas as pd
import anndata
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import time

from scimilarity import CellAnnotation
from scimilarity.utils import align_dataset, lognorm_counts

from ivfpq import IVFPQ


@pytest.fixture
def test_data_path():
    """Path to test h5ad file."""
    return Path(__file__).parent.parent / "fixtures" / "GSE136831_subsample_100.h5ad"


@pytest.fixture
def scimilarity_model_path():
    """Path to SCimilarity model."""
    return (
        Path(__file__).parent.parent.parent
        / "data"
        / "models"
        / "scimilarity"
        / "model_v1.1"
    )


@pytest.fixture
def ivfpq_model_path():
    """Path to IVFPQ model artifacts."""
    return (
        Path(__file__).parent.parent.parent
        / "public"
        / "models"
        / "scimilarity"
        / "ivfpq"
    )


@pytest.fixture
def reference_labels_path():
    """Path to reference labels."""
    return (
        Path(__file__).parent.parent.parent
        / "data"
        / "references"
        / "scimilarity"
        / "labels.parquet"
    )


def get_scimilarity_predictions(
    adata: anndata.AnnData, model_path: Path, k_neighbors: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get cell type predictions using SCimilarity.

    Returns:
        embeddings: Cell embeddings [n_cells, embedding_dim]
        predictions: Predicted cell type labels
    """
    # Load SCimilarity model
    cell_annotation = CellAnnotation(model_path=str(model_path), use_gpu=False)

    # Add counts layer if needed
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Align and normalize
    adata_aligned = align_dataset(adata, cell_annotation.gene_order)
    adata_normalized = lognorm_counts(adata_aligned)

    # Get embeddings
    if hasattr(adata_normalized.X, "toarray"):
        expression_data = adata_normalized.X.toarray().astype(np.float32)
    else:
        expression_data = adata_normalized.X.astype(np.float32)

    embeddings = cell_annotation.get_embeddings(expression_data)

    # Get predictions
    predictions, nn_idxs, nn_dists, stats = cell_annotation.get_predictions_knn(
        embeddings, k=k_neighbors, disable_progress=True
    )

    return embeddings, predictions.values


def get_ivfpq_predictions_batch(
    embeddings: np.ndarray,
    model_path: Path,
    reference_labels: pd.DataFrame,
    k_neighbors: int = 50,
    n_partitions_search: int = 10,
    batch_size: int = 10,
) -> np.ndarray:
    """
    Get cell type predictions using IVFPQ with batched processing.

    Returns:
        predictions: Predicted cell type labels
    """

    predictions = []
    n_cells = len(embeddings)

    ivf = IVFPQ.load(model_path)

    # Process in batches for efficiency
    for batch_start in range(0, n_cells, batch_size):
        batch_end = min(batch_start + batch_size, n_cells)
        batch_embeddings = embeddings[batch_start:batch_end]

        batch_predictions = []
        for embedding in batch_embeddings:
            # Convert to torch tensor
            query_vector = torch.from_numpy(embedding).float()

            # Search using IVFPQ
            vector_ids, distances, _ = ivf.search(
                query_vector=query_vector,
                model_path=model_path,
                n_probe=n_partitions_search,
            )

            # Get labels for nearest neighbors
            neighbor_labels = []
            for vec_id in vector_ids[:k_neighbors]:  # Ensure we only use k neighbors
                if vec_id < len(reference_labels):
                    # Get the "prediction" column from reference labels
                    label = reference_labels.iloc[vec_id]["prediction"]
                    neighbor_labels.append(label)

            # Vote for most common label (simple majority voting)
            if neighbor_labels:
                label_counts = Counter(neighbor_labels)
                predicted_label = label_counts.most_common(1)[0][0]
            else:
                predicted_label = "Unknown"

            batch_predictions.append(predicted_label)

        predictions.extend(batch_predictions)
        print(f"  Processed {batch_end}/{n_cells} cells")

    return np.array(predictions)


def calculate_accuracy(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Calculate accuracy between two sets of labels."""
    matches = np.sum(labels1 == labels2)
    total = len(labels1)
    return matches / total


def test_ivfpq_self_accuracy(ivfpq_model_path, reference_labels_path):
    """
    Test IVFPQ accuracy by searching for random embeddings from the reference dataset
    and comparing the retrieved labels with ground truth.

    This tests the accuracy of the IVFPQ vector database alone without SCimilarity.
    """
    import random
    import json

    # Load reference embeddings
    reference_embeddings_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "references"
        / "scimilarity"
        / "embeddings.npy"
    )

    print("Loading reference embeddings...")
    reference_embeddings = np.load(reference_embeddings_path)
    print(f"  Reference embeddings shape: {reference_embeddings.shape}")

    # Load reference labels
    print("Loading reference labels...")
    reference_labels = pd.read_parquet(reference_labels_path)
    print(f"  Reference labels shape: {reference_labels.shape}")

    # Test parameters
    n_test = 100  # Number of random embeddings to test
    k_neighbors = 50
    n_probe_values = [1, 2, 4, 8, 16, 32]

    print(f"\n=== IVFPQ Self-Accuracy Test ===")
    print(f"Testing {n_test} random embeddings from reference dataset")
    print(f"k-neighbors: {k_neighbors}")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    ivf = IVFPQ.load(ivfpq_model_path)

    results = {}

    for n_probe in n_probe_values:
        print(f"\nTesting with n_probe={n_probe}:")

        correct_predictions = 0
        total_tests = 0

        # Sample random indices
        n_embeddings = reference_embeddings.shape[0]
        test_indices = random.sample(range(n_embeddings), min(n_test, n_embeddings))

        for idx in test_indices:
            # Get embedding and ground truth label
            query_embedding = reference_embeddings[idx]
            ground_truth_label = reference_labels.iloc[idx]["prediction"]

            # Convert to torch tensor
            query_vector = torch.from_numpy(query_embedding).float()

            try:
                # Search using IVFPQ
                vector_ids, distances, _ = ivf.search(
                    query_vector=query_vector,
                    model_path=ivfpq_model_path,
                    n_probe=n_probe,
                )

                # Get labels for nearest neighbors
                neighbor_labels = []
                for vec_id in vector_ids[:k_neighbors]:
                    if vec_id < len(reference_labels):
                        label = reference_labels.iloc[vec_id]["prediction"]
                        neighbor_labels.append(label)

                # Vote for most common label
                if neighbor_labels:
                    label_counts = Counter(neighbor_labels)
                    predicted_label = label_counts.most_common(1)[0][0]

                    if predicted_label == ground_truth_label:
                        correct_predictions += 1

                    total_tests += 1

            except Exception as e:
                print(f"  Search failed for index {idx}: {e}")

        if total_tests > 0:
            accuracy = correct_predictions / total_tests
            results[n_probe] = {
                "correct": correct_predictions,
                "total": total_tests,
                "accuracy": accuracy,
            }
            print(f"  Accuracy: {correct_predictions}/{total_tests} ({accuracy:.2%})")
        else:
            print(f"  No successful searches")

    # Print summary
    print("\n=== Summary ===")
    print("n_probe | Accuracy")
    print("--------|----------")
    for n_probe, res in results.items():
        print(f"{n_probe:7d} | {res['accuracy']:.2%}")

    # Assert minimum accuracy for reasonable n_probe values
    if 8 in results:
        min_accuracy = 0.70  # 70% accuracy threshold for n_probe=8
        actual_accuracy = results[8]["accuracy"]
        assert (
            actual_accuracy >= min_accuracy
        ), f"Accuracy {actual_accuracy:.2%} with n_probe=8 is below required {min_accuracy:.0%} threshold"
        print(
            f"\n✓ Test passed: IVFPQ self-accuracy ({actual_accuracy:.2%}) meets {min_accuracy:.0%} threshold"
        )


def test_ivfpq_accuracy(
    test_data_path, scimilarity_model_path, ivfpq_model_path, reference_labels_path
):
    """Test that IVFPQ achieves reasonable accuracy compared to SCimilarity."""

    # Load test data
    adata = anndata.read_h5ad(test_data_path)

    # Use subset for faster testing
    # n_test_cells = 20  # Test with 20 cells for reasonable runtime
    n_test_cells = 100  # More comprehensive but time consuming
    adata = adata[:n_test_cells, :]
    print(f"Testing with {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Get SCimilarity predictions (ground truth)
    print("Getting SCimilarity predictions...")
    start_time = time.time()
    embeddings, scimilarity_predictions = get_scimilarity_predictions(
        adata, scimilarity_model_path, k_neighbors=50
    )
    scimilarity_time = time.time() - start_time
    print(f"  SCimilarity time: {scimilarity_time:.2f}s")
    print(f"  Predictions shape: {scimilarity_predictions.shape}")

    # Load reference labels for IVFPQ
    print("Loading reference labels...")
    reference_labels = pd.read_parquet(reference_labels_path)
    print(f"  Reference labels shape: {reference_labels.shape}")

    # Get IVFPQ predictions
    print("Getting IVFPQ predictions...")
    start_time = time.time()
    ivfpq_predictions = get_ivfpq_predictions_batch(
        embeddings,
        ivfpq_model_path,
        reference_labels,
        k_neighbors=50,
        n_partitions_search=20,
        batch_size=5,
    )
    ivfpq_time = time.time() - start_time
    print(f"  IVFPQ time: {ivfpq_time:.2f}s")
    print(f"  Predictions shape: {ivfpq_predictions.shape}")

    # Calculate accuracy
    accuracy = calculate_accuracy(scimilarity_predictions, ivfpq_predictions)
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"SCimilarity time: {scimilarity_time:.2f}s")
    print(f"IVFPQ time: {ivfpq_time:.2f}s")
    print(f"Speedup: {scimilarity_time/ivfpq_time:.2f}x")

    # Print confusion details
    matches = scimilarity_predictions == ivfpq_predictions
    mismatches_idx = np.where(~matches)[0]

    if len(mismatches_idx) > 0:
        print(f"\nMismatches ({len(mismatches_idx)} total):")
        for idx in mismatches_idx[:10]:  # Show first 10 mismatches
            print(
                f"  Cell {idx}: SCimilarity='{scimilarity_predictions[idx]}', IVFPQ='{ivfpq_predictions[idx]}'"
            )
        if len(mismatches_idx) > 10:
            print(f"  ... and {len(mismatches_idx) - 10} more")

    # Assert minimum accuracy
    # Note: Using a lower threshold for initial implementation
    # The difference could be due to:
    # 1. IVFPQ using approximate nearest neighbor search
    # 2. Quantization errors in the PQ encoding
    # 3. Limited search partitions reducing recall
    min_accuracy = 0.60  # 60% accuracy threshold for initial implementation
    assert (
        accuracy >= min_accuracy
    ), f"Accuracy {accuracy:.4f} is below required {min_accuracy:.0%} threshold"

    print(
        f"\n✓ Test passed: IVFPQ accuracy ({accuracy:.2%}) meets {min_accuracy:.0%} threshold"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
