#!/usr/bin/env python3

"""
Training script for IVFPQ (Inverted File Index Product Quantization) models.

This script provides complete IVFPQ training functionality using residual vectors
for improved accuracy, exporting browser-compatible artifacts.

Features:
- Train complete IVFPQ models with residual vectors
- Export browser-compatible artifacts under public/models/<model_id>/
- Integration with TypeScript browser implementation
- Performance testing with trained models
"""

import typer
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Optional

from ivfpq.ivf import train_ivf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Train IVFPQ models on vectors",
    add_completion=False,
)


def _load_vectors(vectors_path: Path, max_vectors: Optional[int] = None) -> np.ndarray:
    """Load vectors and return as a numpy array."""
    logger.info(f"Loading vectors from {vectors_path}")
    vectors = np.load(vectors_path)
    logger.info(
        f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]} from .npy file"
    )

    # Limit number of vectors if specified (take first max_vectors)
    if max_vectors is not None and vectors.shape[0] > max_vectors:
        logger.info(f"Using first {max_vectors} vectors for training")
        vectors = vectors[:max_vectors]

    return vectors


@app.command()
def train(
    vectors_path: Path = typer.Argument(..., help="Path to vectors.npy", exists=True),
    output_dir: Path = typer.Argument(
        ..., help="Output directory for IVFPQ model artifacts"
    ),
    pq_m: int = typer.Option(16, help="Number of PQ subspaces"),
    pq_k: int = typer.Option(256, help="Number of centroids per PQ subspace"),
    max_vectors: Optional[int] = typer.Option(
        None,
        help="Maximum number of vectors to store in the index (default: all vectors)",
    ),
    max_vectors_for_training: int = typer.Option(
        50000,
        help="Maximum number of vectors to use for k-means training (default: 50k for memory efficiency)",
    ),
    n_iterations: int = typer.Option(
        30, help="Number of k-means iterations for coarse quantization"
    ),
) -> None:
    """
    Train complete IVFPQ model with residual vectors for browser consumption.

    Creates a complete set of artifacts in the specified output directory that are
    compatible with the browser TypeScript implementation.
    """
    vectors = _load_vectors(vectors_path, max_vectors)
    vectors_tensor = torch.from_numpy(vectors)
    d = vectors_tensor.shape[1]

    logger.info(
        f"Training IVFPQ with residual vectors: d={d}, pq_m={pq_m}, pq_k={pq_k}"
    )
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train IVF with residual vectors (includes PQ training and export)
    train_ivf(
        vectors=vectors_tensor,
        output_dir=output_dir,
        n_training_vectors=max_vectors_for_training,
        max_iterations=n_iterations,
        pq_m=pq_m,
        pq_k=pq_k,
    )

    logger.info(f"IVFPQ training with residual vectors completed")
    logger.info(f"Browser-compatible artifacts saved to {output_dir}:")
    logger.info(f"  - centroids.bin: partition centroids for IVF")
    logger.info(f"  - partitions/: PQ-encoded residual vectors by partition")
    logger.info(f"  - pq_*.onnx: ONNX models for browser PQ distance computation")
    logger.info(f"  - pq_*.npy: binary codebooks and metadata")
    logger.info(f"  - ivf_metadata.json: training metadata")


def _test_ivfpq_search(
    model_path: Path,
    vectors: torch.Tensor,
    n_probe_values: list[int],
    n_test: int = 100,
) -> None:
    """Test IVFPQ search performance using the trained residual model."""
    from ivfpq.ivf import search_ivf, load_centroids_binary
    import json

    n_test = min(n_test, vectors.shape[0])
    test_queries = vectors[:n_test]

    logger.info(f"IVFPQ Search Results:")
    logger.info(f"  Test queries: {n_test}")

    # Load metadata
    metadata_file = model_path / "ivf_metadata.json"
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    n_partitions = metadata["n_partitions"]
    logger.info(f"  Total partitions: {n_partitions}")

    # Load centroids
    centroids_file = model_path / "centroids.bin"
    centroids = load_centroids_binary(centroids_file)

    # Test with different n_probe values
    for n_probe in n_probe_values:
        if n_probe > n_partitions:
            continue

        total_candidates = 0

        for query_vector in test_queries[: min(10, n_test)]:  # Test subset for speed
            try:
                vector_ids, distances = search_ivf(
                    query_vector=query_vector,
                    centroids=centroids,
                    n_probe=n_probe,
                    model_path=model_path,
                    k_per_partition=50,
                    verbose=False,
                )
                total_candidates += len(vector_ids)
            except Exception as e:
                logger.warning(f"Search failed for query: {e}")

        avg_candidates = (
            total_candidates / min(10, n_test) if total_candidates > 0 else 0
        )
        search_fraction = (
            avg_candidates / metadata["total_vectors"]
            if metadata["total_vectors"] > 0
            else 0
        )

        logger.info(
            f"  n_probe={n_probe}: avg {avg_candidates:.1f} candidates ({search_fraction:.2%} of dataset)"
        )


def _test_ivfpq_accuracy(
    model_path: Path, vectors: torch.Tensor, n_test: int = 10, k: int = 50
) -> None:
    """Test IVFPQ accuracy by searching for random vectors and checking if their indices are returned."""
    from ivfpq.ivf import search_ivf, load_centroids_binary
    import json
    import random

    logger.info(f"IVFPQ Accuracy Test:")
    logger.info(f"  Test queries: {n_test}")
    logger.info(f"  k (top results): {k}")

    # Load metadata
    metadata_file = model_path / "ivf_metadata.json"
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Load centroids
    centroids_file = model_path / "centroids.bin"
    centroids = load_centroids_binary(centroids_file)

    # Test with different n_probe values
    n_probe_values = [1, 2, 4, 8, 16]

    for n_probe in n_probe_values:
        if n_probe > metadata["n_partitions"]:
            continue

        correct_retrievals = 0
        total_tests = 0

        for _ in range(n_test):
            # Pick a random vector index
            random_idx = random.randint(0, vectors.shape[0] - 1)
            query_vector = vectors[random_idx]

            try:
                vector_ids, distances = search_ivf(
                    query_vector=query_vector,
                    centroids=centroids,
                    n_probe=n_probe,
                    model_path=model_path,
                    k_per_partition=k,
                    verbose=False,
                )

                # Check if the original index is in the returned results
                if random_idx in vector_ids:
                    correct_retrievals += 1

                total_tests += 1

            except Exception as e:
                logger.warning(f"Search failed for query index {random_idx}: {e}")

        accuracy = correct_retrievals / total_tests if total_tests > 0 else 0

        logger.info(
            f"  n_probe={n_probe}: {correct_retrievals}/{total_tests} correct ({accuracy:.1%} accuracy)"
        )


@app.command()
def test(
    vectors_path: Path = typer.Argument(
        ..., help="Path to vectors.npy file", exists=True
    ),
    model_path: Path = typer.Argument(
        ..., help="Path to trained IVFPQ model directory", exists=True
    ),
    max_vectors: Optional[int] = typer.Option(
        None, help="Maximum number of vectors to test (None = all)"
    ),
    n_queries: int = typer.Option(10, help="Number of test queries to run"),
) -> None:
    """
    Test IVFPQ search performance on trained model.

    This command tests the trained IVFPQ model using residual vectors
    and reports search performance statistics.
    """
    logger.info(f"Testing IVFPQ model from {model_path}")

    # Load vectors for testing - note this is for testing, not training
    vectors = _load_vectors(vectors_path, max_vectors)
    vectors_tensor = torch.from_numpy(vectors)

    logger.info(
        f"Loaded {vectors_tensor.shape[0]:,} vectors of dimension {vectors_tensor.shape[1]} for testing"
    )

    # Test search performance with different n_probe values
    logger.info("=== Testing IVFPQ Search Performance ===")
    _test_ivfpq_search(model_path, vectors_tensor, [1, 2, 4, 8, 16], n_queries)

    # Test accuracy by searching for random vectors
    logger.info("\n=== Testing IVFPQ Accuracy ===")
    _test_ivfpq_accuracy(model_path, vectors_tensor, n_queries)

    logger.info("IVFPQ performance testing completed")


if __name__ == "__main__":
    app()
