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

import random
import typer
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Optional

from ivfpq.ivfpq import IVFPQ

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42  # Fixed seed for reproducibility

app = typer.Typer(
    help="Train IVFPQ models on vectors",
    add_completion=False,
)

# New helper for reproducibility
def _set_seed(seed: int) -> None:
    logger.info(f"Setting global random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # noqa: SIM108
        torch.cuda.manual_seed_all(seed)
        logger.info("CUDA available: seeded all CUDA devices")
    # MPS (Apple Silicon) shares CPU RNG; manual_seed suffices. Log if present.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # noqa: SIM108
        logger.info("MPS backend available: seeded via torch.manual_seed")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        pass


def _load_vectors(vectors_path: Path, max_vectors: Optional[int] = None) -> np.ndarray:
    """Load vectors and return as a numpy array."""
    logger.info(f"Loading vectors from {vectors_path}")

    # Use memory mapping when max_vectors is specified to avoid loading entire file
    if max_vectors is not None:
        vectors_mmap = np.load(vectors_path, mmap_mode="r")
        shape = vectors_mmap.shape
        logger.info(
            f"Memory-mapped {shape[0]} vectors of dimension {shape[1]} from .npy file"
        )

        # Only load the required vectors into memory
        actual_max = min(max_vectors, shape[0])
        logger.info(f"Loading first {actual_max} vectors into memory")
        vectors = np.array(vectors_mmap[:actual_max])
    else:
        # Load entire file when no limit specified
        vectors = np.load(vectors_path)
        logger.info(
            f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]} from .npy file"
        )

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
    sample_training_vectors: bool = typer.Option(
        False,
        help="Randomly sample training vectors instead of taking first n (default: False)",
    ),
    test: bool = typer.Option(
        False,
        help="Run performance tests on the trained IVFPQ model after training (default: False)",
    ),
) -> None:
    """Train complete IVFPQ model with residual vectors for browser consumption (deterministic)."""
    _set_seed(DEFAULT_SEED)

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
    _ = IVFPQ.build(
        vectors=vectors_tensor,
        output_dir=output_dir,
        n_partitions=None,  # Use default partitioning
        num_training_vectors=max_vectors_for_training,
        sample_training_vectors=sample_training_vectors,
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

    if test:
        logger.info("Running performance tests on trained IVFPQ model...")
        measure_recall(
            vectors_path=vectors_path,
            model_path=output_dir,
            max_vectors=max_vectors,
            n_queries=100,  # Default number of queries for testing
        )


@app.command()
def measure_recall(
    vectors_path: Path = typer.Argument(
        ..., help="Path to vectors.npy file", exists=True
    ),
    model_path: Path = typer.Argument(
        ..., help="Path to trained IVFPQ model directory", exists=True
    ),
    max_vectors: Optional[int] = typer.Option(
        None, help="Maximum number of vectors to test (None = all)"
    ),
    n_queries: int = typer.Option(100, help="Number of test queries to run"),
) -> None:
    """
    Test IVFPQ recall performance on trained model.

    This command tests the trained IVFPQ model using residual vectors
    and reports recall performance statistics.
    """
    _set_seed(DEFAULT_SEED)

    logger.info(f"Testing IVFPQ model from {model_path}")

    # Load vectors for testing
    vectors = _load_vectors(vectors_path, max_vectors)
    vectors_tensor = torch.from_numpy(vectors)

    logger.info(
        f"Loaded {vectors_tensor.shape[0]:,} vectors of dimension {vectors_tensor.shape[1]} for testing"
    )

    # Load the IVFPQ model
    ivfpq = IVFPQ.load(model_path)

    # Select random indices for testing
    total_vectors = vectors_tensor.shape[0]
    test_indices = random.sample(range(total_vectors), min(n_queries, total_vectors))

    logger.info(
        f"\n=== Testing Self-Accuracy on {len(test_indices)} randomly selected vectors ==="
    )

    # Test 1: Verify that searching for randomly selected vectors yields themselves as closest
    correct_self_matches = 0
    for idx in test_indices:
        query_vector = vectors_tensor[idx]

        try:
            vector_ids, _ = ivfpq.search(
                query_vector=query_vector,
                model_path=model_path,
                n_probe=8,  # Use reasonable n_probe for accuracy
            )

            # Check if the first result is the query vector itself
            if len(vector_ids) > 0 and vector_ids[0] == idx:
                correct_self_matches += 1
            else:
                # Find where the vector appears in the results
                if idx in vector_ids:
                    position = vector_ids.index(idx) + 1
                    logger.info(
                        f"  Vector {idx}: ✗ Not found as closest match. Found at position {position} in results"
                    )
                else:
                    logger.info(
                        f"  Vector {idx}: ✗ Not found in top {len(vector_ids)} results"
                    )

        except Exception as e:
            logger.warning(f"  Vector {idx}: Search failed - {e}")

    self_accuracy = (
        correct_self_matches / len(test_indices) if len(test_indices) > 0 else 0
    )
    logger.info(
        f"\nSelf-accuracy: {correct_self_matches}/{len(test_indices)} ({self_accuracy:.1%})"
    )

    # Test 2: Perturb the same vectors and verify they still find the original
    logger.info(
        f"\n=== Testing Perturbation Accuracy on same {len(test_indices)} vectors ==="
    )

    perturbation_scale = 0.01  # Small perturbation (1% of vector magnitude)
    correct_perturbed_matches = 0

    for idx in test_indices:
        original_vector = vectors_tensor[idx]

        # Create a small perturbation
        perturbation = (
            torch.randn_like(original_vector)
            * perturbation_scale
            * torch.norm(original_vector)
        )
        perturbed_vector = original_vector + perturbation

        try:
            vector_ids, _ = ivfpq.search(
                query_vector=perturbed_vector,
                model_path=model_path,
                n_probe=8,
            )

            # Check if the first result is the original unperturbed vector
            if len(vector_ids) > 0 and vector_ids[0] == idx:
                correct_perturbed_matches += 1
            else:
                # Find where the original vector appears in the results
                if idx in vector_ids:
                    position = vector_ids.index(idx) + 1
                    logger.info(
                        f"  Perturbed vector {idx}: ✗ Original not found as closest match. Found at position {position} in results"
                    )
                else:
                    logger.info(
                        f"  Perturbed vector {idx}: ✗ Original not found in top {len(vector_ids)} results"
                    )

        except Exception as e:
            logger.warning(f"  Perturbed vector {idx}: Search failed - {e}")

    perturbed_accuracy = (
        correct_perturbed_matches / len(test_indices) if len(test_indices) > 0 else 0
    )
    logger.info(
        f"\nPerturbation accuracy: {correct_perturbed_matches}/{len(test_indices)} ({perturbed_accuracy:.1%})"
    )

    logger.info("\nIVFPQ validation completed")


if __name__ == "__main__":
    app()
