#!/usr/bin/env python3

"""
Training script for IVFPQ (Inverted File Index Product Quantization) models.

This script extends the PQ training functionality to include IVF index training,
enabling the complete IVFPQ pipeline for approximate nearest neighbor search.
Supports both Python and ONNX k-means implementations.

Features:
- Train PQ models on vectors using Python or ONNX k-means
- Train IVF index for dataset partitioning using Python or ONNX k-means
- Export models and indices for browser consumption
- Performance testing and validation
- Support for real vector datasets
- Choice between Python k-means (faster training) and ONNX k-means (browser compatible)
"""

import typer
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Optional
import json

from cytoverse.ivfpq import ProductQuantizer, InvertedFileIndex, IVFPQ
from cytoverse.ivfpq.pq import train_pq_codebooks, export_pq_models

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
def pq_train(
    vectors_path: Path = typer.Argument(..., help="Path to vectors.npy", exists=True),
    output_dir: Path = typer.Argument(
        ..., help="Output directory for trained model and codebooks"
    ),
    m: int = typer.Option(
        64, help="Number of subquantizers (must divide vector dimension)"
    ),
    k: int = typer.Option(
        256, help="Number of centroids per subquantizer (codebook size)"
    ),
    max_vectors: Optional[int] = typer.Option(
        None, help="Maximum number of vectors to use for training (None = all)"
    ),
    n_iterations: int = typer.Option(
        50, help="Number of k-means iterations per subquantizer"
    ),
    onnx_kmeans: bool = typer.Option(
        False, help="Use ONNX k-means instead of Python k-means (default: False)"
    ),
) -> None:
    """
    Train a Product Quantization model on vectors using Python or ONNX k-means.
    """
    vectors = _load_vectors(vectors_path, max_vectors)
    vectors_tensor = torch.from_numpy(vectors)
    d = vectors_tensor.shape[1]

    kmeans_type = "ONNX" if onnx_kmeans else "Python"
    logger.info(f"Training PQ with {kmeans_type} k-means: d={d}, m={m}, k={k}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train PQ codebooks using specified k-means method
    result = train_pq_codebooks(
        vectors_tensor,
        m=m,
        k=k,
        max_iterations=n_iterations,
        output_dir=output_dir,
        save_binary=True,
        use_onnx_kmeans=onnx_kmeans,
    )

    codebooks = result["codebooks"]
    metadata = result["metadata"]

    # Test reconstruction error
    _test_pq_reconstruction_with_codebooks(codebooks, vectors_tensor, m, k)

    # Export ONNX models
    export_pq_models(codebooks, output_dir)

    logger.info(f"PQ training completed with ONNX k-means")
    logger.info(f"Exported PQ models and artifacts to {output_dir}")


def _test_pq_reconstruction_with_codebooks(
    codebooks: torch.Tensor, vectors: torch.Tensor, m: int, k: int
) -> None:
    """Test PQ reconstruction error using trained codebooks."""
    # Use a subset for testing
    n_test = min(1000, vectors.shape[0])
    test_vectors = vectors[:n_test]

    d = vectors.shape[1]
    d_sub = d // m

    # Encode vectors manually using codebooks
    batch_size = test_vectors.shape[0]
    test_vectors_split = test_vectors.view(batch_size, m, d_sub)

    codes = torch.zeros(batch_size, m, dtype=torch.long)
    reconstructed = torch.zeros_like(test_vectors)

    for i in range(m):
        # Compute distances to all centroids in this subspace
        subvectors = test_vectors_split[:, i]  # [N, d_sub]
        codebook = codebooks[i]  # [k, d_sub]

        # Compute squared distances
        distances = torch.cdist(subvectors, codebook, p=2.0) ** 2  # [N, k]

        # Find nearest centroids
        codes[:, i] = torch.argmin(distances, dim=1)

        # Reconstruct using nearest centroids
        reconstructed[:, i * d_sub : (i + 1) * d_sub] = codebook[codes[:, i]]

    # Calculate errors
    mse = torch.mean((test_vectors - reconstructed) ** 2).item()
    relative_error = mse / torch.mean(test_vectors**2).item()

    # Calculate compression ratio
    original_bits = test_vectors.numel() * 32  # 32-bit floats
    compressed_bits = codes.numel() * 8  # 8-bit codes
    compression_ratio = original_bits / compressed_bits

    logger.info(f"PQ Reconstruction Results:")
    logger.info(f"  Test vectors: {n_test}")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  Relative error: {relative_error:.4f} ({relative_error * 100:.2f}%)")
    logger.info(f"  Compression ratio: {compression_ratio:.1f}x")


@app.command()
def ivf_train(
    vectors_path: Path = typer.Argument(..., help="Path to vectors.npy", exists=True),
    output_dir: Path = typer.Argument(..., help="Output directory for trained index"),
    n_partitions: int = typer.Option(256, help="Number of partitions for IVF index"),
    max_vectors: Optional[int] = typer.Option(
        50000,
        help="Maximum number of vectors to use for training (default: 50k for memory efficiency)",
    ),
    n_iterations: int = typer.Option(
        30, help="Number of k-means iterations for coarse quantization"
    ),
    onnx_kmeans: bool = typer.Option(
        False, help="Use ONNX k-means instead of Python k-means (default: False)"
    ),
) -> None:
    """
    Train an Inverted File Index on vectors using Python or ONNX k-means.
    """
    vectors = _load_vectors(vectors_path, max_vectors)
    vectors_tensor = torch.from_numpy(vectors)
    vector_ids_tensor = torch.arange(vectors.shape[0], dtype=torch.int32)
    d = vectors_tensor.shape[1]

    kmeans_type = "ONNX" if onnx_kmeans else "Python"
    logger.info(f"Training IVF with {kmeans_type} k-means: d={d}, n_partitions={n_partitions}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and train IVF index
    ivf = InvertedFileIndex(d=d, n_partitions=n_partitions)
    ivf.train_ivf(
        vectors_tensor, vector_ids_tensor, n_iterations=n_iterations, verbose=True, use_onnx_kmeans=onnx_kmeans
    )

    _test_ivf_search(ivf, vectors_tensor, n_probe_values=[1, 2, 4, 8])

    # Save IVF index
    ivf_model_path = output_dir / "model.pkl"
    ivf.save(ivf_model_path)
    logger.info(f"Saved IVF index to {ivf_model_path}")

    # Export metadata
    metadata_path = output_dir / "metadata.json"
    metadata = ivf.export_metadata()
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved IVF metadata to {metadata_path}")

    # Export centroids for browser use in binary format (expected by TypeScript)
    centroids_path = output_dir / "centroids.bin"
    centroids_data = ivf.centroids.detach().cpu().numpy().astype(np.float32)

    # Write binary format: n_partitions (4 bytes), d (4 bytes), then centroids data
    with open(centroids_path, "wb") as f:
        # Write header: n_partitions and d as uint32
        n_partitions, d = centroids_data.shape
        f.write(np.array([n_partitions], dtype=np.uint32).tobytes())
        f.write(np.array([d], dtype=np.uint32).tobytes())
        # Write centroids data
        f.write(centroids_data.tobytes())

    logger.info(f"Saved centroids to {centroids_path} (binary format for TypeScript)")


def _test_ivf_search(
    ivf: InvertedFileIndex, vectors: torch.Tensor, n_probe_values: list[int]
) -> None:
    """Test IVF search performance."""
    n_test = min(100, vectors.shape[0])
    test_queries = vectors[:n_test]

    logger.info(f"IVF Search Results:")
    logger.info(f"  Test queries: {n_test}")
    logger.info(f"  Total partitions: {ivf.n_partitions}")

    stats = ivf.get_partition_stats()
    logger.info(f"  Partition stats:")
    logger.info(f"    Mean size: {stats['mean_size']:.1f}")
    logger.info(f"    Min/Max size: {stats['min_size']}/{stats['max_size']}")
    logger.info(
        f"    Empty partitions: {stats['empty_partitions']}/{stats['n_partitions']}"
    )

    for n_probe in n_probe_values:
        if n_probe > ivf.n_partitions:
            continue

        partitions = ivf.search_partitions(test_queries, n_probe=n_probe)

        # Calculate average partition sizes
        total_vectors_searched = 0
        for query_partitions in partitions:
            for partition_id in query_partitions:
                total_vectors_searched += len(
                    ivf.get_partition_vector_ids(partition_id)
                )

        avg_vectors_per_query = total_vectors_searched / len(partitions)
        search_fraction = avg_vectors_per_query / stats["total_vectors"]

        logger.info(
            f"  n_probe={n_probe}: avg {avg_vectors_per_query:.1f} vectors/query ({search_fraction:.2%} of dataset)"
        )


@app.command()
def ivfpq_export(
    models_path: Path = typer.Argument(
        ..., help="Path to directory containing pq/ and ivf/ model folders", exists=True
    ),
    vectors_path: Path = typer.Argument(
        ..., help="Path to vectors.npy file", exists=True
    ),
    max_vectors: Optional[int] = typer.Option(
        None, help="Maximum number of vectors to use (None = all)"
    ),
    test_performance: bool = typer.Option(
        True, help="Test performance with sample queries"
    ),
) -> None:
    """
    Export browser-ready IVFPQ assets from pre-trained PQ and IVF models.

    This command:
    1. Loads pre-trained PQ model from models_path/pq/
    2. Loads pre-trained IVF model from models_path/ivf/
    3. Creates IVFPQ partitions with PQ-encoded vectors
    4. Exports to models_path/ivfpq/partitions/ and centroid index
    5. Includes all metadata needed for TypeScript/Python search
    """
    # Validate model directories exist
    pq_model_dir = models_path / "pq"
    ivf_model_dir = models_path / "ivf"

    if not pq_model_dir.exists():
        raise ValueError(f"PQ model directory not found: {pq_model_dir}")
    if not ivf_model_dir.exists():
        raise ValueError(f"IVF model directory not found: {ivf_model_dir}")

    logger.info(f"Loading pre-trained models from {models_path}")

    # Load vectors and vector IDs
    vectors = _load_vectors(vectors_path, max_vectors)
    vectors_tensor = torch.from_numpy(vectors)
    vector_ids_tensor = torch.arange(vectors.shape[0], dtype=torch.int32)

    logger.info(
        f"Loaded {vectors_tensor.shape[0]:,} vectors of dimension {vectors_tensor.shape[1]}"
    )

    # Create IVFPQ instance (it will load the models automatically)
    logger.info("Creating IVFPQ from pre-trained components")
    ivfpq = IVFPQ(models_path)

    # Encode vectors with the pre-trained models
    logger.info("Encoding vectors with IVFPQ")
    ivfpq.encode_vectors(vectors_tensor, vector_ids_tensor, verbose=True)

    # Export browser assets
    logger.info("=== Exporting Browser Assets ===")
    ivfpq.export(models_path)

    # Test performance with sample queries
    if test_performance:
        logger.info("=== Testing Performance ===")
        _test_ivfpq_performance(ivfpq, vectors_tensor)

    logger.info(
        f"IVFPQ browser assets exported successfully to {models_path / 'ivfpq'}"
    )
    logger.info(f"Partitions available at: {models_path / 'ivfpq' / 'partitions'}")
    logger.info(
        f"Use this directory for browser-based search with the PQ and IVF models"
    )


def _test_ivfpq_performance(
    ivfpq: IVFPQ, vectors_tensor: torch.Tensor, n_queries: int = 10
) -> None:
    """Test IVFPQ performance with random queries."""
    # Sample random query vectors
    n_vectors = vectors_tensor.shape[0]
    query_indices = torch.randperm(n_vectors)[:n_queries]
    query_vectors = vectors_tensor[query_indices]

    logger.info(f"Testing with {n_queries} random query vectors")

    # Test partition selection with different n_probe values
    for n_probe in [1, 2, 4, 8]:
        total_candidates = 0

        for i, query_vector in enumerate(query_vectors):
            selected_partitions = ivfpq.search_partitions(query_vector, n_probe=n_probe)

            # Count total candidate vectors
            candidates_in_partitions = sum(
                ivfpq.get_partition_data(pid)["size"] for pid in selected_partitions
            )
            total_candidates += candidates_in_partitions

        avg_candidates = total_candidates / n_queries
        search_fraction = avg_candidates / n_vectors

        logger.info(
            f"n_probe={n_probe}: {avg_candidates:.0f} avg candidates ({search_fraction:.1%} of dataset)"
        )

    # Show partition statistics
    stats = ivfpq.get_partition_stats()
    logger.info(
        f"Dataset organized into {stats['non_empty_partitions']}/{stats['total_partitions']} partitions"
    )
    logger.info(f"Average partition size: {stats['avg_partition_size']:.0f} vectors")


if __name__ == "__main__":
    app()
