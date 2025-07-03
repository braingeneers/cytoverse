#!/usr/bin/env python3

"""
Training script for IVFPQ (Inverted File Index Product Quantization) models.

This script extends the PQ training functionality to include IVF index training,
enabling the complete IVFPQ pipeline for approximate nearest neighbor search.

Features:
- Train PQ models on vectors
- Train IVF index for dataset partitioning
- Export models and indices for browser consumption
- Performance testing and validation
- Support for real vector datasets
"""

import typer
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Optional
import json

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cellspace.ivfpq.pq import ProductQuantizer
from src.cellspace.ivfpq.ivf import InvertedFileIndex
from src.cellspace.ivfpq.ivfpq import IVFPQ

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Train IVFPQ models on vectors",
    add_completion=False,
)


def _load_vectors(
    vectors_path: Path, vector_ids_path: Path, max_vectors: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load vectors and vector IDs from files and return as numpy arrays."""
    logger.info(f"Loading vectors from {vectors_path}")
    vectors = np.load(vectors_path)
    logger.info(
        f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]} from .npy file"
    )

    logger.info(f"Loading vector IDs from {vector_ids_path}")
    vector_ids = np.load(vector_ids_path)
    logger.info(f"Loaded {len(vector_ids)} vector IDs")

    if len(vector_ids) != vectors.shape[0]:
        raise ValueError(
            f"Number of vector IDs ({len(vector_ids)}) must match number of vectors ({vectors.shape[0]})"
        )

    # Limit number of vectors if specified (take first max_vectors)
    if max_vectors is not None and vectors.shape[0] > max_vectors:
        logger.info(f"Using first {max_vectors} vectors for training")
        vectors = vectors[:max_vectors]
        vector_ids = vector_ids[:max_vectors]

    return vectors.astype(np.float32), vector_ids


@app.command()
def pq_train(
    vectors_path: Path = typer.Argument(..., help="Path to vectors.npy", exists=True),
    vector_ids_path: Path = typer.Argument(
        ..., help="Path to vector IDs.npy", exists=True
    ),
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
) -> None:
    """
    Train a Product Quantization model on vectors.
    """
    embeddings, vector_ids = _load_vectors(vectors_path, vector_ids_path, max_vectors)
    embeddings_tensor = torch.from_numpy(embeddings)
    d = embeddings_tensor.shape[1]

    logger.info(f"Training PQ with parameters: d={d}, m={m}, k={k}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and train PQ model
    pq = ProductQuantizer(d=d, m=m, k=k)
    pq.train_pq(embeddings_tensor, n_iterations=n_iterations)

    # Test reconstruction error
    _test_pq_reconstruction(pq, embeddings_tensor)

    # Save PQ model
    pq_model_path = output_dir / "model.pkl"
    pq.save(pq_model_path)
    logger.info(f"Saved PQ model to {pq_model_path}")

    # Export ONNX model
    onnx_path = output_dir / "model.onnx"
    pq.eval()
    torch.onnx.export(
        pq,
        torch.zeros(1, pq.d),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
    )
    logger.info(f"PQ model exported to {onnx_path}")

    # Export codebooks for browser decoding
    codebooks_path = output_dir / "codebooks.bin"
    codebooks_np = pq.codebooks.detach().cpu().numpy()
    codebooks_np.astype(np.float32).tofile(codebooks_path)

    # Export metadata as JSON
    metadata = {
        "d": pq.d,
        "m": pq.m,
        "k": pq.k,
        "d_sub": pq.d_sub,
        "codebooks_shape": list(codebooks_np.shape),
        "is_trained": pq.is_trained,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Exported PQ browser assets:")
    logger.info(f"  Codebooks: {codebooks_path}")
    logger.info(f"  Metadata: {metadata_path}")


def _test_pq_reconstruction(pq: ProductQuantizer, embeddings: torch.Tensor) -> None:
    """Test PQ reconstruction error."""
    # Use a subset for testing
    n_test = min(1000, embeddings.shape[0])
    test_vectors = embeddings[:n_test]

    # Encode and decode
    codes = pq(test_vectors)
    reconstructed = pq.decode(codes)

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
    vector_ids_path: Path = typer.Argument(
        ..., help="Path to vector IDs file (.npy)", exists=True
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for trained index"),
    n_partitions: int = typer.Option(256, help="Number of partitions for IVF index"),
    max_vectors: Optional[int] = typer.Option(
        None, help="Maximum number of vectors to use for training (None = all)"
    ),
    n_iterations: int = typer.Option(
        50, help="Number of k-means iterations for coarse quantization"
    ),
) -> None:
    """
    Train an Inverted File Index on vectors.
    """
    vectors, vector_ids = _load_vectors(vectors_path, vector_ids_path, max_vectors)
    vectors_tensor = torch.from_numpy(vectors)
    vector_ids_tensor = torch.from_numpy(vector_ids)
    d = vectors_tensor.shape[1]

    logger.info(f"Training IVF with parameters: d={d}, n_partitions={n_partitions}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and train IVF index
    ivf = InvertedFileIndex(d=d, n_partitions=n_partitions)
    ivf.train_ivf(
        vectors_tensor, vector_ids_tensor, n_iterations=n_iterations, verbose=True
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

    # Export centroids for browser use
    centroids_path = output_dir / "centroids.arrow"
    centroids_df = pd.DataFrame(ivf.centroids.detach().cpu().numpy())
    centroids_df.to_parquet(centroids_path)
    logger.info(f"Saved centroids to {centroids_path}")


def _test_ivf_search(
    ivf: InvertedFileIndex, embeddings: torch.Tensor, n_probe_values: list[int]
) -> None:
    """Test IVF search performance."""
    n_test = min(100, embeddings.shape[0])
    test_queries = embeddings[:n_test]

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
                total_vectors_searched += len(ivf.get_partition_vectors(partition_id))

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
    vector_ids_path: Path = typer.Argument(
        ..., help="Path to vector_ids.npy file", exists=True
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
    vectors, vector_ids = _load_vectors(vectors_path, vector_ids_path, max_vectors)
    vectors_tensor = torch.from_numpy(vectors)
    vector_ids_tensor = torch.from_numpy(vector_ids)

    logger.info(f"Loaded {vectors_tensor.shape[0]:,} vectors of dimension {vectors_tensor.shape[1]}")

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

    logger.info(f"IVFPQ browser assets exported successfully to {models_path / 'ivfpq'}")
    logger.info(f"Partitions available at: {models_path / 'ivfpq' / 'partitions'}")
    logger.info(f"Use this directory for browser-based search with the PQ and IVF models")


def _test_ivfpq_performance(
    ivfpq: IVFPQ, embeddings_tensor: torch.Tensor, n_queries: int = 10
) -> None:
    """Test IVFPQ performance with random queries."""
    # Sample random query vectors
    n_vectors = embeddings_tensor.shape[0]
    query_indices = torch.randperm(n_vectors)[:n_queries]
    query_vectors = embeddings_tensor[query_indices]

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
