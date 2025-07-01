#!/usr/bin/env python3

"""
Training script for IVFPQ (Inverted File Index Product Quantization) models.

This script extends the PQ training functionality to include IVF index training,
enabling the complete IVFPQ pipeline for approximate nearest neighbor search.

Features:
- Train PQ models on embedding vectors
- Train IVF index for dataset partitioning
- Export models and indices for browser consumption
- Performance testing and validation
- Support for real embedding datasets
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Train IVFPQ models on embedding vectors",
    add_completion=False,
)


@app.command()
def train_pq(
    embeddings_path: Path = typer.Argument(
        ..., help="Path to embeddings file (.npy or .parquet)", exists=True
    ),
    output_dir: Path = typer.Argument(
        ..., help="Output directory for trained model and exports"
    ),
    m: int = typer.Option(
        64, help="Number of subquantizers (must divide embedding dimension)"
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
    export_onnx: bool = typer.Option(True, help="Export trained model to ONNX format"),
    test_reconstruction: bool = typer.Option(
        True, help="Test reconstruction error on a subset of data"
    ),
) -> None:
    """
    Train a Product Quantization model on embedding vectors.
    """
    embeddings_tensor, d = _load_embeddings(embeddings_path, max_vectors)

    logger.info(f"Training PQ with parameters: d={d}, m={m}, k={k}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and train PQ model
    pq = ProductQuantizer(d=d, m=m, k=k)
    pq.train_pq(embeddings_tensor, n_iterations=n_iterations, verbose=True)

    # Save trained model
    model_path = output_dir / "pq_model.pkl"
    pq.save(model_path)
    logger.info(f"Saved PQ model to {model_path}")

    # Export ONNX model
    if export_onnx:
        onnx_path = output_dir / "pq_model.onnx"
        pq.export_onnx(onnx_path, batch_size=-1)

        # Export codebooks for browser decoding
        _export_browser_assets(pq, output_dir)

    # Test reconstruction error
    if test_reconstruction:
        _test_pq_reconstruction(pq, embeddings_tensor)


@app.command()
def train_ivf(
    embeddings_path: Path = typer.Argument(
        ..., help="Path to embeddings file (.npy or .parquet)", exists=True
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for trained index"),
    n_clusters: int = typer.Option(256, help="Number of coarse clusters for IVF index"),
    max_vectors: Optional[int] = typer.Option(
        None, help="Maximum number of vectors to use for training (None = all)"
    ),
    n_iterations: int = typer.Option(
        50, help="Number of k-means iterations for coarse quantization"
    ),
    vector_ids: Optional[Path] = typer.Option(
        None, help="Path to vector IDs file (.npy) - if None, uses indices 0...N-1"
    ),
) -> None:
    """
    Train an Inverted File Index on embedding vectors.
    """
    embeddings_tensor, d = _load_embeddings(embeddings_path, max_vectors)

    # Load vector IDs if provided
    vector_ids_tensor = None
    if vector_ids is not None:
        ids = np.load(vector_ids)
        vector_ids_tensor = torch.from_numpy(ids)
        logger.info(f"Loaded {len(ids)} vector IDs")

        if len(ids) != embeddings_tensor.shape[0]:
            raise ValueError(
                f"Number of vector IDs ({len(ids)}) must match number of embeddings ({embeddings_tensor.shape[0]})"
            )

    logger.info(f"Training IVF with parameters: d={d}, n_clusters={n_clusters}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and train IVF index
    ivf = InvertedFileIndex(d=d, n_clusters=n_clusters)
    ivf.train_ivf(
        embeddings_tensor, vector_ids_tensor, n_iterations=n_iterations, verbose=True
    )

    # Save trained index
    index_path = output_dir / "ivf_index.pkl"
    ivf.save(index_path)
    logger.info(f"Saved IVF index to {index_path}")

    # Export metadata
    metadata_path = output_dir / "ivf_metadata.json"
    metadata = ivf.export_metadata()
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved IVF metadata to {metadata_path}")

    # Export coarse centroids for browser use
    centroids_path = output_dir / "coarse_centroids.npy"
    np.save(centroids_path, ivf.coarse_centroids.detach().cpu().numpy())
    logger.info(f"Saved coarse centroids to {centroids_path}")


@app.command()
def train_ivfpq(
    embeddings_path: Path = typer.Argument(
        ..., help="Path to embeddings file (.npy or .parquet)", exists=True
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for trained models"),
    # PQ parameters
    m: int = typer.Option(
        64, help="Number of subquantizers (must divide embedding dimension)"
    ),
    k: int = typer.Option(
        256, help="Number of centroids per subquantizer (codebook size)"
    ),
    # IVF parameters
    n_clusters: int = typer.Option(256, help="Number of coarse clusters for IVF index"),
    # Training parameters
    max_vectors: Optional[int] = typer.Option(
        None, help="Maximum number of vectors to use for training (None = all)"
    ),
    pq_iterations: int = typer.Option(
        50, help="Number of k-means iterations for PQ training"
    ),
    ivf_iterations: int = typer.Option(
        50, help="Number of k-means iterations for IVF training"
    ),
    vector_ids: Optional[Path] = typer.Option(
        None, help="Path to vector IDs file (.npy) - if None, uses indices 0...N-1"
    ),
    export_onnx: bool = typer.Option(True, help="Export trained models to ONNX format"),
    test_reconstruction: bool = typer.Option(
        True, help="Test reconstruction error and search performance"
    ),
) -> None:
    """
    Train complete IVFPQ models (both PQ and IVF) on embedding vectors.
    """
    embeddings_tensor, d = _load_embeddings(embeddings_path, max_vectors)

    # Load vector IDs if provided
    vector_ids_tensor = None
    if vector_ids is not None:
        ids = np.load(vector_ids)
        vector_ids_tensor = torch.from_numpy(ids)
        logger.info(f"Loaded {len(ids)} vector IDs")

        if len(ids) != embeddings_tensor.shape[0]:
            raise ValueError(
                f"Number of vector IDs ({len(ids)}) must match number of embeddings ({embeddings_tensor.shape[0]})"
            )

    logger.info(f"Training IVFPQ with parameters:")
    logger.info(f"  PQ: d={d}, m={m}, k={k}")
    logger.info(f"  IVF: n_clusters={n_clusters}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train PQ model
    logger.info("=== Training Product Quantization ===")
    pq = ProductQuantizer(d=d, m=m, k=k)
    pq.train_pq(embeddings_tensor, n_iterations=pq_iterations, verbose=True)

    # Save PQ model
    pq_path = output_dir / "pq_model.pkl"
    pq.save(pq_path)
    logger.info(f"Saved PQ model to {pq_path}")

    # Train IVF index
    logger.info("=== Training Inverted File Index ===")
    ivf = InvertedFileIndex(d=d, n_clusters=n_clusters)
    ivf.train_ivf(
        embeddings_tensor, vector_ids_tensor, n_iterations=ivf_iterations, verbose=True
    )

    # Save IVF index
    ivf_path = output_dir / "ivf_index.pkl"
    ivf.save(ivf_path)
    logger.info(f"Saved IVF index to {ivf_path}")

    # Export for browser use
    if export_onnx:
        logger.info("=== Exporting Browser Assets ===")

        # Export PQ ONNX model
        onnx_path = output_dir / "pq_model.onnx"
        pq.export_onnx(onnx_path, batch_size=-1)

        # Export PQ browser assets
        _export_browser_assets(pq, output_dir)

        # Export IVF assets
        metadata_path = output_dir / "ivf_metadata.json"
        metadata = ivf.export_metadata()
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        centroids_path = output_dir / "coarse_centroids.npy"
        np.save(centroids_path, ivf.coarse_centroids.detach().cpu().numpy())

        logger.info(f"Exported browser assets to {output_dir}")

    # Performance testing
    if test_reconstruction:
        logger.info("=== Performance Testing ===")
        _test_pq_reconstruction(pq, embeddings_tensor)
        _test_ivf_search(ivf, embeddings_tensor, n_probe_values=[1, 2, 4, 8])


@app.command()
def test_trained_models(
    model_dir: Path = typer.Argument(
        ..., help="Directory containing trained models", exists=True
    ),
    embeddings_path: Optional[Path] = typer.Option(
        None, help="Path to test embeddings (uses training data if None)"
    ),
    n_test_vectors: int = typer.Option(
        100, help="Number of vectors to use for testing"
    ),
) -> None:
    """
    Test previously trained PQ and/or IVF models.
    """
    # Look for trained models
    pq_path = model_dir / "pq_model.pkl"
    ivf_path = model_dir / "ivf_index.pkl"

    pq_exists = pq_path.exists()
    ivf_exists = ivf_path.exists()

    if not pq_exists and not ivf_exists:
        logger.error(f"No trained models found in {model_dir}")
        return

    # Load test embeddings
    if embeddings_path is None:
        # Look for embeddings in common locations
        possible_paths = [
            Path("data/scimilarity/embeddings.npy"),
            Path("data/sample.h5ad"),  # Will need to be processed
        ]

        embeddings_path = None
        for path in possible_paths:
            if path.exists():
                embeddings_path = path
                break

        if embeddings_path is None:
            logger.error("No test embeddings found. Please specify --embeddings-path")
            return

    embeddings_tensor, d = _load_embeddings(embeddings_path, max_vectors=None)

    # Subsample for testing
    if embeddings_tensor.shape[0] > n_test_vectors:
        indices = torch.randperm(embeddings_tensor.shape[0])[:n_test_vectors]
        test_embeddings = embeddings_tensor[indices]
    else:
        test_embeddings = embeddings_tensor

    logger.info(f"Testing with {test_embeddings.shape[0]} vectors")

    # Test PQ model
    if pq_exists:
        logger.info("=== Testing PQ Model ===")
        pq = ProductQuantizer.load(pq_path)
        _test_pq_reconstruction(pq, test_embeddings)

    # Test IVF index
    if ivf_exists:
        logger.info("=== Testing IVF Index ===")
        ivf = InvertedFileIndex.load(ivf_path)
        _test_ivf_search(ivf, test_embeddings, n_probe_values=[1, 2, 4, 8])


def _load_embeddings(
    embeddings_path: Path, max_vectors: Optional[int]
) -> tuple[torch.Tensor, int]:
    """Load embeddings from file and return as tensor with dimension."""
    logger.info(f"Loading embeddings from {embeddings_path}")

    # Load embeddings based on file extension
    if embeddings_path.suffix == ".npy":
        embeddings = np.load(embeddings_path)
        logger.info(
            f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} from .npy file"
        )
    elif embeddings_path.suffix == ".parquet":
        df = pd.read_parquet(embeddings_path)
        # Assume embedding columns are named 'embedding_0', 'embedding_1', etc.
        embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
        if not embedding_cols:
            raise ValueError(
                "No embedding columns found in parquet file. Expected columns like 'embedding_0', 'embedding_1', etc."
            )
        embeddings = df[embedding_cols].values
        logger.info(
            f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} from .parquet file"
        )
    else:
        raise ValueError(
            f"Unsupported file format: {embeddings_path.suffix}. Use .npy or .parquet"
        )

    # Limit number of vectors if specified
    if max_vectors is not None and embeddings.shape[0] > max_vectors:
        logger.info(f"Using random subset of {max_vectors} vectors for training")
        indices = np.random.choice(embeddings.shape[0], max_vectors, replace=False)
        embeddings = embeddings[indices]

    # Convert to torch tensor
    embeddings_tensor = torch.from_numpy(embeddings.astype(np.float32))
    d = embeddings_tensor.shape[1]

    return embeddings_tensor, d


def _export_browser_assets(pq: ProductQuantizer, output_dir: Path) -> None:
    """Export PQ assets for browser consumption."""
    # Export codebooks as binary file
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

    metadata_path = output_dir / "pq_metadata.json"
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


def _test_ivf_search(
    ivf: InvertedFileIndex, embeddings: torch.Tensor, n_probe_values: list[int]
) -> None:
    """Test IVF search performance."""
    n_test = min(100, embeddings.shape[0])
    test_queries = embeddings[:n_test]

    logger.info(f"IVF Search Results:")
    logger.info(f"  Test queries: {n_test}")
    logger.info(f"  Total clusters: {ivf.n_clusters}")

    stats = ivf.get_cluster_stats()
    logger.info(f"  Cluster stats:")
    logger.info(f"    Mean size: {stats['mean_size']:.1f}")
    logger.info(f"    Min/Max size: {stats['min_size']}/{stats['max_size']}")
    logger.info(f"    Empty clusters: {stats['empty_clusters']}")

    for n_probe in n_probe_values:
        if n_probe > ivf.n_clusters:
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


if __name__ == "__main__":
    app()
