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


@app.command()
def train_ivfpq(
    vectors_path: Path = typer.Argument(
        ..., help="Path to vectors file (.npy or .parquet)", exists=True
    ),
    sample_ids: Path = typer.Argument(None, help="Path to sample IDs file (.npy)"),
    output_dir: Path = typer.Argument(..., help="Output directory for trained models"),
    # PQ parameters
    m: int = typer.Option(
        64, help="Number of subquantizers (must divide vector dimension)"
    ),
    k: int = typer.Option(
        256, help="Number of centroids per subquantizer (codebook size)"
    ),
    # IVF parameters
    n_clusters: int = typer.Option(256, help="Number of partitions for IVF index"),
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
    test_reconstruction: bool = typer.Option(
        True, help="Test reconstruction error and search performance"
    ),
    export_onnx: bool = typer.Option(True, help="Export trained models to ONNX format"),
) -> None:
    """
    Train complete IVFPQ models (both PQ and IVF) on vectors and export assets.
    """
    embeddings, ids = _load_vectors(vectors_path, sample_ids, max_vectors)
    embeddings_tensor = torch.from_numpy(embeddings)
    sample_ids_tensor = torch.from_numpy(ids)
    d = embeddings_tensor.shape[1]

    logger.info(f"Training IVFPQ with parameters:")
    logger.info(f"  PQ: d={d}, m={m}, k={k}")
    logger.info(f"  IVF: n_partitions={n_clusters}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train PQ model
    logger.info("=== Training Product Quantization ===")
    pq = ProductQuantizer(d=d, m=m, k=k)
    pq.train(embeddings_tensor, n_iterations=pq_iterations, verbose=True)

    # Save PQ model
    pq_path = output_dir / "pq_model.pkl"
    pq.save(pq_path)
    logger.info(f"Saved PQ model to {pq_path}")

    # Train IVF index
    logger.info("=== Training Inverted File Index ===")
    ivf = InvertedFileIndex(d=d, n_partitions=n_clusters)
    ivf.train_ivf(
        embeddings_tensor, sample_ids_tensor, n_iterations=ivf_iterations, verbose=True
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

        # Export PQ browser assets (similar to pq_train function)
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

        pq_metadata_path = output_dir / "pq_metadata.json"
        with open(pq_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Export IVF assets
        metadata_path = output_dir / "ivf_metadata.json"
        metadata = ivf.export_metadata()
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        centroids_path = output_dir / "coarse_centroids.npy"
        np.save(centroids_path, ivf.centroids.detach().cpu().numpy())

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
    vectors_path: Optional[Path] = typer.Option(
        None, help="Path to test vectors (uses training data if None)"
    ),
    vector_ids_path: Optional[Path] = typer.Option(
        None, help="Path to vector IDs (generates indices if None)"
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
    if vectors_path is None:
        # Look for embeddings in common locations
        possible_paths = [
            Path("data/scimilarity/embeddings.npy"),
            Path("data/sample.h5ad"),  # Will need to be processed
        ]

        vectors_path = None
        for path in possible_paths:
            if path.exists():
                vectors_path = path
                break

        if vectors_path is None:
            logger.error("No test vectors found. Please specify --vectors-path")
            return

    # Handle vector_ids - if not provided, try to find or generate them
    if vector_ids_path is None:
        # Try to find vector_ids in common locations
        possible_id_paths = [
            Path("data/scimilarity/vector_ids.npy"),
            model_dir / "vector_ids.npy",
        ]

        for path in possible_id_paths:
            if path.exists():
                vector_ids_path = path
                break

        if vector_ids_path is None:
            # Generate temporary vector IDs file
            logger.info("No vector IDs found, generating sequential indices")
            temp_vectors = np.load(vectors_path)
            temp_ids = np.arange(len(temp_vectors))
            vector_ids_path = Path("/tmp/temp_vector_ids.npy")
            np.save(vector_ids_path, temp_ids)

    embeddings, vector_ids = _load_vectors(
        vectors_path, vector_ids_path, max_vectors=None
    )
    embeddings_tensor = torch.from_numpy(embeddings)
    d = embeddings_tensor.shape[1]

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


@app.command()
def train_complete_ivfpq(
    vectors_path: Path = typer.Argument(
        ..., help="Path to vectors file (.npy or .parquet)", exists=True
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for trained models"),
    # PQ parameters
    m: int = typer.Option(
        16, help="Number of subquantizers (must divide vector dimension)"
    ),
    k: int = typer.Option(
        256, help="Number of centroids per subquantizer (codebook size)"
    ),
    # IVF parameters
    n_clusters: int = typer.Option(256, help="Number of partitions for IVF index"),
    # Training parameters
    max_vectors: Optional[int] = typer.Option(
        None, help="Maximum number of vectors to use for training (None = all)"
    ),
    ivf_iterations: int = typer.Option(
        50, help="Number of k-means iterations for IVF training"
    ),
    pq_iterations: int = typer.Option(
        50, help="Number of k-means iterations for PQ training"
    ),
    vector_ids: Optional[Path] = typer.Option(
        None, help="Path to vector IDs file (.npy) - if None, uses indices 0...N-1"
    ),
    export_onnx: bool = typer.Option(True, help="Export trained models to ONNX format"),
    test_performance: bool = typer.Option(
        True, help="Test performance with sample queries"
    ),
) -> None:
    """
    Train a complete IVFPQ model combining IVF and PQ components.

    This command creates a unified IVFPQ model that handles dataset partitioning
    and PQ encoding within each partition, providing the foundation for
    efficient approximate nearest neighbor search.
    """
    # Handle vector_ids - if not provided, generate sequential indices
    if vector_ids is None:
        logger.info("No vector IDs provided, will generate sequential indices")
        temp_vectors = np.load(vectors_path)
        temp_ids = np.arange(len(temp_vectors))
        vector_ids_path = Path("/tmp/temp_vector_ids.npy")
        np.save(vector_ids_path, temp_ids)
        vector_ids = vector_ids_path

    embeddings, ids = _load_vectors(vectors_path, vector_ids, max_vectors)
    embeddings_tensor = torch.from_numpy(embeddings)
    vector_ids_tensor = torch.from_numpy(ids)
    d = embeddings_tensor.shape[1]

    logger.info(f"Training complete IVFPQ model with parameters:")
    logger.info(f"  Dimension: {d}")
    logger.info(f"  PQ: m={m}, k={k}")
    logger.info(f"  IVF: n_partitions={n_clusters}")
    logger.info(f"  Training vectors: {embeddings_tensor.shape[0]:,}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and train complete IVFPQ model
    ivfpq = IVFPQ(d=d, m=m, k=k, n_clusters=n_clusters)

    ivfpq.train(
        vectors=embeddings_tensor,
        vector_ids=vector_ids_tensor,
        ivf_iterations=ivf_iterations,
        pq_iterations=pq_iterations,
        verbose=True,
    )

    # Export browser assets (Arrow format) - this is now the primary export
    logger.info("=== Exporting Browser Assets (Arrow Format) ===")
    ivfpq.export_browser_assets(output_dir)

    # Test performance with sample queries
    if test_performance:
        logger.info("=== Testing Performance ===")
        _test_ivfpq_performance(ivfpq, embeddings_tensor)


@app.command()
def export_browser_assets(
    model_dir: Path = typer.Argument(
        ..., help="Directory containing trained IVFPQ model", exists=True
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for browser assets"),
) -> None:
    """
    Export trained IVFPQ model to Arrow format for browser consumption.

    This creates optimized browser assets including:
    - Individual partition files in Parquet format
    - Centroid index with partition metadata
    - ONNX models and codebooks for client-side inference
    """
    # Load the trained model
    ivfpq = IVFPQ.load(model_dir)

    # Export browser assets
    ivfpq.export_browser_assets(output_dir)

    logger.info(f"Browser assets exported successfully to {output_dir}")


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
        f"Dataset organized into {stats['non_empty_partitions']}/{stats['n_partitions']} partitions"
    )
    logger.info(f"Average partition size: {stats['avg_partition_size']:.0f} vectors")


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


if __name__ == "__main__":
    app()
