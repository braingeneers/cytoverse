#!/usr/bin/env python3

import typer
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cellspace.ivfpq.pq import ProductQuantizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Train Product Quantization models on embedding vectors",
    add_completion=False,
)


@app.command()
def train(
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

    logger.info(f"Training PQ with parameters: d={d}, m={m}, k={k}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and train PQ model
    pq = ProductQuantizer(d=d, m=m, k=k)

    # Train the model
    logger.info("Starting PQ training...")
    pq.train_pq(embeddings_tensor, n_iterations=n_iterations, verbose=True)

    # Save the trained model
    model_path = output_dir / "pq_model.pkl"
    pq.save(model_path)
    logger.info(f"Saved trained model to {model_path}")

    # Test reconstruction error if requested
    if test_reconstruction:
        logger.info("Computing reconstruction error...")

        # Use a subset for testing if dataset is large
        test_size = min(1000, embeddings_tensor.shape[0])
        test_indices = torch.randperm(embeddings_tensor.shape[0])[:test_size]
        test_vectors = embeddings_tensor[test_indices]

        mse = pq.compute_reconstruction_error(test_vectors)
        logger.info(f"Mean squared reconstruction error: {mse:.6f}")

        # Compute compression ratio
        original_bits = d * 32  # 32 bits per float
        compressed_bits = m * np.log2(k)  # log2(k) bits per subquantizer
        compression_ratio = original_bits / compressed_bits
        logger.info(
            f"Compression ratio: {compression_ratio:.1f}x ({original_bits:.0f} -> {compressed_bits:.1f} bits)"
        )

    # Export to ONNX if requested
    if export_onnx:
        logger.info("Exporting to ONNX format...")
        onnx_path = output_dir / "pq_encoder.onnx"
        pq.export_onnx(onnx_path, batch_size=-1)  # Dynamic batch size

        # Test ONNX model
        logger.info("Validating ONNX export...")
        import onnxruntime as ort

        # Create test input
        test_input = embeddings_tensor[:5].numpy()  # Test with 5 vectors

        # Run PyTorch model
        with torch.no_grad():
            torch_codes = pq(torch.from_numpy(test_input))

        # Run ONNX model
        ort_session = ort.InferenceSession(str(onnx_path))
        onnx_codes = ort_session.run(None, {"input": test_input})[0]

        # Compare results
        codes_match = np.allclose(torch_codes.numpy(), onnx_codes)
        logger.info(f"ONNX validation: {'✅ PASSED' if codes_match else '❌ FAILED'}")

        if codes_match:
            logger.info(f"ONNX model exported successfully to {onnx_path}")
        else:
            logger.error("ONNX model outputs don't match PyTorch model!")

    # Export codebooks and metadata for browser
    logger.info("Exporting codebooks and metadata for browser...")

    # Export codebooks as binary file
    codebooks_path = output_dir / "codebooks.bin"
    codebooks_array = pq.codebooks.data.cpu().numpy().astype(np.float32)
    with open(codebooks_path, "wb") as f:
        f.write(codebooks_array.tobytes())
    logger.info(f"Saved codebooks to {codebooks_path}")

    # Export metadata as JSON
    metadata = {
        "d": pq.d,
        "m": pq.m,
        "k": pq.k,
        "d_sub": pq.d_sub,
        "compression_ratio": (pq.d * 32) / (pq.m * 8),
        "codebooks_shape": [pq.m, pq.k, pq.d_sub],
        "codebooks_size": int(pq.m * pq.k * pq.d_sub),
        "version": "1.0",
    }

    metadata_path = output_dir / "metadata.json"
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info("🎉 PQ training completed successfully!")


@app.command()
def test(
    model_path: Path = typer.Argument(
        ..., help="Path to trained PQ model (.pkl)", exists=True
    ),
    embeddings_path: Path = typer.Argument(
        ..., help="Path to test embeddings file (.npy or .parquet)", exists=True
    ),
    n_test: int = typer.Option(1000, help="Number of test vectors to use"),
) -> None:
    """
    Test a trained PQ model on embedding vectors.
    """
    logger.info(f"Loading trained model from {model_path}")
    pq = ProductQuantizer.load(model_path)

    logger.info(f"Loading test embeddings from {embeddings_path}")

    # Load embeddings
    if embeddings_path.suffix == ".npy":
        embeddings = np.load(embeddings_path)
    elif embeddings_path.suffix == ".parquet":
        df = pd.read_parquet(embeddings_path)
        embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
        embeddings = df[embedding_cols].values
    else:
        raise ValueError(f"Unsupported file format: {embeddings_path.suffix}")

    # Use subset for testing
    if embeddings.shape[0] > n_test:
        indices = np.random.choice(embeddings.shape[0], n_test, replace=False)
        embeddings = embeddings[indices]

    embeddings_tensor = torch.from_numpy(embeddings.astype(np.float32))

    # Test encoding/decoding
    logger.info("Testing encoding and decoding...")
    with torch.no_grad():
        codes = pq(embeddings_tensor)
        reconstructed = pq.decode(codes)

    # Compute metrics
    mse = torch.mean((embeddings_tensor - reconstructed) ** 2).item()
    rmse = np.sqrt(mse)

    # Compute relative error
    original_norm = torch.norm(embeddings_tensor, dim=1).mean().item()
    relative_error = rmse / original_norm

    logger.info(f"Results on {embeddings_tensor.shape[0]} test vectors:")
    logger.info(f"  Mean Squared Error: {mse:.6f}")
    logger.info(f"  Root Mean Squared Error: {rmse:.6f}")
    logger.info(f"  Relative Error: {relative_error:.4f} ({relative_error*100:.2f}%)")

    # Compression info
    original_bits = pq.d * 32
    compressed_bits = pq.m * np.log2(pq.k)
    compression_ratio = original_bits / compressed_bits
    logger.info(
        f"  Compression: {compression_ratio:.1f}x ({original_bits:.0f} -> {compressed_bits:.1f} bits)"
    )


if __name__ == "__main__":
    app()
