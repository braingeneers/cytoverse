"""
Integration tests for Product Quantization on real embedding data.

These tests validate PQ functionality using actual embedding files from the project,
providing end-to-end validation with real-world data.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import logging

import sys
from pathlib import Path

# Add project root to path for standalone execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from src.cytoverse.ivfpq.pq import ProductQuantizer

# Configure logging for pytest
logger = logging.getLogger(__name__)


class TestRealEmbeddings:
    """Test cases using real embedding data."""

    @pytest.fixture
    def embeddings_data(self):
        """Load real embeddings data for testing."""
        embeddings_path = Path("data/scimilarity/embeddings.npy")

        if not embeddings_path.exists():
            pytest.skip(
                f"Embeddings file not found: {embeddings_path}. Run 'make scimilarity-export' first."
            )

        embeddings = np.load(embeddings_path)

        # Use a subset for testing to keep tests fast
        max_vectors = 1000  # Smaller subset for pytest
        if embeddings.shape[0] > max_vectors:
            indices = np.random.choice(embeddings.shape[0], max_vectors, replace=False)
            embeddings = embeddings[indices]

        return embeddings.astype(np.float32)

    def test_pq_on_real_embeddings(self, embeddings_data):
        """Test PQ training and inference on real embeddings."""
        embeddings_tensor = torch.from_numpy(embeddings_data)
        d = embeddings_data.shape[1]

        logger.info(
            f"Testing PQ on {embeddings_data.shape[0]} embeddings of dimension {d}"
        )

        # Create PQ model adapted to the actual embedding dimension
        if d == 512:
            # True SCimilarity embeddings
            pq = ProductQuantizer(
                d=512, m=32, k=64
            )  # Smaller params for faster testing
        elif d == 128:
            # Some other embedding (maybe UMAP or other model output)
            pq = ProductQuantizer(d=128, m=16, k=64)
        else:
            # Generic case
            m = min(32, d // 4)  # Ensure at least 4 dims per subquantizer
            pq = ProductQuantizer(d=d, m=m, k=64)

        # Train the model
        pq.train_pq(
            embeddings_tensor, n_iterations=10, verbose=False
        )  # Fewer iterations for speed

        assert pq.is_trained

        # Test encoding/decoding on a subset
        test_size = min(50, embeddings_data.shape[0])
        test_vectors = embeddings_tensor[:test_size]

        with torch.no_grad():
            codes = pq(test_vectors)
            reconstructed = pq.decode(codes)

        # Validate shapes
        assert codes.shape == (test_size, pq.m)
        assert reconstructed.shape == (test_size, d)

        # Validate code values are in range
        assert torch.all(codes >= 0)
        assert torch.all(codes < pq.k)

        # Compute reconstruction error
        mse = torch.mean((test_vectors - reconstructed) ** 2).item()
        rmse = np.sqrt(mse)
        original_norm = torch.norm(test_vectors, dim=1).mean().item()
        relative_error = rmse / original_norm

        logger.info(
            f"Reconstruction error: {relative_error:.4f} ({relative_error*100:.2f}%)"
        )

        # Assert reasonable reconstruction quality
        assert relative_error < 0.3, f"Relative error too high: {relative_error:.4f}"

        # Compute compression ratio
        original_bits = d * 32
        compressed_bits = pq.m * np.log2(pq.k)
        compression_ratio = original_bits / compressed_bits

        logger.info(f"Compression ratio: {compression_ratio:.1f}x")

        # Assert meaningful compression
        assert (
            compression_ratio > 5.0
        ), f"Compression ratio too low: {compression_ratio:.1f}x"

    def test_onnx_export_real_embeddings(self, embeddings_data):
        """Test ONNX export functionality with real embeddings."""
        embeddings_tensor = torch.from_numpy(embeddings_data)
        d = embeddings_data.shape[1]

        # Create and train PQ model
        if d == 128:
            pq = ProductQuantizer(d=128, m=8, k=32)  # Small config for speed
        else:
            m = min(16, d // 4)
            pq = ProductQuantizer(d=d, m=m, k=32)

        # Quick training
        pq.train_pq(embeddings_tensor[:500], n_iterations=5, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "real_embeddings_pq.onnx"

            # Export to ONNX with warning suppression
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                pq.export_onnx(onnx_path, batch_size=-1)

            assert onnx_path.exists()

            # Test ONNX model consistency
            import onnxruntime as ort

            test_input = embeddings_data[:3]

            # Run PyTorch model
            with torch.no_grad():
                torch_codes = pq(torch.from_numpy(test_input))

            # Run ONNX model
            ort_session = ort.InferenceSession(str(onnx_path))
            onnx_codes = ort_session.run(None, {"input": test_input})[0]

            # Validate consistency
            assert np.array_equal(
                torch_codes.numpy(), onnx_codes
            ), "ONNX and PyTorch outputs should match"

    def test_save_load_real_embeddings(self, embeddings_data):
        """Test save/load functionality with real embeddings."""
        embeddings_tensor = torch.from_numpy(embeddings_data)
        d = embeddings_data.shape[1]

        # Create and train PQ model
        if d == 128:
            pq = ProductQuantizer(d=128, m=8, k=16)  # Small config for speed
        else:
            m = min(16, d // 4)
            pq = ProductQuantizer(d=d, m=m, k=16)

        # Quick training
        pq.train_pq(embeddings_tensor[:300], n_iterations=5, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "real_embeddings_pq.pkl"

            # Save model
            pq.save(model_path)
            assert model_path.exists()

            # Load model
            pq_loaded = ProductQuantizer.load(model_path)

            # Test that loaded model works identically
            test_vectors = embeddings_tensor[:5]

            with torch.no_grad():
                codes_original = pq(test_vectors)
                codes_loaded = pq_loaded(test_vectors)

                reconstructed_original = pq.decode(codes_original)
                reconstructed_loaded = pq_loaded.decode(codes_loaded)

            # Validate identical behavior
            assert torch.equal(codes_original, codes_loaded)
            assert torch.allclose(
                reconstructed_original, reconstructed_loaded, atol=1e-6
            )

    def test_different_vector_dimensions(self):
        """Test PQ with different vector dimensions (synthetic test)."""
        # Test various dimensions that might appear in real data
        test_dimensions = [64, 128, 256, 512]

        for d in test_dimensions:
            # Create synthetic data
            torch.manual_seed(42)
            n_vectors = 100
            vectors = torch.randn(n_vectors, d)

            # Choose appropriate m for this dimension
            if d == 512:
                pq = ProductQuantizer(d=512, m=16, k=16)  # Small for speed
            else:
                m = min(16, d // 4)
                pq = ProductQuantizer(d=d, m=m, k=16)

            # Train quickly
            pq.train_pq(vectors, n_iterations=3, verbose=False)

            # Test encode/decode
            with torch.no_grad():
                codes = pq(vectors)
                reconstructed = pq.decode(codes)

            # Basic validation
            assert codes.shape == (n_vectors, pq.m)
            assert reconstructed.shape == (n_vectors, d)
            assert torch.all(codes >= 0)
            assert torch.all(codes < pq.k)

            # Check reasonable reconstruction
            mse = torch.mean((vectors - reconstructed) ** 2).item()
            assert mse < 10.0, f"MSE too high for dimension {d}: {mse}"

    def test_large_scale_real_embeddings(self, embeddings_data):
        """Test PQ on larger scale (marked as slow test)."""
        # This test uses more data and iterations for thorough validation
        embeddings_tensor = torch.from_numpy(embeddings_data)
        d = embeddings_data.shape[1]

        # Use more vectors if available
        max_vectors = min(5000, embeddings_data.shape[0])
        test_vectors = embeddings_tensor[:max_vectors]

        # Create PQ model
        if d == 128:
            pq = ProductQuantizer(d=128, m=16, k=64)
        else:
            m = min(32, d // 4)
            pq = ProductQuantizer(d=d, m=m, k=64)

        # More thorough training
        pq.train_pq(test_vectors, n_iterations=20, verbose=False)

        # Test on all vectors
        with torch.no_grad():
            codes = pq(test_vectors)
            reconstructed = pq.decode(codes)

        # Compute detailed metrics
        mse = torch.mean((test_vectors - reconstructed) ** 2).item()
        rmse = np.sqrt(mse)
        original_norm = torch.norm(test_vectors, dim=1).mean().item()
        relative_error = rmse / original_norm

        logger.info(f"Large scale test - Relative error: {relative_error:.4f}")
        logger.info(f"Large scale test - Vectors processed: {max_vectors}")

        # Stricter quality requirements for large scale
        assert (
            relative_error < 0.15
        ), f"Large scale relative error too high: {relative_error:.4f}"


# Standalone test function that can be called directly
def test_real_embeddings_standalone():
    """Standalone test function for direct execution (pytest compatible)."""
    embeddings_path = Path("data/scimilarity/embeddings.npy")

    if not embeddings_path.exists():
        pytest.skip(f"Embeddings file not found: {embeddings_path}")

    # Load and test embeddings
    embeddings = np.load(embeddings_path).astype(np.float32)

    # Create test instance and run basic test
    test_instance = TestRealEmbeddings()
    test_instance.test_pq_on_real_embeddings(embeddings[:1000])  # Test with subset


if __name__ == "__main__":
    # Allow script to be run directly for quick testing
    embeddings_path = Path("data/scimilarity/embeddings.npy")

    if not embeddings_path.exists():
        print(f"❌ Embeddings file not found: {embeddings_path}")
        print("Please run 'make scimilarity-export' first to create embeddings")
        exit(1)

    try:
        # Load and test embeddings
        embeddings = np.load(embeddings_path).astype(np.float32)
        print(
            f"✅ Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}"
        )

        # Create test instance and run basic test
        test_instance = TestRealEmbeddings()
        test_instance.test_pq_on_real_embeddings(embeddings[:1000])  # Test with subset

        print("✅ Real embeddings test passed!")
        exit(0)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)
