"""
Unit tests for Product Quantization module.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from src.cellspace.ivfpq.pq import ProductQuantizer


class TestProductQuantizer:
    """Test cases for ProductQuantizer class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        pq = ProductQuantizer(d=128, m=16, k=256)
        assert pq.d == 128
        assert pq.m == 16
        assert pq.k == 256
        assert pq.d_sub == 8
        assert not pq.is_trained

    def test_init_invalid_params(self):
        """Test initialization with invalid parameters."""
        # d not divisible by m
        with pytest.raises(ValueError):
            ProductQuantizer(d=127, m=8, k=256)

    def test_forward_untrained(self):
        """Test forward pass on untrained model raises error."""
        pq = ProductQuantizer(d=128, m=16, k=256)
        x = torch.randn(10, 128)

        with pytest.raises(RuntimeError):
            pq(x)

    def test_decode_untrained(self):
        """Test decode on untrained model raises error."""
        pq = ProductQuantizer(d=128, m=8, k=256)
        codes = torch.randint(0, 256, (10, 8))

        with pytest.raises(RuntimeError):
            pq.decode(codes)

    def test_training_small_dataset(self):
        """Test training on a small synthetic dataset."""
        # Create synthetic data
        torch.manual_seed(42)
        n_vectors = 1000
        d = 64
        m = 8
        k = 16

        # Generate random vectors
        vectors = torch.randn(n_vectors, d)

        pq = ProductQuantizer(d=d, m=m, k=k)
        pq.train_pq(vectors, n_iterations=10)

        assert pq.is_trained
        assert pq.codebooks.shape == (m, k, d // m)

    def test_encode_decode_cycle(self):
        """Test encode-decode cycle preserves reasonable approximation."""
        torch.manual_seed(42)
        n_vectors = 100
        d = 32
        m = 4
        k = 8

        # Generate random vectors
        vectors = torch.randn(n_vectors, d)

        pq = ProductQuantizer(d=d, m=m, k=k)
        pq.train_pq(vectors, n_iterations=20)

        # Test encode-decode
        with torch.no_grad():
            codes = pq(vectors)
            reconstructed = pq.decode(codes)

        # Check shapes
        assert codes.shape == (n_vectors, m)
        assert reconstructed.shape == (n_vectors, d)

        # Check code values are in valid range
        assert torch.all(codes >= 0)
        assert torch.all(codes < k)

        # Check reconstruction is reasonable (MSE should be finite)
        mse = torch.mean((vectors - reconstructed) ** 2).item()
        assert mse > 0  # Some error expected due to quantization
        assert mse < 100  # But not too large for normalized data

    def test_reconstruction_error(self):
        """Test reconstruction error computation."""
        torch.manual_seed(42)
        vectors = torch.randn(100, 32)

        pq = ProductQuantizer(d=32, m=4, k=8)
        pq.train_pq(vectors, n_iterations=10)

        error = pq.compute_reconstruction_error(vectors)
        assert isinstance(error, float)
        assert error >= 0

    def test_save_load(self):
        """Test saving and loading trained model."""
        torch.manual_seed(42)
        vectors = torch.randn(100, 32)

        pq = ProductQuantizer(d=32, m=4, k=8)
        pq.train_pq(vectors, n_iterations=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pq.pkl"

            # Save model
            pq.save(save_path)
            assert save_path.exists()

            # Load model
            pq_loaded = ProductQuantizer.load(save_path)

            # Check parameters match
            assert pq_loaded.d == pq.d
            assert pq_loaded.m == pq.m
            assert pq_loaded.k == pq.k
            assert pq_loaded.is_trained == pq.is_trained

            # Check codebooks match
            assert torch.allclose(pq_loaded.codebooks, pq.codebooks)

            # Check encode gives same results
            with torch.no_grad():
                codes_original = pq(vectors[:10])
                codes_loaded = pq_loaded(vectors[:10])
                assert torch.equal(codes_original, codes_loaded)

                # Also check that reconstructions are identical
                reconstructed_original = pq.decode(codes_original)
                reconstructed_loaded = pq_loaded.decode(codes_loaded)
                assert torch.allclose(
                    reconstructed_original, reconstructed_loaded, atol=1e-6
                )

    def test_save_untrained_fails(self):
        """Test that saving untrained model raises error."""
        pq = ProductQuantizer(d=32, m=4, k=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pq.pkl"

            with pytest.raises(RuntimeError):
                pq.save(save_path)
