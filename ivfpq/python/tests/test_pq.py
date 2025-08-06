#!/usr/bin/env python3

"""
Test suite for Product Quantization (PQ) implementation with residual vectors.

This module tests the core functionality of the PQ implementation including:
- Training with scikit-learn k-means on residual vectors
- Encoding and decoding with residual vectors
- ONNX model export for browser-side inference
- Optimized distance computation with residual vectors
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json

from ivfpq.pq import (
    ProductQuantizer,
    PQEncode,
    PQDecode,
    PQDistanceResidual,
    train_pq_codebooks,
    export_pq_models,
    create_pq_system,
)


class TestPQResidualBasic:
    """Test basic PQ functionality with residual vectors and synthetic data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.d = 128
        self.m = 16
        self.k = 256
        self.n_vectors = 1000

        # Create synthetic residual vectors (centered around zero as residuals should be)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create residual-like vectors (zero-centered with some structure)
        self.residual_vectors = torch.randn(self.n_vectors, self.d) * 0.5
        
        # Add some structure to make quantization meaningful
        for i in range(self.m):
            start_idx = i * (self.d // self.m)
            end_idx = (i + 1) * (self.d // self.m)
            # Add some consistent patterns within subspaces
            pattern = torch.randn(self.d // self.m) * 0.3
            self.residual_vectors[:, start_idx:end_idx] += pattern.unsqueeze(0)

    def test_product_quantizer_initialization(self):
        """Test ProductQuantizer initialization with residual vectors."""
        pq = ProductQuantizer(self.d, self.m, self.k)
        
        assert pq.d == self.d
        assert pq.m == self.m
        assert pq.k == self.k
        assert pq.d_sub == self.d // self.m
        assert not pq.is_trained
        assert pq.codebooks.shape == (self.m, self.k, pq.d_sub)

    def test_pq_training_with_sklearn(self):
        """Test PQ training using scikit-learn k-means on residual vectors."""
        pq = ProductQuantizer(self.d, self.m, self.k)
        
        # Train on residual vectors
        pq.train_pq(self.residual_vectors, n_iterations=20)
        
        assert pq.is_trained
        assert pq.codebooks.shape == (self.m, self.k, pq.d_sub)
        
        # Check that codebooks are not all zeros or identical
        for i in range(self.m):
            codebook = pq.codebooks[i]
            # Ensure not all centroids are identical
            centroid_means = codebook.mean(dim=1)
            assert not torch.allclose(centroid_means, centroid_means[0].expand_as(centroid_means), atol=1e-3)

    def test_pq_encode_decode_residuals(self):
        """Test PQ encoding and decoding of residual vectors."""
        pq = ProductQuantizer(self.d, self.m, self.k)
        pq.train_pq(self.residual_vectors, n_iterations=10)
        
        # Test encoding
        test_residuals = self.residual_vectors[:100]
        codes = pq(test_residuals)
        
        assert codes.shape == (100, self.m)
        assert codes.dtype == torch.long
        assert torch.all(codes >= 0)
        assert torch.all(codes < self.k)
        
        # Test decoding
        decoded_residuals = pq.decode(codes)
        
        assert decoded_residuals.shape == test_residuals.shape
        assert decoded_residuals.dtype == torch.float32
        
        # Check reconstruction quality (should be reasonable for residuals)
        mse = torch.mean((test_residuals - decoded_residuals) ** 2)
        relative_error = torch.norm(test_residuals - decoded_residuals) / torch.norm(test_residuals)
        
        # Residuals should reconstruct reasonably well
        assert mse < 1.0  # Reasonable MSE for residual vectors
        assert relative_error < 0.5  # Less than 50% relative error

    def test_train_pq_codebooks_function(self, tmp_path):
        """Test the train_pq_codebooks function with residual vectors."""
        output_dir = tmp_path
        
        result = train_pq_codebooks(
            training_vectors=self.residual_vectors,
            m=self.m,
            k=self.k,
            max_iterations=10,
            output_dir=output_dir,
            save_binary=True
        )
        
        # Check return value
        assert "codebooks" in result
        assert "metadata" in result
        
        codebooks = result["codebooks"]
        metadata = result["metadata"]
        
        assert codebooks.shape == (self.m, self.k, self.d // self.m)
        assert metadata["d"] == self.d
        assert metadata["m"] == self.m
        assert metadata["k"] == self.k
        assert metadata["version"] == "residual-1.0"
        
        # Check files were created
        assert (output_dir / "pq_metadata.json").exists()
        assert (output_dir / "pq_codebooks.bin").exists()
        assert (output_dir / "pq_codebooks.npy").exists()

    def test_export_pq_models(self, tmp_path):
        """Test ONNX model export for residual vectors."""
        output_dir = tmp_path
        
        # Train codebooks first
        result = train_pq_codebooks(
            training_vectors=self.residual_vectors,
            m=self.m,
            k=self.k,
            max_iterations=5,
            output_dir=output_dir
        )
        
        # Export ONNX models
        export_pq_models(result["codebooks"], output_dir, k_nn=50)
        
        # Check ONNX files were created
        assert (output_dir / "pq_encode.onnx").exists()
        assert (output_dir / "pq_decode.onnx").exists()
        assert (output_dir / "pq_distance.onnx").exists()

    def test_create_pq_system(self, tmp_path):
        """Test complete PQ system creation for residual vectors."""
        output_dir = tmp_path
        
        result = create_pq_system(
            training_vectors=self.residual_vectors,
            m=self.m,
            k=self.k,
            max_iterations=5,
            output_dir=output_dir
        )
        
        # Check return value
        assert "codebooks" in result
        assert "metadata" in result
        
        # Check all files were created
        expected_files = [
            "pq_metadata.json",
            "pq_codebooks.bin",
            "pq_codebooks.npy",
            "pq_encode.onnx",
            "pq_decode.onnx",
            "pq_distance.onnx"
        ]
        
        for filename in expected_files:
            assert (output_dir / filename).exists(), f"Missing file: {filename}"


class TestPQONNXModels:
    """Test ONNX model functionality for residual vectors."""

    def setup_method(self):
        """Set up test fixtures."""
        self.d = 64  # Smaller for faster testing
        self.m = 8
        self.k = 64
        self.n_vectors = 200

        torch.manual_seed(42)
        self.residual_vectors = torch.randn(self.n_vectors, self.d) * 0.3

    def test_pq_encode_model(self, tmp_path):
        """Test PQEncode ONNX model with residual vectors."""
        # Train codebooks
        output_dir = tmp_path
        result = train_pq_codebooks(
            self.residual_vectors, self.m, self.k, 5, output_dir
        )
        codebooks = result["codebooks"]
        
        # Test PQEncode model
        encode_model = PQEncode()
        encode_model.eval()
        
        test_residuals = self.residual_vectors[:10]
        
        with torch.no_grad():
            codes = encode_model(test_residuals, codebooks)
        
        assert codes.shape == (10, self.m)
        assert codes.dtype == torch.long
        assert torch.all(codes >= 0)
        assert torch.all(codes < self.k)

    def test_pq_decode_model(self, tmp_path):
        """Test PQDecode ONNX model with residual vectors."""
        # Train codebooks
        output_dir = tmp_path
        result = train_pq_codebooks(
            self.residual_vectors, self.m, self.k, 5, output_dir
        )
        codebooks = result["codebooks"]
        
        # Create test codes
        test_codes = torch.randint(0, self.k, (10, self.m))
        
        # Test PQDecode model
        decode_model = PQDecode()
        decode_model.eval()
        
        with torch.no_grad():
            decoded = decode_model(test_codes, codebooks)
        
        assert decoded.shape == (10, self.d)
        assert decoded.dtype == torch.float32

    def test_pq_distance_residual_model(self, tmp_path):
        """Test PQDistanceResidual ONNX model with residual vectors."""
        # Train codebooks
        output_dir = tmp_path
        result = train_pq_codebooks(
            self.residual_vectors, self.m, self.k, 5, output_dir
        )
        codebooks = result["codebooks"]
        
        # Create test data
        query_residual = torch.randn(self.d) * 0.3
        reference_codes = torch.randint(0, self.k, (100, self.m))
        
        # Test PQDistanceResidual model
        distance_model = PQDistanceResidual(k=10)
        distance_model.eval()
        
        with torch.no_grad():
            indices, distances = distance_model(query_residual, reference_codes, codebooks)
        
        assert indices.shape[0] == min(10, 100)
        assert distances.shape == indices.shape
        assert indices.dtype == torch.long
        assert distances.dtype == torch.float32
        assert torch.all(indices >= 0)
        assert torch.all(indices < 100)
        
        # Check that distances are sorted (ascending)
        assert torch.all(distances[:-1] <= distances[1:])


class TestPQResidualIntegration:
    """Integration tests for PQ with residual vectors."""

    def test_different_dimensions(self):
        """Test PQ with different vector dimensions."""
        for d in [32, 64, 128]:
            m = 8 if d >= 64 else 4
            k = 64
            n_vectors = 100
            
            torch.manual_seed(42)
            residual_vectors = torch.randn(n_vectors, d) * 0.2
            
            pq = ProductQuantizer(d, m, k)
            pq.train_pq(residual_vectors, n_iterations=5)
            
            assert pq.is_trained
            assert pq.d == d
            assert pq.m == m
            assert pq.d_sub == d // m
            
            # Test encode/decode
            codes = pq(residual_vectors[:10])
            decoded = pq.decode(codes)
            
            assert codes.shape == (10, m)
            assert decoded.shape == (10, d)

    def test_different_subquantizer_counts(self):
        """Test PQ with different numbers of subquantizers."""
        d = 64
        k = 32
        n_vectors = 100
        
        torch.manual_seed(42)
        residual_vectors = torch.randn(n_vectors, d) * 0.2
        
        for m in [4, 8, 16]:
            if d % m != 0:
                continue
                
            pq = ProductQuantizer(d, m, k)
            pq.train_pq(residual_vectors, n_iterations=5)
            
            assert pq.is_trained
            assert pq.m == m
            assert pq.d_sub == d // m
            
            codes = pq(residual_vectors[:10])
            assert codes.shape == (10, m)

    def test_reconstruction_quality_residuals(self):
        """Test reconstruction quality specifically for residual vectors."""
        d = 128
        m = 16
        k = 256
        n_vectors = 500
        
        torch.manual_seed(42)
        # Create residual-like vectors (zero-centered, moderate variance)
        residual_vectors = torch.randn(n_vectors, d) * 0.4
        
        pq = ProductQuantizer(d, m, k)
        pq.train_pq(residual_vectors, n_iterations=15)
        
        # Test reconstruction on unseen residuals
        test_residuals = torch.randn(50, d) * 0.4
        codes = pq(test_residuals)
        reconstructed = pq.decode(codes)
        
        # Compute reconstruction metrics
        mse = torch.mean((test_residuals - reconstructed) ** 2)
        relative_error = torch.norm(test_residuals - reconstructed) / torch.norm(test_residuals)
        
        # Residual vectors should reconstruct well with sufficient quantization
        assert mse < 0.8, f"MSE too high: {mse}"
        assert relative_error < 0.7, f"Relative error too high: {relative_error}"
        
        # Check compression ratio
        original_bits = d * 32  # 32-bit floats
        compressed_bits = m * 8  # 8-bit codes
        compression_ratio = original_bits / compressed_bits
        
        assert compression_ratio >= 4.0  # Should achieve good compression

    def test_distance_computation_accuracy(self):
        """Test accuracy of distance computation with residual vectors."""
        d = 64
        m = 8
        k = 128
        n_vectors = 200
        
        torch.manual_seed(42)
        residual_vectors = torch.randn(n_vectors, d) * 0.3
        
        pq = ProductQuantizer(d, m, k)
        pq.train_pq(residual_vectors, n_iterations=10)
        
        # Encode all vectors
        all_codes = pq(residual_vectors)
        
        # Test distance computation
        query_residual = residual_vectors[0]
        reference_codes = all_codes[1:21]  # Use codes 1-20 as references
        
        distance_model = PQDistanceResidual(k=5)
        distance_model.eval()
        
        with torch.no_grad():
            indices, distances = distance_model(query_residual, reference_codes, pq.codebooks.data)
        
        # Verify results
        assert len(indices) == 5
        assert len(distances) == 5
        
        # Check that distances are reasonable (not too large for residuals)
        assert torch.all(distances < 15.0), f"Distances too large: {distances}"
        
        # Verify indices are valid
        assert torch.all(indices >= 0)
        assert torch.all(indices < 20)


class TestPQResidualCompatibility:
    """Test compatibility and edge cases for residual-based PQ."""

    def test_save_load_compatibility(self, tmp_path):
        """Test save/load compatibility with residual-based PQ."""
        d = 64
        m = 8
        k = 64
        n_vectors = 100
        
        torch.manual_seed(42)
        residual_vectors = torch.randn(n_vectors, d) * 0.3
        
        pq = ProductQuantizer(d, m, k)
        pq.train_pq(residual_vectors, n_iterations=5)
        
        save_path = tmp_path / "pq_residual.pkl"
        
        # Save
        pq.save(save_path)
        assert save_path.exists()
        
        # Load
        pq_loaded = ProductQuantizer.load(save_path)
        
        # Check loaded quantizer matches original
        assert pq_loaded.d == pq.d
        assert pq_loaded.m == pq.m
        assert pq_loaded.k == pq.k
        assert pq_loaded.is_trained == pq.is_trained
        assert torch.allclose(pq_loaded.codebooks, pq.codebooks)
        
        # Test functionality
        test_residuals = residual_vectors[:10]
        codes_orig = pq(test_residuals)
        codes_loaded = pq_loaded(test_residuals)
        
        assert torch.equal(codes_orig, codes_loaded)

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        d = 64
        m = 8
        k = 64
        
        # Test invalid dimension combinations
        with pytest.raises(ValueError):
            ProductQuantizer(63, 8, k)  # d not divisible by m
        
        pq = ProductQuantizer(d, m, k)
        
        # Test operations on untrained quantizer
        test_vectors = torch.randn(10, d)
        test_codes = torch.randint(0, k, (10, m))
        
        with pytest.raises(RuntimeError):
            pq(test_vectors)
        
        with pytest.raises(RuntimeError):
            pq.decode(test_codes)
        
        with pytest.raises(RuntimeError):
            pq.save(Path("test.pkl"))
        
        # Train quantizer
        pq.train_pq(torch.randn(100, d) * 0.3, n_iterations=5)
        
        # Test with wrong dimensions
        wrong_vectors = torch.randn(10, 32)  # Wrong dimension
        
        with pytest.raises(ValueError):
            pq.train_pq(wrong_vectors)

    def test_small_codebook_sizes(self):
        """Test PQ with small codebook sizes."""
        d = 32
        m = 4
        k = 8  # Small codebook
        n_vectors = 50
        
        torch.manual_seed(42)
        residual_vectors = torch.randn(n_vectors, d) * 0.2
        
        pq = ProductQuantizer(d, m, k)
        pq.train_pq(residual_vectors, n_iterations=5)
        
        assert pq.is_trained
        
        codes = pq(residual_vectors)
        assert torch.all(codes < k)
        
        decoded = pq.decode(codes)
        assert decoded.shape == residual_vectors.shape

    def test_metadata_consistency(self, tmp_path):
        """Test metadata consistency in residual-based PQ."""
        d = 128
        m = 16
        k = 256
        n_vectors = 300
        
        torch.manual_seed(42)
        residual_vectors = torch.randn(n_vectors, d) * 0.3
        
        output_dir = tmp_path
        
        result = train_pq_codebooks(
            residual_vectors, m, k, 10, output_dir
        )
        
        metadata = result["metadata"]
        
        # Check metadata fields
        assert metadata["d"] == d
        assert metadata["m"] == m
        assert metadata["k"] == k
        assert metadata["d_sub"] == d // m
        assert metadata["training_samples"] == n_vectors
        assert metadata["version"] == "residual-1.0"
        assert metadata["compression_ratio"] == (d * 32) / (m * 8)
        
        # Load metadata from file
        with open(output_dir / "pq_metadata.json", 'r') as f:
            file_metadata = json.load(f)
        
        # Should match
        for key in metadata:
            assert file_metadata[key] == metadata[key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])