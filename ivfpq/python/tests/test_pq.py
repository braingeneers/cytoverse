#!/usr/bin/env python3

"""
Test suite for Product Quantization (PQ) implementation.

This module tests the core functionality of the PQ implementation including:
- Training with scikit-learn k-means
- Encoding and decoding
- ONNX model export for browser-side inference
- Optimized distance computation
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
import onnxruntime as ort

from ivfpq.pq import (
    PQ,
    PQDistance,
    create_pq_system,
)


class TestPQBasic:
    """Test basic PQ functionality with synthetic data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.d = 128
        self.m = 16
        self.k = 256
        self.n_vectors = 1000

        # Create synthetic vectors
        torch.manual_seed(42)
        np.random.seed(42)

        # Create regular vectors with some structure
        self.vectors = torch.randn(self.n_vectors, self.d)

        # Add some structure to make quantization meaningful
        for i in range(self.m):
            start_idx = i * (self.d // self.m)
            end_idx = (i + 1) * (self.d // self.m)
            # Add some consistent patterns within subspaces
            pattern = torch.randn(self.d // self.m) * 0.5
            self.vectors[:, start_idx:end_idx] += pattern.unsqueeze(0)

    def test_product_quantizer_initialization(self):
        """Test PQ initialization with vectors."""
        pq = PQ(self.d, self.m, self.k)

        assert pq.d == self.d
        assert pq.m == self.m
        assert pq.k == self.k
        assert pq.d_sub == self.d // self.m
        assert not pq.is_trained
        assert pq.codebooks.shape == (self.m, self.k, pq.d_sub)

    def test_pq_training_with_sklearn(self):
        """Test PQ training using scikit-learn k-means."""
        pq = PQ(self.d, self.m, self.k)

        # Train on vectors
        pq.train_pq(self.vectors, n_iterations=20)

        assert pq.is_trained
        assert pq.codebooks.shape == (self.m, self.k, pq.d_sub)

        # Check that codebooks are not all zeros or identical
        for i in range(self.m):
            codebook = pq.codebooks[i]
            # Ensure not all centroids are identical
            centroid_means = codebook.mean(dim=1)
            assert not torch.allclose(
                centroid_means, centroid_means[0].expand_as(centroid_means), atol=1e-3
            )

    def test_pq_encode_decode(self):
        """Test PQ encoding and decoding."""
        pq = PQ(self.d, self.m, self.k)
        pq.train_pq(self.vectors, n_iterations=10)

        # Test encoding
        test_vectors = self.vectors[:100]
        codes = pq(test_vectors)

        assert codes.shape == (100, self.m)
        assert codes.dtype == torch.uint8
        assert torch.all(codes >= 0)
        assert torch.all(codes <= (self.k - 1))

        # Test decoding
        decoded_vectors = pq.decode(codes)

        assert decoded_vectors.shape == test_vectors.shape
        assert decoded_vectors.dtype == torch.float32

        # Check reconstruction quality
        mse = torch.mean((test_vectors - decoded_vectors) ** 2)
        relative_error = torch.norm(test_vectors - decoded_vectors) / torch.norm(
            test_vectors
        )

        # Vectors should reconstruct reasonably well
        assert mse < 10.0  # Reasonable MSE
        assert relative_error < 0.8  # Less than 80% relative error

    def test_product_quantizer_save_load(self, tmp_path):
        """Test the PQ save/load functionality with vectors."""
        output_dir = tmp_path

        pq = PQ(self.d, self.m, self.k)
        pq.train_pq(self.vectors, n_iterations=10)

        # Save the quantizer
        pq.save(output_dir)

        # Check files were created
        assert (output_dir / "pq_metadata.json").exists()
        assert (output_dir / "pq_codebooks.bin").exists()

        # Load and verify
        pq_loaded = PQ.load(output_dir)

        assert pq_loaded.d == self.d
        assert pq_loaded.m == self.m
        assert pq_loaded.k == self.k
        assert pq_loaded.is_trained
        assert torch.allclose(pq_loaded.codebooks, pq.codebooks)

    def test_export_pq_models_via_create_system(self, tmp_path):
        """Test ONNX model export via create_pq_system."""
        output_dir = tmp_path

        # Create complete PQ system with ONNX export
        result = create_pq_system(
            training_vectors=self.vectors,
            m=self.m,
            k=self.k,
            max_iterations=5,
            output_dir=output_dir,
        )

        # Check ONNX files were created
        assert (output_dir / "pq_encode.onnx").exists()
        assert (output_dir / "pq_distance.onnx").exists()
        # Also check binary files are created
        assert (output_dir / "pq_metadata.json").exists()
        assert (output_dir / "pq_codebooks.bin").exists()

    def test_create_pq_system(self, tmp_path):
        """Test complete PQ system creation for vectors."""
        output_dir = tmp_path

        result = create_pq_system(
            training_vectors=self.vectors,
            m=self.m,
            k=self.k,
            max_iterations=5,
            output_dir=output_dir,
        )

        # Check return value
        assert "codebooks" in result
        assert "metadata" in result

        # Check all files were created
        expected_files = [
            "pq_metadata.json",
            "pq_codebooks.bin",
            "pq_encode.onnx",
            "pq_distance.onnx",
        ]

        for filename in expected_files:
            assert (output_dir / filename).exists(), f"Missing file: {filename}"


class TestPQONNXModels:
    """Test ONNX model functionality for vectors."""

    def setup_method(self):
        """Set up test fixtures."""
        self.d = 64  # Smaller for faster testing
        self.m = 8
        self.k = 64
        self.n_vectors = 200

        torch.manual_seed(42)
        self.vectors = torch.randn(self.n_vectors, self.d)

    def test_pq_encode_function(self, tmp_path):
        """Test PQ forward (encode) with vectors."""
        # Train quantizer
        pq = PQ(self.d, self.m, self.k)
        pq.train_pq(self.vectors, n_iterations=5)

        test_vectors = self.vectors[:10]

        with torch.no_grad():
            codes = pq(test_vectors)

        assert codes.shape == (10, self.m)
        assert codes.dtype == torch.uint8
        assert torch.all(codes >= 0)
        assert torch.all(codes <= (self.k - 1))

    def test_pq_decode_function(self, tmp_path):
        """Test PQ decode with vectors."""
        # Train quantizer
        pq = PQ(self.d, self.m, self.k)
        pq.train_pq(self.vectors, n_iterations=5)

        # Create test codes
        test_codes = torch.randint(0, self.k, (10, self.m))

        with torch.no_grad():
            decoded = pq.decode(test_codes)

        assert decoded.shape == (10, self.d)
        assert decoded.dtype == torch.float32

    def test_pq_distance_model(self, tmp_path):
        """Test PQDistance ONNX model with vectors."""
        # Train quantizer
        pq = PQ(self.d, self.m, self.k)
        pq.train_pq(self.vectors, n_iterations=5)
        codebooks = pq.codebooks.data

        # Create test data
        query_vector = torch.randn(self.d)
        reference_codes = torch.randint(0, self.k, (100, self.m))

        # Test PQDistance model
        distance_model = PQDistance()
        distance_model.eval()

        with torch.no_grad():
            indices, distances = distance_model(
                query_vector, reference_codes, codebooks
            )

        assert indices.shape[0] == 100  # Returns all distances
        assert distances.shape == indices.shape
        assert indices.dtype == torch.int32
        assert distances.dtype == torch.float32
        assert torch.all(indices >= 0)
        assert torch.all(indices < 100)

        # Check that distances are sorted (ascending)
        assert torch.all(distances[:-1] <= distances[1:])


class TestPQIntegration:
    """Integration tests for PQ with vectors."""

    def test_different_dimensions(self):
        """Test PQ with different vector dimensions."""
        for d in [32, 64, 128]:
            m = 8 if d >= 64 else 4
            k = 64
            n_vectors = 100

            torch.manual_seed(42)
            vectors = torch.randn(n_vectors, d)

            pq = PQ(d, m, k)
            pq.train_pq(vectors, n_iterations=5)

            assert pq.is_trained
            assert pq.d == d
            assert pq.m == m
            assert pq.d_sub == d // m

            # Test encode/decode
            codes = pq(vectors[:10])
            decoded = pq.decode(codes)

            assert codes.shape == (10, m)
            assert decoded.shape == (10, d)

    def test_different_subquantizer_counts(self):
        """Test PQ with different numbers of subquantizers."""
        d = 64
        k = 32
        n_vectors = 100

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        for m in [4, 8, 16]:
            if d % m != 0:
                continue

            pq = PQ(d, m, k)
            pq.train_pq(vectors, n_iterations=5)

            assert pq.is_trained
            assert pq.m == m
            assert pq.d_sub == d // m

            codes = pq(vectors[:10])
            assert codes.shape == (10, m)

    def test_reconstruction_quality(self):
        """Test reconstruction quality."""
        d = 128
        m = 16
        k = 256
        n_vectors = 500

        torch.manual_seed(42)
        # Create vectors
        vectors = torch.randn(n_vectors, d)

        pq = PQ(d, m, k)
        pq.train_pq(vectors, n_iterations=15)

        # Test reconstruction on unseen vectors
        test_vectors = torch.randn(50, d)
        codes = pq(test_vectors)
        reconstructed = pq.decode(codes)

        # Compute reconstruction metrics
        mse = torch.mean((test_vectors - reconstructed) ** 2)
        relative_error = torch.norm(test_vectors - reconstructed) / torch.norm(
            test_vectors
        )

        # Vectors should reconstruct reasonably well with sufficient quantization
        assert mse < 5.0, f"MSE too high: {mse}"
        assert relative_error < 0.8, f"Relative error too high: {relative_error}"

        # Check compression ratio
        original_bits = d * 32  # 32-bit floats
        compressed_bits = m * 8  # 8-bit codes
        compression_ratio = original_bits / compressed_bits

        assert compression_ratio >= 4.0  # Should achieve good compression

    def test_distance_computation_accuracy(self):
        """Test accuracy of distance computation with vectors."""
        d = 64
        m = 8
        k = 128
        n_vectors = 200

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        pq = PQ(d, m, k)
        pq.train_pq(vectors, n_iterations=10)

        # Encode all vectors
        all_codes = pq(vectors)

        # Test distance computation
        query_vector = vectors[0]
        reference_codes = all_codes[1:21]  # Use codes 1-20 as references

        distance_model = PQDistance()
        distance_model.eval()

        with torch.no_grad():
            indices, distances = distance_model(
                query_vector, reference_codes, pq.codebooks.data
            )

        # Verify results - now returns all distances, take top 5 for validation
        assert len(indices) == 20  # Returns all 20 reference codes
        assert len(distances) == 20

        # We can take top 5 if needed
        top5_indices = indices[:5]
        top5_distances = distances[:5]

        # Check that distances are reasonable
        assert torch.all(distances < 200.0), f"Distances too large: {distances}"

        # Verify indices are valid
        assert torch.all(indices >= 0)
        assert torch.all(indices < 20)

        # Check that distances are sorted
        assert torch.all(distances[:-1] <= distances[1:])


class TestPQCompatibility:
    """Test compatibility and edge cases for PQ."""

    def test_save_load_compatibility(self, tmp_path):
        """Test save/load compatibility with PQ."""
        d = 64
        m = 8
        k = 64
        n_vectors = 100

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        pq = PQ(d, m, k)
        pq.train_pq(vectors, n_iterations=5)

        save_path = tmp_path / "pq"

        # Save
        pq.save(save_path)
        assert save_path.exists()
        assert (save_path / "pq_metadata.json").exists()
        assert (save_path / "pq_codebooks.bin").exists()

        # Load
        pq_loaded = PQ.load(save_path)

        # Check loaded quantizer matches original
        assert pq_loaded.d == pq.d
        assert pq_loaded.m == pq.m
        assert pq_loaded.k == pq.k
        assert pq_loaded.is_trained == pq.is_trained
        assert torch.allclose(pq_loaded.codebooks, pq.codebooks)

        # Test functionality
        test_vectors = vectors[:10]
        codes_orig = pq(test_vectors)
        codes_loaded = pq_loaded(test_vectors)

        assert torch.equal(codes_orig, codes_loaded)

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        d = 64
        m = 8
        k = 64

        # Test invalid dimension combinations
        with pytest.raises(ValueError):
            PQ(63, 8, k)  # d not divisible by m

        pq = PQ(d, m, k)

        # Test operations on untrained quantizer
        test_vectors = torch.randn(10, d)
        test_codes = torch.randint(0, k, (10, m))

        with pytest.raises(RuntimeError):
            pq(test_vectors)

        with pytest.raises(RuntimeError):
            pq.decode(test_codes)

        with pytest.raises(RuntimeError):
            pq.save(Path("test_dir"))

        # Train quantizer
        pq.train_pq(torch.randn(100, d), n_iterations=5)

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
        vectors = torch.randn(n_vectors, d)

        pq = PQ(d, m, k)
        pq.train_pq(vectors, n_iterations=5)

        assert pq.is_trained

        codes = pq(vectors)
        assert torch.all(codes < k)

        decoded = pq.decode(codes)
        assert decoded.shape == vectors.shape

    def test_metadata_consistency(self, tmp_path):
        """Test metadata consistency in PQ."""
        d = 128
        m = 16
        k = 256
        n_vectors = 300

        torch.manual_seed(42)
        vectors = torch.randn(n_vectors, d)

        output_dir = tmp_path

        pq = PQ(d, m, k)
        pq.train_pq(vectors, n_iterations=10)
        pq.save(output_dir)

        # Load metadata from file
        with open(output_dir / "pq_metadata.json", "r") as f:
            metadata = json.load(f)

        # Check metadata fields
        assert metadata["d"] == d
        assert metadata["m"] == m
        assert metadata["k"] == k
        assert metadata["d_sub"] == d // m
        assert metadata["version"] == "1.0"
        assert metadata["compression_ratio"] == (d * 32) / (m * 8)


class TestPQONNXVerification:
    """Test ONNX model verification for exported PQ encode and decode models."""

    def setup_method(self):
        """Set up test fixtures."""
        self.d = 128
        self.m = 16
        self.k = 256
        self.n_vectors = 500

        torch.manual_seed(42)
        self.vectors = torch.randn(self.n_vectors, self.d)

    def test_pq_encode_onnx_model(self, tmp_path):
        """Test that exported pq_encode.onnx model produces same results as PyTorch."""
        # Create and train PQ system
        create_pq_system(
            training_vectors=self.vectors,
            m=self.m,
            k=self.k,
            max_iterations=10,
            output_dir=tmp_path,
        )

        # Load PyTorch model
        pq = PQ.load(tmp_path)

        # Test vectors
        test_vectors = torch.randn(10, self.d)

        # Get PyTorch results
        with torch.no_grad():
            pytorch_codes = pq(test_vectors)

        # Load ONNX model
        onnx_path = tmp_path / "pq_encode.onnx"
        assert onnx_path.exists(), "pq_encode.onnx was not created"

        ort_session = ort.InferenceSession(str(onnx_path))

        # Run ONNX inference
        onnx_outputs = ort_session.run(
            None,
            {"vectors": test_vectors.numpy()},
        )
        onnx_codes = onnx_outputs[0]

        # Compare results
        assert onnx_codes.shape == pytorch_codes.shape
        np.testing.assert_array_equal(onnx_codes, pytorch_codes.numpy())

    def test_pq_decode_onnx_model_export(self, tmp_path):
        """Test that we can export and use a pq_decode.onnx model."""
        # Create and train PQ system
        pq = PQ(self.d, self.m, self.k)
        pq.train_pq(self.vectors, n_iterations=10)
        pq.save(tmp_path)

        # Create a decode wrapper for ONNX export
        class PQDecoder(torch.nn.Module):
            def __init__(self, pq):
                super().__init__()
                self.pq = pq

            def forward(self, codes):
                return self.pq.decode(codes)

        decoder = PQDecoder(pq)
        decoder.eval()

        # Export decode model
        dummy_codes = torch.randint(0, self.k, (10, self.m), dtype=torch.uint8)
        decode_onnx_path = tmp_path / "pq_decode.onnx"

        torch.onnx.export(
            decoder,
            dummy_codes,
            decode_onnx_path,
            input_names=["codes"],
            output_names=["decoded_vectors"],
            dynamic_axes={
                "codes": {0: "n_points"},
                "decoded_vectors": {0: "n_points"},
            },
            opset_version=13,
            do_constant_folding=True,
            export_params=True,
        )

        assert decode_onnx_path.exists(), "pq_decode.onnx was not created"

        # Test the exported model
        test_codes = torch.randint(0, self.k, (5, self.m), dtype=torch.uint8)

        # Get PyTorch results
        with torch.no_grad():
            pytorch_decoded = pq.decode(test_codes)

        # Load and run ONNX model
        ort_session = ort.InferenceSession(str(decode_onnx_path))
        onnx_outputs = ort_session.run(
            None,
            {"codes": test_codes.numpy()},
        )
        onnx_decoded = onnx_outputs[0]

        # Compare results
        assert onnx_decoded.shape == pytorch_decoded.shape
        np.testing.assert_allclose(
            onnx_decoded, pytorch_decoded.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_pq_distance_onnx_model(self, tmp_path):
        """Test that exported pq_distance.onnx model produces same results as PyTorch."""
        # Create and train PQ system
        create_pq_system(
            training_vectors=self.vectors,
            m=self.m,
            k=self.k,
            max_iterations=10,
            output_dir=tmp_path,
        )

        # Load PyTorch model
        pq = PQ.load(tmp_path)

        # Create test data
        query_vector = torch.randn(self.d)
        reference_codes = torch.randint(0, self.k, (20, self.m), dtype=torch.uint8)

        # Get PyTorch results
        distance_model = PQDistance()
        distance_model.eval()

        with torch.no_grad():
            pytorch_indices, pytorch_distances = distance_model(
                query_vector, reference_codes, pq.codebooks.data
            )

        # Load ONNX model
        onnx_path = tmp_path / "pq_distance.onnx"
        assert onnx_path.exists(), "pq_distance.onnx was not created"

        ort_session = ort.InferenceSession(str(onnx_path))

        # Run ONNX inference
        onnx_outputs = ort_session.run(
            None,
            {
                "query_vector": query_vector.numpy(),
                "reference_codes": reference_codes.numpy(),
                "codebooks": pq.codebooks.data.numpy(),
            },
        )
        onnx_indices = onnx_outputs[0]
        onnx_distances = onnx_outputs[1]

        # Compare results
        assert onnx_indices.shape == pytorch_indices.shape
        assert onnx_distances.shape == pytorch_distances.shape

        np.testing.assert_array_equal(onnx_indices, pytorch_indices.numpy())
        np.testing.assert_allclose(
            onnx_distances, pytorch_distances.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_onnx_models_with_different_batch_sizes(self, tmp_path):
        """Test ONNX models work with different batch sizes."""
        # Create and train PQ system
        create_pq_system(
            training_vectors=self.vectors,
            m=self.m,
            k=self.k,
            max_iterations=10,
            output_dir=tmp_path,
        )

        # Load ONNX encode model
        encode_session = ort.InferenceSession(str(tmp_path / "pq_encode.onnx"))

        # Test with different batch sizes
        for batch_size in [1, 5, 32, 100]:
            test_vectors = torch.randn(batch_size, self.d)

            onnx_outputs = encode_session.run(
                None,
                {"vectors": test_vectors.numpy()},
            )
            codes = onnx_outputs[0]

            assert codes.shape == (batch_size, self.m)
            assert codes.dtype == np.uint8
            assert np.all(codes >= 0)
            assert np.all(codes < self.k)

    def test_onnx_encode_decode_round_trip(self, tmp_path):
        """Test encode-decode round trip using ONNX models."""
        # Create and train PQ system
        pq = PQ(self.d, self.m, self.k)
        pq.train_pq(self.vectors, n_iterations=10)
        pq.save(tmp_path)

        # Export encode model
        torch.onnx.export(
            pq,
            torch.zeros(10, self.d),
            tmp_path / "pq_encode.onnx",
            input_names=["vectors"],
            output_names=["codes"],
            dynamic_axes={"vectors": {0: "n_points"}, "codes": {0: "n_points"}},
            opset_version=13,
            do_constant_folding=True,
            export_params=True,
        )

        # Export decode model
        class PQDecoder(torch.nn.Module):
            def __init__(self, pq):
                super().__init__()
                self.pq = pq

            def forward(self, codes):
                return self.pq.decode(codes)

        decoder = PQDecoder(pq)
        decoder.eval()

        dummy_codes = torch.randint(0, self.k, (10, self.m), dtype=torch.uint8)
        torch.onnx.export(
            decoder,
            dummy_codes,
            tmp_path / "pq_decode.onnx",
            input_names=["codes"],
            output_names=["decoded_vectors"],
            dynamic_axes={
                "codes": {0: "n_points"},
                "decoded_vectors": {0: "n_points"},
            },
            opset_version=13,
            do_constant_folding=True,
            export_params=True,
        )

        # Load ONNX models
        encode_session = ort.InferenceSession(str(tmp_path / "pq_encode.onnx"))
        decode_session = ort.InferenceSession(str(tmp_path / "pq_decode.onnx"))

        # Test round trip
        test_vectors = torch.randn(5, self.d)

        # Encode
        codes = encode_session.run(None, {"vectors": test_vectors.numpy()})[0]

        # Decode
        decoded = decode_session.run(None, {"codes": codes})[0]

        # Check reconstruction
        assert decoded.shape == test_vectors.shape

        # Compute reconstruction error
        mse = np.mean((test_vectors.numpy() - decoded) ** 2)
        assert mse < 10.0  # Reasonable reconstruction error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
