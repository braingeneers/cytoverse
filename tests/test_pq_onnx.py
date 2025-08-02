"""
Tests for Product Quantization implementation and ONNX export.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import pytest

from src.pq import (
    PQEncode,
    PQDecode,
    PQDistance,
    PQDistanceBase,
    train_pq_codebooks,
    export_pq_models,
    create_pq_system
)


@pytest.fixture
def test_data():
    """Generate test data with default parameters."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    d = 128
    m = 8
    k = 256
    n_train = 10000
    n_test = 1000
    
    # Generate training data with some structure
    training_vectors = torch.randn(n_train, d)
    test_vectors = torch.randn(n_test, d)
    
    return {
        "d": d,
        "m": m,
        "k": k,
        "training_vectors": training_vectors,
        "test_vectors": test_vectors
    }


@pytest.fixture
def trained_codebooks(test_data):
    """Train codebooks for testing."""
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    result = train_pq_codebooks(
        test_data["training_vectors"],
        m=test_data["m"],
        k=test_data["k"],
        max_iterations=10,  # Fewer iterations for testing
        output_dir=str(artifacts_dir)
    )
    yield result["codebooks"], str(artifacts_dir)


def test_pq_encode():
    """Test PQ encoding model."""
    d = 128
    m = 8
    k = 256
    d_sub = d // m
    n_points = 100
    
    model = PQEncode()
    
    # Create test data
    embeddings = torch.randn(n_points, d)
    codebooks = torch.randn(m, k, d_sub)
    
    # Test forward pass
    codes = model(embeddings, codebooks)
    
    assert codes.shape == (n_points, m)
    assert codes.dtype == torch.int64
    assert codes.min() >= 0
    assert codes.max() < k


def test_pq_decode():
    """Test PQ decoding model."""
    d = 128
    m = 8
    k = 256
    d_sub = d // m
    n_points = 100
    
    model = PQDecode()
    
    # Create test data
    codes = torch.randint(0, k, (n_points, m))
    codebooks = torch.randn(m, k, d_sub)
    
    # Test forward pass
    embeddings = model(codes, codebooks)
    
    assert embeddings.shape == (n_points, d)
    assert embeddings.dtype == codebooks.dtype
    assert torch.isfinite(embeddings).all()


def test_pq_distance():
    """Test PQ distance computation."""
    d = 128
    m = 8
    k_centroids = 256
    n_refs = 1000
    k_nn = 50
    
    model = PQDistance(k=k_nn)
    
    # Create test data
    query = torch.randn(d)
    reference_codes = torch.randint(0, k_centroids, (n_refs, m))
    codebooks = torch.randn(m, k_centroids, d // m)
    
    # Test forward pass
    indices = model(query, reference_codes, codebooks)
    
    assert indices.shape == (k_nn,)
    assert indices.dtype == torch.int64
    assert indices.min() >= 0
    assert indices.max() < n_refs
    assert len(torch.unique(indices)) == k_nn  # All indices should be unique


def test_encode_decode_consistency():
    """Test that encode->decode produces reasonable reconstructions."""
    d = 128
    m = 8
    k = 256
    d_sub = d // m
    n_points = 100
    
    # Create models
    encode_model = PQEncode()
    decode_model = PQDecode()
    
    # Create test data
    embeddings = torch.randn(n_points, d)
    codebooks = torch.randn(m, k, d_sub)
    
    # Encode then decode
    codes = encode_model(embeddings, codebooks)
    reconstructed = decode_model(codes, codebooks)
    
    assert reconstructed.shape == embeddings.shape
    
    # Check that reconstruction is reasonable (not too far from original)
    mse = torch.mean((embeddings - reconstructed) ** 2)
    relative_error = torch.sqrt(mse) / torch.std(embeddings)
    
    # With random codebooks, we expect some error but not complete chaos
    assert relative_error < 2.0  # Reasonable threshold for random codebooks


def test_train_pq_codebooks(test_data):
    """Test PQ codebook training."""
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    result = train_pq_codebooks(
        test_data["training_vectors"],
        m=test_data["m"],
        k=test_data["k"],
        max_iterations=5,  # Quick test
        output_dir=str(artifacts_dir)
    )
    
    codebooks = result["codebooks"]
    metadata = result["metadata"]
    
    # Check codebooks shape
    assert codebooks.shape == (test_data["m"], test_data["k"], test_data["d"] // test_data["m"])
    assert torch.isfinite(codebooks).all()
    
    # Check metadata
    assert metadata["d"] == test_data["d"]
    assert metadata["m"] == test_data["m"]
    assert metadata["k"] == test_data["k"]
    assert metadata["d_sub"] == test_data["d"] // test_data["m"]
    
    # Check files were created
    assert (artifacts_dir / "pq_metadata.json").exists()
    assert (artifacts_dir / "pq_codebooks.bin").exists()
    assert (artifacts_dir / "pq_codebooks.npy").exists()


def test_export_pq_models(trained_codebooks):
    """Test ONNX model export."""
    codebooks, tmpdir = trained_codebooks
    
    export_pq_models(codebooks, tmpdir)
    
    # Check that ONNX files were created
    assert (Path(tmpdir) / "pq_encode.onnx").exists()
    assert (Path(tmpdir) / "pq_decode.onnx").exists()
    assert (Path(tmpdir) / "pq_distance.onnx").exists()
    assert (Path(tmpdir) / "pq_distance_base.onnx").exists()


def test_onnx_pq_encode(trained_codebooks):
    """Test ONNX PQ encoding."""
    codebooks, tmpdir = trained_codebooks
    export_pq_models(codebooks, tmpdir)
    
    # Load ONNX model
    session = ort.InferenceSession(str(Path(tmpdir) / "pq_encode.onnx"))
    
    m, k, d_sub = codebooks.shape
    d = m * d_sub
    n_points = 50
    
    # Create test inputs
    embeddings = np.random.randn(n_points, d).astype(np.float32)
    codebooks_np = codebooks.numpy().astype(np.float32)
    
    # Run ONNX inference
    outputs = session.run(None, {
        "embeddings": embeddings,
        "codebooks": codebooks_np
    })
    
    codes = outputs[0]
    
    assert codes.shape == (n_points, m)
    assert codes.min() >= 0
    assert codes.max() < k


def test_onnx_pq_decode(trained_codebooks):
    """Test ONNX PQ decoding."""
    codebooks, tmpdir = trained_codebooks
    export_pq_models(codebooks, tmpdir)
    
    # Load ONNX model
    session = ort.InferenceSession(str(Path(tmpdir) / "pq_decode.onnx"))
    
    m, k, d_sub = codebooks.shape
    d = m * d_sub
    n_points = 50
    
    # Create test inputs
    codes = np.random.randint(0, k, (n_points, m), dtype=np.int64)
    codebooks_np = codebooks.numpy().astype(np.float32)
    
    # Run ONNX inference
    outputs = session.run(None, {
        "codes": codes,
        "codebooks": codebooks_np
    })
    
    embeddings = outputs[0]
    
    assert embeddings.shape == (n_points, d)
    assert np.isfinite(embeddings).all()


def test_onnx_pq_distance(trained_codebooks):
    """Test ONNX PQ distance computation."""
    codebooks, tmpdir = trained_codebooks
    export_pq_models(codebooks, tmpdir)
    
    # Load ONNX model
    session = ort.InferenceSession(str(Path(tmpdir) / "pq_distance.onnx"))
    
    m, k_centroids, d_sub = codebooks.shape
    d = m * d_sub
    n_refs = 1000
    k_nn = 50
    
    # Create test inputs
    query = np.random.randn(d).astype(np.float32)
    reference_codes = np.random.randint(0, k_centroids, (n_refs, m), dtype=np.int64)
    codebooks_np = codebooks.numpy().astype(np.float32)
    k = np.array(k_nn, dtype=np.int64)
    
    # Run ONNX inference
    outputs = session.run(None, {
        "query": query,
        "reference_codes": reference_codes,
        "codebooks": codebooks_np
    })
    
    indices = outputs[0]
    
    assert indices.shape == (k_nn,)
    assert indices.min() >= 0
    assert indices.max() < n_refs


def test_pytorch_onnx_equivalence(trained_codebooks):
    """Test that PyTorch and ONNX models produce identical results."""
    codebooks, tmpdir = trained_codebooks
    export_pq_models(codebooks, tmpdir)
    
    m, k, d_sub = codebooks.shape
    d = m * d_sub
    n_points = 100
    
    # Create test data
    torch.manual_seed(42)
    np.random.seed(42)
    
    embeddings_torch = torch.randn(n_points, d)
    embeddings_np = embeddings_torch.numpy().astype(np.float32)
    codebooks_np = codebooks.numpy().astype(np.float32)
    
    # Test PQ Encoding
    print("Testing PQ Encode equivalence...")
    
    # PyTorch
    encode_model = PQEncode()
    encode_model.eval()
    with torch.no_grad():
        pytorch_codes = encode_model(embeddings_torch, codebooks).numpy()
    
    # ONNX
    encode_session = ort.InferenceSession(str(Path(tmpdir) / "pq_encode.onnx"))
    onnx_codes = encode_session.run(None, {
        "embeddings": embeddings_np,
        "codebooks": codebooks_np
    })[0]
    
    assert np.array_equal(pytorch_codes, onnx_codes), "Encode: PyTorch and ONNX results differ"
    
    # Test PQ Decoding
    print("Testing PQ Decode equivalence...")
    
    # Use the codes from encoding
    codes_torch = torch.from_numpy(pytorch_codes)
    
    # PyTorch
    decode_model = PQDecode()
    decode_model.eval()
    with torch.no_grad():
        pytorch_decoded = decode_model(codes_torch, codebooks).numpy()
    
    # ONNX
    decode_session = ort.InferenceSession(str(Path(tmpdir) / "pq_decode.onnx"))
    onnx_decoded = decode_session.run(None, {
        "codes": pytorch_codes,
        "codebooks": codebooks_np
    })[0]
    
    assert np.allclose(pytorch_decoded, onnx_decoded, rtol=1e-6, atol=1e-7), \
        "Decode: PyTorch and ONNX results differ"
    
    # Test PQ Distance
    print("Testing PQ Distance equivalence...")
    
    query_torch = torch.randn(d)
    query_np = query_torch.numpy().astype(np.float32)
    
    n_refs = 500
    k_nn = 50
    reference_codes_torch = torch.randint(0, k, (n_refs, m))
    reference_codes_np = reference_codes_torch.numpy().astype(np.int64)
    
    # PyTorch
    distance_model = PQDistance(k=k_nn)
    distance_model.eval()
    with torch.no_grad():
        pytorch_indices = distance_model(
            query_torch, reference_codes_torch, codebooks
        ).numpy()
    
    # ONNX
    distance_session = ort.InferenceSession(str(Path(tmpdir) / "pq_distance.onnx"))
    onnx_indices = distance_session.run(None, {
        "query": query_np,
        "reference_codes": reference_codes_np,
        "codebooks": codebooks_np
    })[0]
    
    assert np.array_equal(pytorch_indices, onnx_indices), \
        "Distance: PyTorch and ONNX results differ"
    
    print("All equivalence tests passed!")


def test_pq_distance_base_onnx(trained_codebooks):
    """Test ONNX PQ distance base model (no top-k)."""
    codebooks, tmpdir = trained_codebooks
    export_pq_models(codebooks, tmpdir)
    
    # Load ONNX model
    session = ort.InferenceSession(str(Path(tmpdir) / "pq_distance_base.onnx"))
    
    m, k_centroids, d_sub = codebooks.shape
    d = m * d_sub
    n_refs = 500
    
    # Create test inputs
    query = np.random.randn(d).astype(np.float32)
    reference_codes = np.random.randint(0, k_centroids, (n_refs, m), dtype=np.int64)
    codebooks_np = codebooks.numpy().astype(np.float32)
    
    # Run ONNX inference
    outputs = session.run(None, {
        "query": query,
        "reference_codes": reference_codes,
        "codebooks": codebooks_np
    })
    
    distances = outputs[0]
    
    # Verify output shape and properties
    assert distances.shape == (n_refs,)
    assert np.isfinite(distances).all()
    assert (distances >= 0).all()  # Squared distances should be non-negative
    
    # Compare with PyTorch implementation
    distance_base_model = PQDistanceBase()
    distance_base_model.eval()
    
    query_torch = torch.from_numpy(query)
    reference_codes_torch = torch.from_numpy(reference_codes)
    
    with torch.no_grad():
        pytorch_distances = distance_base_model(
            query_torch, reference_codes_torch, codebooks
        ).numpy()
    
    # Check that results match
    assert np.allclose(distances, pytorch_distances, rtol=1e-5, atol=1e-6), \
        "Distance base: PyTorch and ONNX results differ"
    
    print("PQ distance base ONNX test passed!")


def test_optimized_pq_distance_onnx(trained_codebooks):
    """Test optimized ONNX PQ distance model if it exists."""
    codebooks, tmpdir = trained_codebooks
    export_pq_models(codebooks, tmpdir)
    
    optimized_path = Path(tmpdir) / "pq_distance_base_optimized.onnx"
    if not optimized_path.exists():
        pytest.skip("Optimized ONNX model not created")
    
    # Load both models
    base_session = ort.InferenceSession(str(Path(tmpdir) / "pq_distance_base.onnx"))
    opt_session = ort.InferenceSession(str(optimized_path))
    
    m, k_centroids, d_sub = codebooks.shape
    d = m * d_sub
    n_refs = 500
    
    # Create test inputs
    query = np.random.randn(d).astype(np.float32)
    reference_codes = np.random.randint(0, k_centroids, (n_refs, m), dtype=np.int64)
    codebooks_np = codebooks.numpy().astype(np.float32)
    
    inputs = {
        "query": query,
        "reference_codes": reference_codes,
        "codebooks": codebooks_np
    }
    
    # Run both models
    base_outputs = base_session.run(None, inputs)
    opt_outputs = opt_session.run(None, inputs)
    
    # Results should be identical
    assert np.allclose(base_outputs[0], opt_outputs[0], rtol=1e-6, atol=1e-7), \
        "Base and optimized models produce different results"
    
    print("Optimized PQ distance ONNX test passed!")


def test_pq_system_integration(test_data):
    """Test complete PQ system creation and usage."""
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Create complete system
    result = create_pq_system(
        test_data["training_vectors"],
        m=test_data["m"],
        k=test_data["k"],
        max_iterations=5,
        output_dir=str(artifacts_dir)
    )
    
    codebooks = result["codebooks"]
    
    # Test system with test vectors
    encode_model = PQEncode()
    decode_model = PQDecode()
    distance_model = PQDistance()
    
    # Encode test vectors
    codes = encode_model(test_data["test_vectors"], codebooks)
    
    # Decode back
    reconstructed = decode_model(codes, codebooks)
    
    # Test distance computation
    query = test_data["test_vectors"][0]
    indices = distance_model(query, codes, codebooks)
    
    # Check that query is its own nearest neighbor (index 0)
    assert indices[0] == 0, "Query should be its own nearest neighbor"
    
    # Compute reconstruction error
    mse = torch.mean((test_data["test_vectors"] - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Check that reconstruction is reasonable
    assert mse < 10.0  # With proper training, this should be much lower


def test_pq_compression_ratio():
    """Test that PQ achieves expected compression."""
    d = 128
    m = 8
    k = 256
    
    # Original storage: d * 32 bits (float32)
    original_bits = d * 32
    
    # PQ storage: m * log2(k) bits
    pq_bits = m * np.log2(k)
    
    expected_ratio = original_bits / pq_bits
    
    # Test with actual implementation
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    training_vectors = torch.randn(1000, d)
    result = train_pq_codebooks(
        training_vectors, m=m, k=k, max_iterations=5, output_dir=str(artifacts_dir)
    )
    
    actual_ratio = result["metadata"]["compression_ratio"]
    
    assert abs(actual_ratio - expected_ratio) < 0.1


def test_edge_cases():
    """Test edge cases and error conditions."""
    d = 32
    m = 4
    k = 16
    d_sub = d // m
    
    # Test with minimal data
    embeddings = torch.randn(1, d)
    codebooks = torch.randn(m, k, d_sub)
    
    encode_model = PQEncode()
    decode_model = PQDecode()
    
    codes = encode_model(embeddings, codebooks)
    reconstructed = decode_model(codes, codebooks)
    
    assert codes.shape == (1, m)
    assert reconstructed.shape == (1, d)
    
    # Test dimension mismatch
    with pytest.raises(ValueError):
        train_pq_codebooks(torch.randn(100, 33), m=4, k=16)  # 33 not divisible by 4


def test_host_topk_integration():
    """Test integration of distance computation with host-side top-k."""
    d = 128
    m = 8
    k = 256
    k_nn = 50
    n_refs = 1000
    
    # Generate test data
    torch.manual_seed(42)
    codebooks = torch.randn(m, k, d // m)
    query = torch.randn(d)
    reference_codes = torch.randint(0, k, (n_refs, m))
    
    # Compute distances using base model
    distance_base_model = PQDistanceBase()
    distance_base_model.eval()
    
    with torch.no_grad():
        distances = distance_base_model(query, reference_codes, codebooks)
    
    # Perform top-k selection in "host" code
    k_actual = min(k_nn, n_refs)
    topk_values, topk_indices = torch.topk(distances, k_actual, largest=False, sorted=True)
    
    # Compare with full PQDistance model
    distance_model = PQDistance(k=k_nn)
    distance_model.eval()
    
    with torch.no_grad():
        indices_full = distance_model(query, reference_codes, codebooks)
    
    # Results should be identical
    assert torch.equal(topk_indices, indices_full), \
        "Host-side top-k produces different results than integrated model"
    
    print("Host top-k integration test passed!")


def test_default_parameters():
    """Test with the default parameters specified in requirements."""
    d = 128
    m = 8
    k = 256
    k_nn = 50
    
    # Generate test data
    torch.manual_seed(42)
    training_vectors = torch.randn(5000, d)
    test_vectors = torch.randn(100, d)
    
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Train with default parameters
    result = create_pq_system(
        training_vectors, m=m, k=k, max_iterations=10, output_dir=str(artifacts_dir)
    )
    
    codebooks = result["codebooks"]
    
    # Test encoding
    encode_model = PQEncode()
    codes = encode_model(test_vectors, codebooks)
    
    # Test distance computation with k=50
    distance_model = PQDistance(k=k_nn)
    query = test_vectors[0]
    indices = distance_model(query, codes, codebooks)
    
    assert codes.shape == (100, m)
    assert indices.shape == (k_nn,)
    assert indices[0] == 0  # Query should be nearest to itself
    
    print(f"Default parameters test passed:")
    print(f"  Embedding dimension: {d}")
    print(f"  Subquantizers: {m}")
    print(f"  Centroids per subquantizer: {k}")
    print(f"  Top-k retrieval: {k_nn}")
    print(f"  Compression ratio: {result['metadata']['compression_ratio']:.1f}x")


if __name__ == "__main__":
    # Run a quick demo
    print("Running PQ demo with default parameters...")
    
    d = 128
    m = 8
    k = 256
    k_nn = 50
    
    torch.manual_seed(42)
    training_vectors = torch.randn(1000, d)
    
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    result = create_pq_system(
        training_vectors, m=m, k=k, max_iterations=5, output_dir=str(artifacts_dir)
    )
    
    print(f"Demo completed successfully!")
    print(f"Compression ratio: {result['metadata']['compression_ratio']:.1f}x")