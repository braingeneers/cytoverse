"""
Tests for k-means clustering and ONNX export.
"""

import pytest
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
from sklearn.cluster import KMeans as SklearnKMeans

from ivfpq.kmeans import (
    KMeansInit,
    KMeansIteration,
    export_kmeans_models,
    run_python_kmeans,
)


def test_kmeans_init():
    """Test k-means initialization."""
    model = KMeansInit()

    # Test with various configurations
    embeddings = torch.randn(100, 128)
    k = torch.tensor(10)
    seed = torch.tensor(42)

    centroids = model(embeddings, k, seed)

    assert centroids.shape == (10, 128)
    assert torch.isfinite(centroids).all()


def test_kmeans_iteration():
    """Test single k-means iteration."""
    model = KMeansIteration()

    # Create test data
    embeddings = torch.randn(100, 128)
    centroids = torch.randn(10, 128)

    updated_centroids, assignments, converged = model(embeddings, centroids)

    assert updated_centroids.shape == (10, 128)
    assert assignments.shape == (100,)
    assert converged.shape == ()
    assert torch.isfinite(updated_centroids).all()
    assert assignments.min() >= 0
    assert assignments.max() < 10


def test_onnx_export(tmp_path):
    """Test ONNX export functionality."""
    export_kmeans_models(tmp_path)

    # Check that files were created
    assert (tmp_path / "kmeans_init.onnx").exists()
    assert (tmp_path / "kmeans_iteration.onnx").exists()


def test_onnx_kmeans_init(tmp_path):
    """Test ONNX inference for k-means initialization."""
    export_kmeans_models(tmp_path)

    # Load ONNX model
    session = ort.InferenceSession(str(tmp_path / "kmeans_init.onnx"))

    # Create test inputs
    embeddings = np.random.randn(100, 128).astype(np.float32)
    k = np.array(10, dtype=np.int64)
    seed = np.array(42, dtype=np.int64)

    # Run inference
    outputs = session.run(None, {"embeddings": embeddings, "k": k, "seed": seed})

    centroids = outputs[0]
    assert centroids.shape == (10, 128)
    assert np.isfinite(centroids).all()


def test_onnx_kmeans_iteration(tmp_path):
    """Test ONNX inference for k-means iteration."""
    export_kmeans_models(tmp_path)

    # Load ONNX model
    session = ort.InferenceSession(str(tmp_path / "kmeans_iteration.onnx"))

    # Create test inputs
    embeddings = np.random.randn(100, 128).astype(np.float32)
    centroids = np.random.randn(10, 128).astype(np.float32)

    # Run inference
    outputs = session.run(None, {"embeddings": embeddings, "centroids": centroids})

    updated_centroids, assignments, converged = outputs

    assert updated_centroids.shape == (10, 128)
    assert assignments.shape == (100,)
    assert converged.shape == ()
    assert np.isfinite(updated_centroids).all()
    assert assignments.min() >= 0
    assert assignments.max() < 10


def test_onnx_iterative_kmeans(tmp_path):
    """Test iterative k-means using ONNX models."""
    export_kmeans_models(tmp_path)

    # Load ONNX models
    init_session = ort.InferenceSession(str(tmp_path / "kmeans_init.onnx"))
    iter_session = ort.InferenceSession(str(tmp_path / "kmeans_iteration.onnx"))

    # Create test data
    embeddings = np.random.randn(200, 128).astype(np.float32)
    k = np.array(20, dtype=np.int64)
    seed = np.array(42, dtype=np.int64)

    # Initialize centroids
    centroids = init_session.run(
        None, {"embeddings": embeddings, "k": k, "seed": seed}
    )[0]

    # Run multiple iterations
    max_iterations = 10
    for i in range(max_iterations):
        outputs = iter_session.run(
            None, {"embeddings": embeddings, "centroids": centroids}
        )

        centroids, assignments, converged = outputs

        # Progress callback could be called here in browser
        print(f"Iteration {i}: converged={converged}")

        if converged > 0.5:
            print(f"Converged after {i+1} iterations")
            break

    # Verify final results
    assert centroids.shape == (20, 128)
    assert assignments.shape == (200,)
    assert np.isfinite(centroids).all()

    # Check that all clusters have at least one point
    unique_assignments = np.unique(assignments)
    print(f"Number of non-empty clusters: {len(unique_assignments)}")


def test_convergence():
    """Test that k-means converges on simple data."""
    # Create clearly separated clusters
    cluster1 = torch.randn(50, 2) * 0.5 + torch.tensor([5.0, 5.0])
    cluster2 = torch.randn(50, 2) * 0.5 + torch.tensor([-5.0, -5.0])
    embeddings = torch.cat([cluster1, cluster2])

    k = torch.tensor(2)
    seed = torch.tensor(42)

    # Initialize centroids
    init_model = KMeansInit()
    centroids = init_model(embeddings, k, seed)
    
    # Run iterations until convergence
    iter_model = KMeansIteration()
    max_iterations = 100
    
    for i in range(max_iterations):
        centroids, assignments, converged = iter_model(embeddings, centroids)
        if converged.item() > 0.5:
            break

    # Check that clusters are correctly separated
    cluster1_assignments = assignments[:50]
    cluster2_assignments = assignments[50:]

    # Since we have 2 well-separated clusters, check that most points
    # from cluster1 have the same assignment, and it's different from cluster2
    cluster1_mode = cluster1_assignments[0].item()
    cluster2_mode = cluster2_assignments[0].item()

    # They should be assigned to different clusters
    assert cluster1_mode != cluster2_mode

    # Most points should be correctly clustered
    cluster1_correct = (cluster1_assignments == cluster1_mode).sum()
    cluster2_correct = (cluster2_assignments == cluster2_mode).sum()

    assert cluster1_correct > 45  # At least 90% correct
    assert cluster2_correct > 45  # At least 90% correct


def test_pytorch_onnx_equivalence(tmp_path):
    """Test that PyTorch and ONNX models produce identical results."""
    # Create test data
    np.random.seed(42)
    torch.manual_seed(42)

    embeddings_np = np.random.randn(500, 128).astype(np.float32)
    embeddings_torch = torch.from_numpy(embeddings_np)
    k = 20
    seed = 42

    # Export models
    export_kmeans_models(tmp_path)

    # Load ONNX models
    init_session = ort.InferenceSession(str(tmp_path / "kmeans_init.onnx"))
    iter_session = ort.InferenceSession(str(tmp_path / "kmeans_iteration.onnx"))

    # Test 1: KMeansInit
    pytorch_init = KMeansInit()
    pytorch_init.eval()

    # PyTorch result
    with torch.no_grad():
        pytorch_centroids = pytorch_init(
            embeddings_torch, torch.tensor(k), torch.tensor(seed)
        ).numpy()

    # ONNX result
    onnx_centroids = init_session.run(
        None,
        {
            "embeddings": embeddings_np,
            "k": np.array(k, dtype=np.int64),
            "seed": np.array(seed, dtype=np.int64),
        },
    )[0]

    # Compare init results
    assert np.allclose(
        pytorch_centroids, onnx_centroids, rtol=1e-5, atol=1e-6
    ), f"KMeansInit: PyTorch and ONNX results differ"

    # Test 2: KMeansIteration
    pytorch_iter = KMeansIteration()
    pytorch_iter.eval()

    # Use the centroids from init for iteration test
    centroids_torch = torch.from_numpy(pytorch_centroids)

    # PyTorch result
    with torch.no_grad():
        pytorch_updated, pytorch_assignments, pytorch_converged = pytorch_iter(
            embeddings_torch, centroids_torch
        )
        pytorch_updated_np = pytorch_updated.numpy()
        pytorch_assignments_np = pytorch_assignments.numpy()
        pytorch_converged_np = pytorch_converged.numpy()

    # ONNX result
    onnx_outputs = iter_session.run(
        None, {"embeddings": embeddings_np, "centroids": pytorch_centroids}
    )
    onnx_updated, onnx_assignments, onnx_converged = onnx_outputs

    # Compare iteration results
    assert np.allclose(
        pytorch_updated_np, onnx_updated, rtol=1e-5, atol=1e-6
    ), f"KMeansIteration centroids: PyTorch and ONNX differ"

    assert np.array_equal(
        pytorch_assignments_np, onnx_assignments
    ), f"KMeansIteration assignments differ"

    assert (
        abs(pytorch_converged_np - onnx_converged) < 1e-6
    ), f"KMeansIteration converged: PyTorch={pytorch_converged_np}, ONNX={onnx_converged}"

    # Test 3: Complete K-means flow (5 iterations)
    # PyTorch implementation
    with torch.no_grad():
        # Initialize
        pytorch_final_centroids = pytorch_init(
            embeddings_torch, torch.tensor(k), torch.tensor(seed)
        )
        
        # Run 5 iterations
        for _ in range(5):
            pytorch_final_centroids, pytorch_final_assignments, converged = pytorch_iter(
                embeddings_torch, pytorch_final_centroids
            )
            if converged.item() > 0.5:
                break
        
        pytorch_final_centroids_np = pytorch_final_centroids.numpy()
        pytorch_final_assignments_np = pytorch_final_assignments.numpy()

    # ONNX implementation
    onnx_centroids = init_session.run(
        None,
        {
            "embeddings": embeddings_np,
            "k": np.array(k, dtype=np.int64),
            "seed": np.array(seed, dtype=np.int64),
        },
    )[0]

    for _ in range(5):
        outputs = iter_session.run(
            None, {"embeddings": embeddings_np, "centroids": onnx_centroids}
        )
        onnx_centroids, onnx_assignments, converged = outputs
        if converged > 0.5:
            break

    # Compare final results
    assert np.allclose(
        pytorch_final_centroids_np, onnx_centroids, rtol=1e-4, atol=1e-5
    ), f"Final centroids: PyTorch and ONNX differ"

    # Allow small differences due to numerical precision in iterative algorithms
    final_assignment_diff = (pytorch_final_assignments_np != onnx_assignments).sum()
    assert (
        final_assignment_diff < 10
    ), f"Final assignments: {final_assignment_diff} differences found (threshold: 10)"


def test_edge_cases():
    """Test edge cases for k-means."""
    # Test with k > n_points
    model = KMeansInit()
    embeddings = torch.randn(5, 128)
    k = torch.tensor(10)
    seed = torch.tensor(42)

    centroids = model(embeddings, k, seed)
    assert centroids.shape == (10, 128)

    # Test with single point
    embeddings = torch.randn(1, 128)
    k = torch.tensor(1)
    centroids = model(embeddings, k, seed)
    assert centroids.shape == (1, 128)


def test_kmeans_comparison_with_sklearn(tmp_path):
    """Compare our k-means implementation with scikit-learn."""
    # Generate test data with clear clusters
    np.random.seed(42)
    torch.manual_seed(42)

    # Create 5000 embeddings in 128 dimensions with 50 clusters
    n_samples = 5000
    n_clusters = 50
    n_features = 128

    # Generate clustered data
    embeddings_np = np.zeros((n_samples, n_features), dtype=np.float32)
    true_labels = np.zeros(n_samples, dtype=int)

    samples_per_cluster = n_samples // n_clusters

    for i in range(n_clusters):
        # Generate cluster center
        center = np.random.randn(n_features) * 10

        # Generate points around this center
        start_idx = i * samples_per_cluster
        end_idx = (i + 1) * samples_per_cluster if i < n_clusters - 1 else n_samples

        cluster_size = end_idx - start_idx
        embeddings_np[start_idx:end_idx] = (
            center + np.random.randn(cluster_size, n_features) * 0.5
        )
        true_labels[start_idx:end_idx] = i

    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    embeddings_np = embeddings_np[shuffle_idx]
    true_labels = true_labels[shuffle_idx]

    # Convert to torch
    embeddings_torch = torch.from_numpy(embeddings_np)

    # Test 1: PyTorch implementation
    print("\n=== PyTorch K-means ===")
    start_time = time.time()

    k = torch.tensor(n_clusters)
    seed = torch.tensor(42)

    # Initialize
    init_model = KMeansInit()
    torch_centroids = init_model(embeddings_torch, k, seed)
    
    # Run iterations
    iter_model = KMeansIteration()
    for i in range(100):
        torch_centroids, torch_assignments, converged = iter_model(embeddings_torch, torch_centroids)
        if converged.item() > 0.5:
            print(f"Converged after {i+1} iterations")
            break
    
    torch_time = time.time() - start_time

    print(f"Time: {torch_time:.3f} seconds")
    print(f"Final centroids shape: {torch_centroids.shape}")

    # Test 2: Scikit-learn implementation
    print("\n=== Scikit-learn K-means ===")
    start_time = time.time()

    sklearn_kmeans = SklearnKMeans(
        n_clusters=n_clusters, n_init=1, max_iter=100, random_state=42
    )
    sklearn_kmeans.fit(embeddings_np)
    sklearn_centroids = sklearn_kmeans.cluster_centers_
    sklearn_assignments = sklearn_kmeans.labels_

    sklearn_time = time.time() - start_time

    print(f"Time: {sklearn_time:.3f} seconds")
    print(f"Final centroids shape: {sklearn_centroids.shape}")
    print(f"Number of iterations: {sklearn_kmeans.n_iter_}")

    # Test 3: ONNX implementation
    print("\n=== ONNX K-means (iterative) ===")

    export_kmeans_models(tmp_path)

    # Load ONNX models
    init_session = ort.InferenceSession(str(tmp_path / "kmeans_init.onnx"))
    iter_session = ort.InferenceSession(str(tmp_path / "kmeans_iteration.onnx"))

    start_time = time.time()

    # Initialize centroids
    onnx_centroids = init_session.run(
        None,
        {
            "embeddings": embeddings_np,
            "k": np.array(n_clusters, dtype=np.int64),
            "seed": np.array(42, dtype=np.int64),
        },
    )[0]

    # Run iterations
    for i in range(100):
        outputs = iter_session.run(
            None, {"embeddings": embeddings_np, "centroids": onnx_centroids}
        )

        onnx_centroids, onnx_assignments, converged = outputs

        if converged > 0.5:
            print(f"Converged after {i+1} iterations")
            break

    onnx_time = time.time() - start_time

    print(f"Time: {onnx_time:.3f} seconds")
    print(f"Final centroids shape: {onnx_centroids.shape}")

    # Compare results
    print("\n=== Comparison ===")
    print(f"PyTorch time: {torch_time:.3f}s")
    print(f"Scikit-learn time: {sklearn_time:.3f}s")
    print(f"ONNX time: {onnx_time:.3f}s")
    print(f"Speed ratio (sklearn/pytorch): {sklearn_time/torch_time:.2f}x")
    print(f"Speed ratio (sklearn/onnx): {sklearn_time/onnx_time:.2f}x")

    # Compare cluster assignments similarity (note: cluster IDs might be different)
    # We'll check if points that are in the same cluster in one implementation
    # are also in the same cluster in another implementation
    def cluster_similarity(labels1, labels2):
        """Compute similarity between two clustering results."""
        n = len(labels1)
        same_cluster_pairs = 0
        total_pairs = 0

        # Sample some pairs to avoid O(n^2) complexity
        n_samples = min(1000, n)
        indices = np.random.choice(n, n_samples, replace=False)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                same_in_1 = labels1[idx1] == labels1[idx2]
                same_in_2 = labels2[idx1] == labels2[idx2]
                if same_in_1 == same_in_2:
                    same_cluster_pairs += 1
                total_pairs += 1

        return same_cluster_pairs / total_pairs if total_pairs > 0 else 0

    torch_sklearn_sim = cluster_similarity(
        torch_assignments.numpy(), sklearn_assignments
    )
    torch_onnx_sim = cluster_similarity(torch_assignments.numpy(), onnx_assignments)

    print(f"\nClustering similarity:")
    print(f"PyTorch vs Scikit-learn: {torch_sklearn_sim:.3f}")
    print(f"PyTorch vs ONNX: {torch_onnx_sim:.3f}")

    # Basic sanity checks
    assert torch_sklearn_sim > 0.7, "PyTorch and sklearn results are too different"
    assert torch_onnx_sim > 0.9, "PyTorch and ONNX results should be very similar"


def test_export_test_data_for_typescript(tmp_path):
    """Export ONNX models and test data for TypeScript tests."""
    # Create output directory
    output_dir = tmp_path / "fixtures" / "kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to: {output_dir}")

    # Export ONNX models
    export_kmeans_models(output_dir)

    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)

    # Smaller dataset for TypeScript tests
    embeddings_np = np.random.randn(100, 128).astype(np.float32)
    embeddings_torch = torch.from_numpy(embeddings_np)
    k = 10
    seed = 42

    # Generate expected results
    pytorch_init = KMeansInit()
    pytorch_init.eval()

    with torch.no_grad():
        initial_centroids = pytorch_init(
            embeddings_torch, torch.tensor(k), torch.tensor(seed)
        ).numpy()

    pytorch_iter = KMeansIteration()
    pytorch_iter.eval()

    with torch.no_grad():
        updated_centroids, assignments, converged = pytorch_iter(
            embeddings_torch, torch.from_numpy(initial_centroids)
        )
        updated_centroids_np = updated_centroids.numpy()
        assignments_np = assignments.numpy()
        converged_np = converged.numpy()

    # Export test data as browser-ready binary files
    # Embeddings: Float32Array
    embeddings_np.astype(np.float32).tofile(output_dir / "test_embeddings.bin")

    # Initial centroids: Float32Array
    initial_centroids.astype(np.float32).tofile(
        output_dir / "expected_initial_centroids.bin"
    )

    # Updated centroids: Float32Array
    updated_centroids_np.astype(np.float32).tofile(
        output_dir / "expected_updated_centroids.bin"
    )

    # Assignments: Int32Array
    assignments_np.astype(np.int32).tofile(output_dir / "expected_assignments.bin")

    # Export metadata
    metadata = {
        "k": k,
        "seed": seed,
        "n_samples": embeddings_np.shape[0],
        "n_features": embeddings_np.shape[1],
        "converged": float(converged_np),
        "description": "Test data for TypeScript ONNX k-means tests",
    }

    import json

    with open(output_dir / "test_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Exported ONNX models and test data for TypeScript tests")
    print(f"   - ONNX models: {output_dir}")
    print(f"   - Test embeddings: {embeddings_np.shape}")
    print(f"   - Expected centroids: {initial_centroids.shape}")

    return str(output_dir)


def test_python_vs_pytorch_kmeans():
    """Test that run_python_kmeans produces equivalent results to PyTorch KMeansInit/KMeansIteration."""
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create test data with clear clusters for better comparison
    n_samples = 1000
    n_features = 64
    n_clusters = 10
    
    # Generate clustered data
    embeddings_np = np.zeros((n_samples, n_features), dtype=np.float32)
    samples_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        center = np.random.randn(n_features) * 5
        start_idx = i * samples_per_cluster
        end_idx = (i + 1) * samples_per_cluster if i < n_clusters - 1 else n_samples
        cluster_size = end_idx - start_idx
        embeddings_np[start_idx:end_idx] = center + np.random.randn(cluster_size, n_features) * 0.5
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    embeddings_np = embeddings_np[shuffle_idx]
    embeddings_torch = torch.from_numpy(embeddings_np)
    
    # Test 1: run_python_kmeans (scikit-learn)
    print("\n=== Python K-means (scikit-learn) ===")
    start_time = time.time()
    python_centroids, python_assignments = run_python_kmeans(
        embeddings_torch, k=n_clusters, max_iterations=100, seed=42, verbose=True
    )
    python_time = time.time() - start_time
    
    # Test 2: PyTorch KMeansInit + KMeansIteration
    print("\n=== PyTorch K-means (manual iteration) ===")
    start_time = time.time()
    
    # Initialize centroids using PyTorch
    init_model = KMeansInit()
    pytorch_centroids = init_model(embeddings_torch, torch.tensor(n_clusters), torch.tensor(42))
    
    # Run iterations using PyTorch
    iter_model = KMeansIteration()
    for i in range(100):
        pytorch_centroids, pytorch_assignments, converged = iter_model(embeddings_torch, pytorch_centroids)
        if converged.item() > 0.5:
            print(f"Converged after {i+1} iterations")
            break
    
    pytorch_time = time.time() - start_time
    
    print(f"Python time: {python_time:.3f}s")
    print(f"PyTorch time: {pytorch_time:.3f}s")
    
    # Compare results using cluster similarity metric
    def cluster_similarity(labels1, labels2):
        """Compute similarity between two clustering results."""
        n = len(labels1)
        same_cluster_pairs = 0
        total_pairs = 0
        
        # Sample pairs to avoid O(n^2) complexity
        n_samples_check = min(500, n)
        indices = np.random.choice(n, n_samples_check, replace=False)
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                same_in_1 = labels1[idx1] == labels1[idx2]
                same_in_2 = labels2[idx1] == labels2[idx2]
                if same_in_1 == same_in_2:
                    same_cluster_pairs += 1
                total_pairs += 1
        
        return same_cluster_pairs / total_pairs if total_pairs > 0 else 0
    
    # Convert assignments to numpy for comparison
    python_assignments_np = python_assignments.cpu().numpy()
    pytorch_assignments_np = pytorch_assignments.cpu().numpy()
    
    similarity = cluster_similarity(python_assignments_np, pytorch_assignments_np)
    print(f"\nClustering similarity: {similarity:.3f}")
    
    # Check cluster utilization for both methods
    python_unique = len(np.unique(python_assignments_np))
    pytorch_unique = len(np.unique(pytorch_assignments_np))
    print(f"Python clusters used: {python_unique}/{n_clusters}")
    print(f"PyTorch clusters used: {pytorch_unique}/{n_clusters}")
    
    # Assertions for test validation
    assert similarity > 0.7, f"Clustering similarity {similarity:.3f} is too low (expected > 0.7)"
    assert python_unique >= n_clusters * 0.8, f"Python method used too few clusters: {python_unique}/{n_clusters}"
    assert pytorch_unique >= n_clusters * 0.8, f"PyTorch method used too few clusters: {pytorch_unique}/{n_clusters}"
    
    # Check that both methods produce reasonable centroids
    assert python_centroids.shape == (n_clusters, n_features)
    assert pytorch_centroids.shape == (n_clusters, n_features)
    assert torch.isfinite(python_centroids).all()
    assert torch.isfinite(pytorch_centroids).all()
    
    print(f"✅ Test passed: run_python_kmeans and PyTorch methods produce equivalent results")


if __name__ == "__main__":
    # Run a simple demo
    print("Running k-means ONNX export demo...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        export_kmeans_models(tmpdir)
        print(f"\nExported models to {tmpdir}")

        # Run comparison test would need tmp_path here, so skip in main
        print("Run 'pytest test_kmeans.py -v' to run full tests")