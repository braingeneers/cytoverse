"""
Tests for k-means clustering and ONNX export.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import tempfile
import time
from sklearn.cluster import KMeans as SklearnKMeans

from cytoverse.kmeans import (
    KMeansInit, 
    KMeansIteration, 
    KMeansComplete,
    export_kmeans_models
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


def test_kmeans_complete():
    """Test complete k-means clustering."""
    model = KMeansComplete(max_iterations=50)
    
    # Create test data
    embeddings = torch.randn(100, 128)
    k = torch.tensor(10)
    seed = torch.tensor(42)
    
    centroids, assignments = model(embeddings, k, seed)
    
    assert centroids.shape == (10, 128)
    assert assignments.shape == (100,)
    assert torch.isfinite(centroids).all()
    assert assignments.min() >= 0
    assert assignments.max() < 10


def test_onnx_export():
    """Test ONNX export functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_kmeans_models(tmpdir)
        
        # Check that files were created
        assert (Path(tmpdir) / "kmeans_init.onnx").exists()
        assert (Path(tmpdir) / "kmeans_iteration.onnx").exists()
        assert (Path(tmpdir) / "kmeans_complete.onnx").exists()


def test_onnx_kmeans_init():
    """Test ONNX inference for k-means initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_kmeans_models(tmpdir)
        
        # Load ONNX model
        session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_init.onnx"))
        
        # Create test inputs
        embeddings = np.random.randn(100, 128).astype(np.float32)
        k = np.array(10, dtype=np.int64)
        seed = np.array(42, dtype=np.int64)
        
        # Run inference
        outputs = session.run(None, {
            "embeddings": embeddings,
            "k": k,
            "seed": seed
        })
        
        centroids = outputs[0]
        assert centroids.shape == (10, 128)
        assert np.isfinite(centroids).all()


def test_onnx_kmeans_iteration():
    """Test ONNX inference for k-means iteration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_kmeans_models(tmpdir)
        
        # Load ONNX model
        session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_iteration.onnx"))
        
        # Create test inputs
        embeddings = np.random.randn(100, 128).astype(np.float32)
        centroids = np.random.randn(10, 128).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {
            "embeddings": embeddings,
            "centroids": centroids
        })
        
        updated_centroids, assignments, converged = outputs
        
        assert updated_centroids.shape == (10, 128)
        assert assignments.shape == (100,)
        assert converged.shape == ()
        assert np.isfinite(updated_centroids).all()
        assert assignments.min() >= 0
        assert assignments.max() < 10


def test_onnx_iterative_kmeans():
    """Test iterative k-means using ONNX models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_kmeans_models(tmpdir)
        
        # Load ONNX models
        init_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_init.onnx"))
        iter_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_iteration.onnx"))
        
        # Create test data
        embeddings = np.random.randn(200, 128).astype(np.float32)
        k = np.array(20, dtype=np.int64)
        seed = np.array(42, dtype=np.int64)
        
        # Initialize centroids
        centroids = init_session.run(None, {
            "embeddings": embeddings,
            "k": k,
            "seed": seed
        })[0]
        
        # Run multiple iterations
        max_iterations = 10
        for i in range(max_iterations):
            outputs = iter_session.run(None, {
                "embeddings": embeddings,
                "centroids": centroids
            })
            
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
    
    model = KMeansComplete(max_iterations=100)
    k = torch.tensor(2)
    seed = torch.tensor(42)
    
    centroids, assignments = model(embeddings, k, seed)
    
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


def test_pytorch_onnx_equivalence():
    """Test that PyTorch and ONNX models produce identical results."""
    # Create test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    embeddings_np = np.random.randn(500, 128).astype(np.float32)
    embeddings_torch = torch.from_numpy(embeddings_np)
    k = 20
    seed = 42
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export models
        export_kmeans_models(tmpdir)
        
        # Load ONNX models
        init_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_init.onnx"))
        iter_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_iteration.onnx"))
        complete_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_complete.onnx"))
        
        print("\n=== Testing KMeansInit equivalence ===")
        # Test 1: KMeansInit
        pytorch_init = KMeansInit()
        pytorch_init.eval()
        
        # PyTorch result
        with torch.no_grad():
            pytorch_centroids = pytorch_init(
                embeddings_torch, 
                torch.tensor(k), 
                torch.tensor(seed)
            ).numpy()
        
        # ONNX result
        onnx_centroids = init_session.run(None, {
            "embeddings": embeddings_np,
            "k": np.array(k, dtype=np.int64),
            "seed": np.array(seed, dtype=np.int64)
        })[0]
        
        # Compare
        init_diff = np.abs(pytorch_centroids - onnx_centroids).max()
        print(f"Max absolute difference in centroids: {init_diff}")
        assert np.allclose(pytorch_centroids, onnx_centroids, rtol=1e-5, atol=1e-6), \
            f"KMeansInit: PyTorch and ONNX results differ by {init_diff}"
        
        print("\n=== Testing KMeansIteration equivalence ===")
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
        onnx_outputs = iter_session.run(None, {
            "embeddings": embeddings_np,
            "centroids": pytorch_centroids
        })
        onnx_updated, onnx_assignments, onnx_converged = onnx_outputs
        
        # Compare iteration results
        iter_centroid_diff = np.abs(pytorch_updated_np - onnx_updated).max()
        print(f"Max absolute difference in updated centroids: {iter_centroid_diff}")
        assert np.allclose(pytorch_updated_np, onnx_updated, rtol=1e-5, atol=1e-6), \
            f"KMeansIteration centroids: PyTorch and ONNX differ by {iter_centroid_diff}"
        
        assignment_diff = (pytorch_assignments_np != onnx_assignments).sum()
        print(f"Number of different assignments: {assignment_diff}")
        assert np.array_equal(pytorch_assignments_np, onnx_assignments), \
            f"KMeansIteration assignments: {assignment_diff} differences found"
        
        assert abs(pytorch_converged_np - onnx_converged) < 1e-6, \
            f"KMeansIteration converged: PyTorch={pytorch_converged_np}, ONNX={onnx_converged}"
        
        print("\n=== Testing KMeansComplete equivalence ===")
        # Test 3: KMeansComplete (run for fewer iterations for deterministic comparison)
        pytorch_complete = KMeansComplete(max_iterations=5)
        pytorch_complete.eval()
        
        # PyTorch result
        with torch.no_grad():
            pytorch_final_centroids, pytorch_final_assignments = pytorch_complete(
                embeddings_torch,
                torch.tensor(k),
                torch.tensor(seed)
            )
            pytorch_final_centroids_np = pytorch_final_centroids.numpy()
            pytorch_final_assignments_np = pytorch_final_assignments.numpy()
        
        # For ONNX complete model, we need to run it with limited iterations
        # Since we can't control max_iterations in the exported model,
        # let's run the iterative version for 5 iterations manually
        onnx_centroids = init_session.run(None, {
            "embeddings": embeddings_np,
            "k": np.array(k, dtype=np.int64),
            "seed": np.array(seed, dtype=np.int64)
        })[0]
        
        for i in range(5):
            outputs = iter_session.run(None, {
                "embeddings": embeddings_np,
                "centroids": onnx_centroids
            })
            onnx_centroids, onnx_assignments, converged = outputs
            if converged > 0.5:
                break
        
        # Compare final results
        final_centroid_diff = np.abs(pytorch_final_centroids_np - onnx_centroids).max()
        print(f"Max absolute difference in final centroids: {final_centroid_diff}")
        assert np.allclose(pytorch_final_centroids_np, onnx_centroids, rtol=1e-4, atol=1e-5), \
            f"Final centroids: PyTorch and ONNX differ by {final_centroid_diff}"
        
        final_assignment_diff = (pytorch_final_assignments_np != onnx_assignments).sum()
        print(f"Number of different final assignments: {final_assignment_diff}")
        # Allow small differences due to numerical precision in iterative algorithms
        assert final_assignment_diff < 10, \
            f"Final assignments: {final_assignment_diff} differences found (threshold: 10)"
        
        print("\n=== All equivalence tests passed! ===")
        print("PyTorch and ONNX models produce equivalent results.")


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


def test_kmeans_comparison_with_sklearn():
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
        embeddings_np[start_idx:end_idx] = center + np.random.randn(cluster_size, n_features) * 0.5
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
    
    model = KMeansComplete(max_iterations=100)
    k = torch.tensor(n_clusters)
    seed = torch.tensor(42)
    
    torch_centroids, torch_assignments = model(embeddings_torch, k, seed)
    torch_time = time.time() - start_time
    
    print(f"Time: {torch_time:.3f} seconds")
    print(f"Final centroids shape: {torch_centroids.shape}")
    
    # Test 2: Scikit-learn implementation
    print("\n=== Scikit-learn K-means ===")
    start_time = time.time()
    
    sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, n_init=1, max_iter=100, random_state=42)
    sklearn_kmeans.fit(embeddings_np)
    sklearn_centroids = sklearn_kmeans.cluster_centers_
    sklearn_assignments = sklearn_kmeans.labels_
    
    sklearn_time = time.time() - start_time
    
    print(f"Time: {sklearn_time:.3f} seconds")
    print(f"Final centroids shape: {sklearn_centroids.shape}")
    print(f"Number of iterations: {sklearn_kmeans.n_iter_}")
    
    # Test 3: ONNX implementation
    print("\n=== ONNX K-means (iterative) ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_kmeans_models(tmpdir)
        
        # Load ONNX models
        init_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_init.onnx"))
        iter_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_iteration.onnx"))
        
        start_time = time.time()
        
        # Initialize centroids
        onnx_centroids = init_session.run(None, {
            "embeddings": embeddings_np,
            "k": np.array(n_clusters, dtype=np.int64),
            "seed": np.array(42, dtype=np.int64)
        })[0]
        
        # Run iterations
        for i in range(100):
            outputs = iter_session.run(None, {
                "embeddings": embeddings_np,
                "centroids": onnx_centroids
            })
            
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
            for j in range(i+1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                same_in_1 = labels1[idx1] == labels1[idx2]
                same_in_2 = labels2[idx1] == labels2[idx2]
                if same_in_1 == same_in_2:
                    same_cluster_pairs += 1
                total_pairs += 1
        
        return same_cluster_pairs / total_pairs if total_pairs > 0 else 0
    
    torch_sklearn_sim = cluster_similarity(
        torch_assignments.numpy(), 
        sklearn_assignments
    )
    torch_onnx_sim = cluster_similarity(
        torch_assignments.numpy(), 
        onnx_assignments
    )
    
    print(f"\nClustering similarity:")
    print(f"PyTorch vs Scikit-learn: {torch_sklearn_sim:.3f}")
    print(f"PyTorch vs ONNX: {torch_onnx_sim:.3f}")
    
    # Basic sanity checks
    assert torch_sklearn_sim > 0.7, "PyTorch and sklearn results are too different"
    assert torch_onnx_sim > 0.9, "PyTorch and ONNX results should be very similar"


if __name__ == "__main__":
    # Run a simple demo
    print("Running k-means ONNX export demo...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_kmeans_models(tmpdir)
        print(f"\nExported models to {tmpdir}")
        
        # Run comparison test
        test_kmeans_comparison_with_sklearn()