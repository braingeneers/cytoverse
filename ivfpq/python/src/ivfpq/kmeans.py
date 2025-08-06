"""
K-means clustering implementation with both Python and ONNX support.

This module provides k-means clustering that can:
1. Run natively in Python for training efficiency
2. Export to ONNX for browser compatibility

Each iteration can be called separately, allowing the browser to provide feedback
and progress updates during clustering.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans


class KMeansInit(nn.Module):
    """
    Initialize k-means centroids using k-means++ algorithm.

    This is a simplified version that randomly selects k points as initial centroids.
    For ONNX compatibility, we avoid complex sampling operations.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, embeddings: torch.Tensor, k: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialize k centroids from the embeddings.

        Args:
            embeddings: Input embeddings of shape [N, D]
            k: Number of clusters as scalar tensor
            seed: Random seed as scalar tensor

        Returns:
            centroids: Initial centroids of shape [K, D]
        """
        n_points, d = embeddings.shape

        # For ONNX compatibility, use a simple deterministic initialization
        # Select first k points as initial centroids
        # This avoids dynamic indexing issues

        # Take first k embeddings (or repeat if n < k)
        if n_points >= k:
            centroids = embeddings[:k]
        else:
            # Repeat embeddings to get at least k points
            repeat_times = (k + n_points - 1) // n_points
            repeated = embeddings.repeat(repeat_times, 1)
            centroids = repeated[:k]

        # Add small deterministic perturbation based on seed
        # Create deterministic noise based on seed value
        seed_factor = seed.float() / 1000.0
        noise = torch.arange(k * d, dtype=torch.float32).reshape(k, d)
        noise = (noise % 10 - 5) * 0.001 * seed_factor

        centroids = centroids + noise.to(centroids.device)

        return centroids


class KMeansIteration(nn.Module):
    """
    Single iteration of k-means clustering for ONNX export.

    This module performs one iteration of k-means:
    1. Assigns each point to nearest centroid
    2. Updates centroids based on assignments

    The stateful nature is handled by passing centroids in and out.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, embeddings: torch.Tensor, centroids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one k-means iteration.

        Args:
            embeddings: Input embeddings of shape [N, D]
            centroids: Current centroids of shape [K, D]

        Returns:
            Tuple of:
            - updated_centroids: Updated centroids [K, D]
            - assignments: Cluster assignments for each embedding [N]
            - converged: Whether algorithm has converged (0 or 1)
        """
        n_points, d = embeddings.shape
        k, _ = centroids.shape

        # Compute distances between all points and centroids
        # Using squared Euclidean distance
        # Expand dimensions for broadcasting
        embeddings_expanded = embeddings.unsqueeze(1)  # [N, 1, D]
        centroids_expanded = centroids.unsqueeze(0)  # [1, K, D]

        # Compute squared distances
        diff = embeddings_expanded - centroids_expanded  # [N, K, D]
        distances = (diff * diff).sum(dim=2)  # [N, K]

        # Assign each point to nearest centroid
        assignments = torch.argmin(distances, dim=1)  # [N]

        # Update centroids using scatter operations for ONNX compatibility
        # Create one-hot encoding of assignments
        assignments_one_hot = torch.nn.functional.one_hot(
            assignments, num_classes=k
        ).float()  # [N, K]

        # Count points per cluster
        counts = assignments_one_hot.sum(dim=0, keepdim=True).T  # [K, 1]

        # Sum embeddings per cluster
        # assignments_one_hot.T @ embeddings gives [K, D]
        cluster_sums = torch.matmul(assignments_one_hot.T, embeddings)  # [K, D]

        # Avoid division by zero
        counts_safe = torch.maximum(counts, torch.ones_like(counts))

        # Compute new centroids
        new_centroids = cluster_sums / counts_safe  # [K, D]

        # For empty clusters, keep old centroids
        empty_mask = (counts == 0).squeeze()
        updated_centroids = torch.where(
            empty_mask.unsqueeze(1), centroids, new_centroids
        )

        # Check convergence (if centroids didn't change much)
        centroid_shift = torch.norm(updated_centroids - centroids, dim=1).max()
        converged = (centroid_shift < 1e-4).float()

        return updated_centroids, assignments, converged


def export_kmeans_models(output_dir: str | Path = "."):
    """
    Export k-means models to ONNX format.

    Args:
        output_dir: Directory to save ONNX models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Export KMeansInit
    init_model = KMeansInit()
    init_model.eval()

    # Example inputs for initialization
    dummy_embeddings = torch.randn(100, 128)
    dummy_k = torch.tensor(10, dtype=torch.int64)
    dummy_seed = torch.tensor(42, dtype=torch.int64)

    torch.onnx.export(
        init_model,
        (dummy_embeddings, dummy_k, dummy_seed),
        output_dir / "kmeans_init.onnx",
        input_names=["embeddings", "k", "seed"],
        output_names=["centroids"],
        dynamic_axes={
            "embeddings": {0: "n_points", 1: "dim"},
            "centroids": {0: "k_out", 1: "dim"},
        },
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )

    # Export KMeansIteration
    iter_model = KMeansIteration()
    iter_model.eval()

    # Example inputs for iteration
    dummy_centroids = torch.randn(10, 128)

    torch.onnx.export(
        iter_model,
        (dummy_embeddings, dummy_centroids),
        output_dir / "kmeans_iteration.onnx",
        input_names=["embeddings", "centroids"],
        output_names=["updated_centroids", "assignments", "converged"],
        dynamic_axes={
            "embeddings": {0: "n_points", 1: "dim"},
            "centroids": {0: "k", 1: "dim"},
            "updated_centroids": {0: "k", 1: "dim"},
            "assignments": {0: "n_points"},
        },
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"Exported k-means models to {output_dir}")
    print("- kmeans_init.onnx: Initialize centroids")
    print("- kmeans_iteration.onnx: Single k-means iteration")


def run_python_kmeans(
    embeddings: torch.Tensor,
    k: int,
    max_iterations: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run k-means clustering using scikit-learn implementation.

    This is more efficient and robust than the hand-rolled implementation.

    Args:
        embeddings: Input embeddings of shape [N, D]
        k: Number of clusters
        max_iterations: Maximum iterations
        seed: Random seed
        verbose: Show progress (only affects display, sklearn handles iterations internally)

    Returns:
        Tuple of:
        - centroids: Final centroids [K, D]
        - assignments: Final cluster assignments [N]
    """
    # Convert to numpy for sklearn
    embeddings_np = embeddings.detach().cpu().numpy()

    if verbose:
        print(
            f"Running scikit-learn K-means: n_samples={embeddings_np.shape[0]}, k={k}, max_iter={max_iterations}"
        )

    # Use scikit-learn KMeans with k-means++ initialization
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=max_iterations,
        random_state=seed,
        n_init=1,  # Only run once since we set the random state
        verbose=1 if verbose else 0,
    )

    # Fit the model
    assignments_np = kmeans.fit_predict(embeddings_np)
    centroids_np = kmeans.cluster_centers_

    # Convert back to torch tensors
    centroids = (
        torch.from_numpy(centroids_np).to(embeddings.device).to(embeddings.dtype)
    )
    assignments = torch.from_numpy(assignments_np).to(embeddings.device).long()

    if verbose:
        print(f"K-means completed in {kmeans.n_iter_} iterations")

        # Check cluster utilization
        unique_assignments = np.unique(assignments_np)
        utilization = len(unique_assignments)
        print(f"Cluster utilization: {utilization}/{k} clusters used")

    return centroids, assignments
