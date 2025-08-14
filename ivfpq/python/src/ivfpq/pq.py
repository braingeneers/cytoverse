"""
Product Quantization (PQ) implementation with optimized ONNX export.

This module provides PyTorch models for Product Quantization including:
- Encode: quantizes embeddings using learned codebooks
- Decode: reconstructs embeddings from PQ codes
- Distance: computes asymmetric distances for efficient kNN search

Based on "Product Quantization for Nearest Neighbor Search" by Jégou et al.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import json
import logging
from sklearn.cluster import KMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PQ(nn.Module):
    """
    Product Quantization encoder/decoder module.

    Product Quantization decomposes high-dimensional vectors into multiple lower-dimensional subvectors, each quantized independently using k-means.
    This allows for significant compression while maintaining reasonable
    approximation quality for search.

    Args:
        d: Input vector dimension
        m: Number of subspaces (subquantizers)
        k: Number of centroids per subspace (codebook size)

    The input dimension d must be divisible by m.
    Each subspace has dimension d/m and k centroids.
    """

    def __init__(self, d: int, m: int, k: int):
        super().__init__()

        if d % m != 0:
            raise ValueError(
                f"Input dimension {d} must be divisible by number of subquantizers {m}"
            )

        self.d = d  # Input dimension
        self.m = m  # Number of subquantizers
        self.k = k  # Number of centroids per subquantizer
        self.d_sub = d // m  # Dimension of each subspace

        # Codebooks: [m, k, d_sub] - m codebooks, each with k centroids of dimension d_sub
        self.codebooks = nn.Parameter(
            torch.randn(m, k, self.d_sub), requires_grad=False
        )

        logger.info(
            f"Initialized PQ: d={self.d}, m={self.m}, k={self.k}, d_sub={self.d_sub} for a compression ratio of {self.d * 32 / (self.m * 8)}x"
        )

        logger.info(
            f"Codebooks shape: {self.codebooks.shape} with dtype {self.codebooks.dtype} and storage size {self.codebooks.numel() * self.codebooks.element_size() / (1024 ** 2):.2f} MB"
        )

        # Whether the quantizer has been trained
        self.is_trained = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode vectors using product quantization.

        Args:
            x: Input vectors of shape [N, d]

        Returns:
            PQ codes of shape [N, m] with integer values in [0, k-1]
        """
        if not self.is_trained:
            raise RuntimeError("PQmust be trained before encoding")

        batch_size = x.shape[0]

        # Split input into subvectors: [N, m, d_sub]
        x_split = x.view(batch_size, self.m, self.d_sub)

        # Find closest centroids for each subvector
        codes = torch.zeros(batch_size, self.m, dtype=torch.long, device=x.device)

        for i in range(self.m):
            # Compute distances to all centroids in this subspace
            sub_vectors = x_split[:, i]  # [N, d_sub]
            centroids = self.codebooks[i]  # [k, d_sub]

            # Compute squared Euclidean distances using broadcasting (ONNX-friendly)
            sub_vectors_expanded = sub_vectors.unsqueeze(1)  # [N, 1, d_sub]
            centroids_expanded = centroids.unsqueeze(0)  # [1, k, d_sub]

            # Compute squared distances: [N, k]
            diff = sub_vectors_expanded - centroids_expanded  # [N, k, d_sub]
            distances = torch.sum(diff * diff, dim=2)  # [N, k]

            # Find nearest centroid indices
            codes[:, i] = torch.argmin(distances, dim=1)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode PQ codes back to approximate vectors.

        Args:
            codes: PQ codes of shape [N, m] with integer values in [0, k-1]

        Returns:
            Decoded vectors of shape [N, d]
        """
        if not self.is_trained:
            raise RuntimeError("PQmust be trained before decoding")

        batch_size = codes.shape[0]

        # Reconstruct vectors by looking up centroids
        decoded = torch.zeros(
            batch_size, self.d, device=codes.device, dtype=self.codebooks.dtype
        )

        for i in range(self.m):
            start_idx = i * self.d_sub
            end_idx = (i + 1) * self.d_sub

            # Look up centroids for this subspace (ensure codes are long type)
            decoded[:, start_idx:end_idx] = self.codebooks[i][codes[:, i].long()]

        return decoded

    def train_pq(
        self,
        vectors: torch.Tensor,
        n_iterations: int = 50,
    ) -> None:
        """
        Train the product quantizer using scikit-learn k-means clustering.

        Args:
            vectors: Training vectors of shape [N, d]
            n_iterations: Number of k-means iterations per subspace
        """
        if vectors.shape[1] != self.d:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match expected {self.d}"
            )

        n_vectors = vectors.shape[0]

        # Split vectors into subvectors
        vectors_split = vectors.view(n_vectors, self.m, self.d_sub)

        with torch.no_grad():
            # Progress bar for subquantizers
            subquantizer_pbar = tqdm(range(self.m), desc="Training subquantizers")

            for i in subquantizer_pbar:
                sub_vectors = vectors_split[:, i]  # [N, d_sub]

                # Adjust n_clusters if we have fewer samples than clusters
                n_samples = sub_vectors.shape[0]
                n_clusters_actual = min(self.k, n_samples)

                # Use scikit-learn k-means
                kmeans = KMeans(
                    n_clusters=n_clusters_actual,
                    max_iter=n_iterations,
                    random_state=42 + i,
                    n_init=1,  # Only run once since we set the random state
                )

                # Convert to numpy for sklearn
                sub_vectors_np = sub_vectors.cpu().numpy()
                kmeans.fit(sub_vectors_np)
                centroids_actual = torch.from_numpy(kmeans.cluster_centers_).float()

                # Store learned centroids, padding with zeros if necessary
                if n_clusters_actual < self.k:
                    centroids = torch.zeros(self.k, self.d_sub)
                    centroids[:n_clusters_actual] = centroids_actual
                else:
                    centroids = centroids_actual

                self.codebooks.data[i] = centroids

                # Check cluster utilization
                assignments = kmeans.labels_
                unique_assignments = np.unique(assignments)
                utilization = len(unique_assignments)
                subquantizer_pbar.set_postfix(clusters_used=f"{utilization}/{self.k}")

            subquantizer_pbar.close()

        self.is_trained = True

    def save(self, path: Path) -> None:
        """Save the trained quantizer to disk in binary format."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained quantizer")

        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        # Save metadata as JSON
        metadata = {
            "d": self.d,
            "m": self.m,
            "k": self.k,
            "d_sub": self.d_sub,
            "compression_ratio": float((self.d * 32) / (self.m * 8)),
            "codebooks_shape": [self.m, self.k, self.d_sub],
            "codebooks_size": int(self.m * self.k * self.d_sub),
            "version": "1.0",
        }

        with open(path / "pq_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save codebooks as binary file
        codebooks_np = self.codebooks.data.cpu().numpy().astype(np.float32)
        codebooks_np.tofile(path / "pq_codebooks.bin")

        logger.info(f"Saved binary codebooks to {path / 'pq_codebooks.bin'}")
        logger.info(f"Metadata saved to {path / 'pq_metadata.json'}")

    @classmethod
    def load(cls, path: Path) -> "PQ":
        """Load a trained quantizer from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "pq_metadata.json", "r") as f:
            metadata = json.load(f)

        # Create quantizer
        quantizer = cls(metadata["d"], metadata["m"], metadata["k"])

        # Load codebooks from binary file
        codebooks_np = np.fromfile(path / "pq_codebooks.bin", dtype=np.float32)
        codebooks_np = codebooks_np.reshape(
            metadata["m"], metadata["k"], metadata["d_sub"]
        )
        quantizer.codebooks.data = torch.from_numpy(codebooks_np).float()
        quantizer.is_trained = True

        return quantizer


class PQDistance(nn.Module):
    """
    Optimized asymmetric distance computation for Product Quantization.

    Computes distances between a query vector and a list of pq encoded
    reference vectors using precomputed distance tables.
    Uses GatherElements optimization for ONNX compatibility.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query_vector: torch.Tensor,
        reference_codes: torch.Tensor,
        codebooks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distance between a query vector and a list of pq encoded reference vectors.

        Args:
            query_vector: Query vector [d]
            reference_codes: PQ codes for reference vectors [N, m]
            codebooks: Codebooks [m, k_centroids, d_sub]

        Returns:
            Tuple of (indices, distances)
        """
        N, m = reference_codes.shape
        _, k_centroids, d_sub = codebooks.shape

        # Reshape query vector into subvectors [m, d_sub]
        query_reshaped = query_vector.view(m, d_sub)

        # Expand query for broadcasting: [m, 1, d_sub]
        query_expanded = query_reshaped.unsqueeze(1)

        # Compute distance tables for all subspaces at once
        # codebooks: [m, k_centroids, d_sub]
        # query_expanded: [m, 1, d_sub]
        diff = query_expanded - codebooks  # [m, k_centroids, d_sub]
        distance_tables = (diff * diff).sum(dim=2)  # [m, k_centroids]

        # Use optimized gathering with proper indexing for ONNX
        # Create batch indices for each subspace
        batch_indices = (
            torch.arange(m, device=reference_codes.device).unsqueeze(0).expand(N, -1)
        )  # [N, m]

        # Create flat indices for GatherElements
        flat_batch_indices = batch_indices.flatten()  # [N*m]
        flat_reference_codes = reference_codes.flatten()  # [N*m]

        # Flatten distance tables for gathering: [m*k_centroids]
        flat_distance_tables = distance_tables.flatten()  # [m*k_centroids]

        # Compute linear indices: batch_idx * k_centroids + code
        linear_indices = flat_batch_indices * k_centroids + flat_reference_codes

        # Gather distances using linear indices
        gathered_distances = flat_distance_tables[linear_indices]  # [N*m]

        # Reshape back to [N, m] and sum across subspaces
        gathered_distances = gathered_distances.view(N, m)  # [N, m]
        total_distances = gathered_distances.sum(dim=1)  # [N]

        # Return all distances and indices sorted (let the caller decide how many to take)
        # This allows merging results from multiple partitions
        top_distances, indices = torch.topk(
            total_distances, N, largest=False, sorted=True
        )

        return indices.to(torch.int32), top_distances


def create_pq_system(
    training_vectors: torch.Tensor,
    m: int = 8,
    k: int = 256,
    max_iterations: int = 100,
    output_dir: Path = Path("."),
) -> Dict:
    """
    Complete PQ system creation: train codebooks and export codebooks and ONNX models.

    Args:
        training_vectors: Training vectors [N, d]
        m: Number of subquantizers
        k: Number of centroids per subquantizer
        max_iterations: Maximum k-means iterations
        output_dir: Directory to save all outputs

    Returns:
        Dictionary with codebooks and metadata
    """
    logger.info("=== Creating Product Quantization System ===")

    d = training_vectors.shape[1]

    # Create, train and save PQ
    pq = PQ(d, m, k)
    pq.train_pq(training_vectors, n_iterations=max_iterations)
    pq.save(output_dir)

    # Export ONNX models
    logger.info(f"Exporting PQ models with d={d}, m={m}, k={k}")

    # Export PQ forward (encode) as ONNX
    torch.onnx.export(
        pq,
        torch.zeros(100, d),
        output_dir / "pq_encode.onnx",
        input_names=["vectors"],
        output_names=["codes"],
        dynamic_axes={"vectors": {0: "n_points"}, "codes": {0: "n_points"}},
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

    # Export PQDistance with optimized distance computation
    distance_model = PQDistance()
    distance_model.eval()

    dummy_query_vector = torch.randn(d)
    dummy_reference_codes = torch.randint(0, k, (1000, m), dtype=torch.uint8)

    torch.onnx.export(
        distance_model,
        (dummy_query_vector, dummy_reference_codes, pq.codebooks.data),
        output_dir / "pq_distance.onnx",
        input_names=["query_vector", "reference_codes", "codebooks"],
        output_names=["indices", "distances"],
        dynamic_axes={
            "reference_codes": {0: "n_references"},
            "indices": {0: "k_out"},
            "distances": {0: "k_out"},
        },
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

    logger.info(f"Exported PQ models to {output_dir}")
    logger.info("- pq_encode.onnx: Encode vectors to PQ codes")
    logger.info(
        "- pq_distance.onnx: Optimized asymmetric distance computation (returns all distances sorted)"
    )

    result = {
        "codebooks": pq.codebooks.data,
        "metadata": {
            "d": pq.d,
            "m": pq.m,
            "k": pq.k,
            "d_sub": pq.d_sub,
            "compression_ratio": float((pq.d * 32) / (pq.m * 8)),
            "training_samples": training_vectors.shape[0],
            "max_iterations": max_iterations,
            "version": "1.0",
        },
    }

    logger.info("=== PQ System Creation Complete ===")
    return result
