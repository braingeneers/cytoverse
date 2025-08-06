"""
Product Quantization (PQ) implementation for residual vectors with optimized ONNX export.

This module provides PyTorch models for Product Quantization on residual vectors including:
- Encode: quantizes residual embeddings using learned codebooks
- Decode: reconstructs residual embeddings from PQ codes
- Distance: computes asymmetric distances for efficient kNN search on residuals

Based on "Product Quantization for Nearest Neighbor Search" by Jégou et al.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import json
import pickle
import logging
from sklearn.cluster import KMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProductQuantizer(nn.Module):
    """
    Product Quantization encoder/decoder module for residual vectors.

    Product Quantization decomposes high-dimensional residual vectors into multiple
    lower-dimensional subvectors, each quantized independently using k-means.
    This allows for significant compression while maintaining reasonable
    approximation quality for residual-based search.

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
            f"Initialized ProductQuantizer: d={self.d}, m={self.m}, k={self.k}, d_sub={self.d_sub} for a compression ratio of {self.d * 32 / (self.m * 8)}x"
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
            raise RuntimeError("ProductQuantizer must be trained before encoding")

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
            raise RuntimeError("ProductQuantizer must be trained before decoding")

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

                # Use scikit-learn k-means
                kmeans = KMeans(
                    n_clusters=self.k,
                    max_iter=n_iterations,
                    random_state=42 + i,
                    n_init=1,  # Only run once since we set the random state
                )

                # Convert to numpy for sklearn
                sub_vectors_np = sub_vectors.cpu().numpy()
                kmeans.fit(sub_vectors_np)
                centroids = torch.from_numpy(kmeans.cluster_centers_).float()

                # Store learned centroids
                self.codebooks.data[i] = centroids

                # Check cluster utilization
                assignments = kmeans.labels_
                unique_assignments = np.unique(assignments)
                utilization = len(unique_assignments)
                subquantizer_pbar.set_postfix(clusters_used=f"{utilization}/{self.k}")

            subquantizer_pbar.close()

        self.is_trained = True

    def save(self, path: Path) -> None:
        """Save the trained quantizer to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained quantizer")

        save_dict = {
            "d": self.d,
            "m": self.m,
            "k": self.k,
            "codebooks": self.codebooks.data.cpu(),
            "is_trained": self.is_trained,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: Path) -> "ProductQuantizer":
        """Load a trained quantizer from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        quantizer = cls(save_dict["d"], save_dict["m"], save_dict["k"])
        quantizer.codebooks.data = save_dict["codebooks"]
        quantizer.is_trained = save_dict["is_trained"]

        return quantizer


class PQEncode(nn.Module):
    """
    Product Quantization encoding model for residual vectors.

    Given residual embeddings and codebooks, outputs PQ codes for each embedding.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, embeddings: torch.Tensor, codebooks: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode embeddings using product quantization.

        Args:
            embeddings: Input embeddings [N, d]
            codebooks: Codebooks [m, k, d_sub] where d = m * d_sub

        Returns:
            codes: PQ codes [N, m] as int64
        """
        N, d = embeddings.shape
        m, k, d_sub = codebooks.shape

        # Reshape embeddings into subvectors [N, m, d_sub]
        embeddings_reshaped = embeddings.view(N, m, d_sub)

        # Compute distances between each subvector and its corresponding codebook
        # embeddings_reshaped: [N, m, d_sub] -> [N, m, 1, d_sub]
        # codebooks: [m, k, d_sub] -> [1, m, k, d_sub]
        embeddings_expanded = embeddings_reshaped.unsqueeze(2)  # [N, m, 1, d_sub]
        codebooks_expanded = codebooks.unsqueeze(0)  # [1, m, k, d_sub]

        # Compute squared distances [N, m, k]
        diff = embeddings_expanded - codebooks_expanded
        distances = (diff * diff).sum(dim=3)  # [N, m, k]

        # Find nearest centroid for each subvector
        codes = torch.argmin(distances, dim=2)  # [N, m]

        return codes.long()


class PQDecode(nn.Module):
    """
    Product Quantization decoding model for residual vectors.

    Given PQ codes and codebooks, reconstructs approximate residual embeddings.
    """

    def __init__(self):
        super().__init__()

    def forward(self, codes: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
        """
        Decode PQ codes back to embeddings.

        Args:
            codes: PQ codes [N, m] as int64
            codebooks: Codebooks [m, k, d_sub]

        Returns:
            embeddings: Reconstructed embeddings [N, d] where d = m * d_sub
        """
        N, m = codes.shape
        _, _, d_sub = codebooks.shape
        d = m * d_sub

        # Initialize output
        embeddings = torch.zeros(N, d, dtype=codebooks.dtype, device=codes.device)

        # For each subquantizer
        for i in range(m):
            # Get codes for this subquantizer [N]
            sub_codes = codes[:, i]

            # Get corresponding centroids [N, d_sub]
            centroids = codebooks[i][sub_codes]

            # Place in output embeddings
            start_idx = i * d_sub
            end_idx = start_idx + d_sub
            embeddings[:, start_idx:end_idx] = centroids

        return embeddings


class PQDistanceResidual(nn.Module):
    """
    Optimized asymmetric distance computation for Product Quantization with residual vectors.

    Computes distances between a query residual and PQ-encoded reference residuals
    using precomputed distance tables and returns top-k nearest neighbors.
    Uses GatherElements optimization for ONNX compatibility.
    """

    def __init__(self, k: int = 50):
        super().__init__()
        self.k = k

    def forward(
        self,
        query_residual: torch.Tensor,
        reference_codes: torch.Tensor,
        codebooks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k nearest neighbors using asymmetric distance on residuals.

        Args:
            query_residual: Query residual vector [d]
            reference_codes: PQ codes for reference residuals [N, m]
            codebooks: Codebooks [m, k_centroids, d_sub]

        Returns:
            Tuple of (indices, distances) for k nearest neighbors
        """
        N, m = reference_codes.shape
        _, k_centroids, d_sub = codebooks.shape
        d = m * d_sub

        # Reshape query residual into subvectors [m, d_sub]
        query_reshaped = query_residual.view(m, d_sub)

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

        # Get top-k indices and distances
        k_actual = min(self.k, N)
        top_distances, indices = torch.topk(
            total_distances, k_actual, largest=False, sorted=True
        )

        return indices, top_distances


def train_pq_codebooks(
    training_vectors: torch.Tensor,
    m: int = 8,
    k: int = 256,
    max_iterations: int = 100,
    output_dir: str | Path = ".",
    save_binary: bool = True,
) -> Dict:
    """
    Train Product Quantization codebooks using scikit-learn k-means.

    Args:
        training_vectors: Training vectors [N, d]
        m: Number of subquantizers
        k: Number of centroids per subquantizer
        max_iterations: Maximum k-means iterations
        output_dir: Directory to save codebooks and metadata
        save_binary: Whether to save codebooks as binary file

    Returns:
        Dictionary containing codebooks and metadata
    """
    N, d = training_vectors.shape

    if d % m != 0:
        raise ValueError(f"Vector dimension {d} must be divisible by m={m}")

    d_sub = d // m
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(
        f"Training PQ codebooks with scikit-learn k-means: d={d}, m={m}, k={k}, d_sub={d_sub}"
    )

    # Train codebooks for each subspace
    codebooks = torch.zeros(m, k, d_sub)

    # Use scikit-learn k-means implementation
    # Progress bar for subquantizers
    subquantizer_pbar = tqdm(range(m), desc="Training PQ subquantizers")

    for i in subquantizer_pbar:
        subquantizer_pbar.set_description(f"Training subquantizer {i+1}/{m}")

        # Extract subvectors for this subspace
        start_idx = i * d_sub
        end_idx = start_idx + d_sub
        subvectors = training_vectors[:, start_idx:end_idx]

        # Adjust k if we have fewer samples than clusters
        n_samples = subvectors.shape[0]
        k_actual = min(k, n_samples)

        # Run scikit-learn k-means
        kmeans = KMeans(
            n_clusters=k_actual,
            max_iter=max_iterations,
            random_state=42 + i,  # Different seed for each subspace
            n_init=1,  # Only run once since we set the random state
            verbose=0,
        )

        # Convert to numpy for sklearn
        subvectors_np = subvectors.cpu().numpy()
        kmeans.fit(subvectors_np)
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()

        # If we had to reduce k, pad with zeros to maintain expected shape
        if k_actual < k:
            padded_centroids = torch.zeros(k, d_sub)
            padded_centroids[:k_actual] = centroids
            codebooks[i] = padded_centroids
        else:
            codebooks[i] = centroids

        # Check cluster utilization
        assignments = kmeans.labels_
        unique_assignments = np.unique(assignments)
        utilization = len(unique_assignments)
        subquantizer_pbar.set_postfix(clusters_used=f"{utilization}/{k_actual}")

    subquantizer_pbar.close()

    # Save codebooks and metadata
    metadata = {
        "d": d,
        "m": m,
        "k": k,
        "d_sub": d_sub,
        "compression_ratio": float((d * 32) / (m * 8)),
        "codebooks_shape": [m, k, d_sub],
        "codebooks_size": int(m * k * d_sub),
        "training_samples": N,
        "max_iterations": max_iterations,
        "version": "residual-1.0",
    }

    with open(output_dir / "pq_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if save_binary:
        # Save as binary file for web usage
        codebooks_np = codebooks.numpy().astype(np.float32)
        codebooks_np.tofile(output_dir / "pq_codebooks.bin")
        logger.info(f"Saved binary codebooks to {output_dir / 'pq_codebooks.bin'}")

    # Also save as .npy for Python usage
    np.save(output_dir / "pq_codebooks.npy", codebooks.numpy())

    logger.info(
        f"Training complete. Compression ratio: {metadata['compression_ratio']:.1f}x"
    )
    logger.info(f"Metadata saved to {output_dir / 'pq_metadata.json'}")

    return {"codebooks": codebooks, "metadata": metadata}


def export_pq_models(
    codebooks: torch.Tensor, output_dir: str | Path = ".", k_nn: int = 50
):
    """
    Export PQ models to ONNX format for residual vector processing.

    Args:
        codebooks: Trained codebooks [m, k, d_sub]
        output_dir: Directory to save ONNX models
        k_nn: Number of nearest neighbors for distance model (default: 50)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    m, k, d_sub = codebooks.shape
    d = m * d_sub

    logger.info(f"Exporting PQ models for residuals with d={d}, m={m}, k={k}")

    # Export PQEncode
    encode_model = PQEncode()
    encode_model.eval()

    dummy_embeddings = torch.randn(100, d)
    dummy_codebooks = codebooks

    torch.onnx.export(
        encode_model,
        (dummy_embeddings, dummy_codebooks),
        output_dir / "pq_encode.onnx",
        input_names=["embeddings", "codebooks"],
        output_names=["codes"],
        dynamic_axes={"embeddings": {0: "n_points"}, "codes": {0: "n_points"}},
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )

    # Export PQDecode
    decode_model = PQDecode()
    decode_model.eval()

    dummy_codes = torch.randint(0, k, (100, m))

    torch.onnx.export(
        decode_model,
        (dummy_codes, dummy_codebooks),
        output_dir / "pq_decode.onnx",
        input_names=["codes", "codebooks"],
        output_names=["embeddings"],
        dynamic_axes={"codes": {0: "n_points"}, "embeddings": {0: "n_points"}},
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )

    # Export PQDistanceResidual with optimized distance computation
    distance_model = PQDistanceResidual(k=k_nn)
    distance_model.eval()

    dummy_query_residual = torch.randn(d)
    dummy_reference_codes = torch.randint(0, k, (1000, m))

    torch.onnx.export(
        distance_model,
        (dummy_query_residual, dummy_reference_codes, dummy_codebooks),
        output_dir / "pq_distance.onnx",
        input_names=["query_residual", "reference_codes", "codebooks"],
        output_names=["indices", "distances"],
        dynamic_axes={
            "reference_codes": {0: "n_references"},
            "indices": {0: "k_out"},
            "distances": {0: "k_out"},
        },
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )

    logger.info(f"Exported PQ models to {output_dir}")
    logger.info("- pq_encode.onnx: Encode residual embeddings to PQ codes")
    logger.info("- pq_decode.onnx: Decode PQ codes to residual embeddings")
    logger.info(
        f"- pq_distance.onnx: Optimized asymmetric distance computation on residuals with top-k (k={k_nn})"
    )


def create_pq_system(
    training_vectors: torch.Tensor,
    m: int = 8,
    k: int = 256,
    max_iterations: int = 100,
    output_dir: str | Path = ".",
) -> Dict:
    """
    Complete PQ system creation for residual vectors: train codebooks and export ONNX models.

    Args:
        training_vectors: Training vectors [N, d]
        m: Number of subquantizers
        k: Number of centroids per subquantizer
        max_iterations: Maximum k-means iterations
        output_dir: Directory to save all outputs

    Returns:
        Dictionary with codebooks and metadata
    """
    logger.info("=== Creating Product Quantization System for Residual Vectors ===")

    # Train codebooks using scikit-learn k-means
    result = train_pq_codebooks(
        training_vectors, m=m, k=k, max_iterations=max_iterations, output_dir=output_dir
    )

    # Export ONNX models
    export_pq_models(result["codebooks"], output_dir)

    logger.info("=== PQ System Creation Complete ===")
    return result
