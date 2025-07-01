"""
Product Quantization (PQ) implementation in PyTorch.

This module implements Product Quantization for vector compression and approximate
similarity search. It supports training on embedding vectors, ONNX export for
browser inference, and efficient encoding/decoding operations.

Based on the paper "Product Quantization for Nearest Neighbor Search" by Jégou et al.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class ProductQuantizer(nn.Module):
    """
    Product Quantization encoder/decoder module.

    Product Quantization decomposes high-dimensional vectors into multiple
    lower-dimensional subvectors, each quantized independently using k-means.
    This allows for significant compression while maintaining reasonable
    approximation quality.

    Args:
        d: Input vector dimension
        m: Number of subspaces (subquantizers)
        k: Number of centroids per subspace (codebook size)

    The input dimension d must be divisible by m.
    Each subspace has dimension d/m and k centroids.

    As an example, for 128-dimensional floating point vectors

    Training:
    - d=128 (input dimension) 128 * 32 bits = 4096 bits per vector
    - m=16 subspaces (128/16 = 8 dims per subspace)
    - k=256 centroids per subspace (stored in the codebook)

    Encoding:
    - Replace each 16 byte subspace with 8 bit codebook index of the closest centroid
    - 16 subspaces per vector, 8 bits per subspace -> 16*8 = 128 bits per vector
    - 4096 bits / 128 bits = 32x compression ratio
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
            # x_split[:, i]: [N, d_sub]
            # codebooks[i]: [k, d_sub]
            sub_vectors = x_split[:, i]  # [N, d_sub]
            centroids = self.codebooks[i]  # [k, d_sub]

            # Compute squared Euclidean distances using broadcasting (ONNX-friendly)
            # sub_vectors: [N, d_sub] -> [N, 1, d_sub]
            # centroids: [k, d_sub] -> [1, k, d_sub]
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

            # Look up centroids for this subspace
            decoded[:, start_idx:end_idx] = self.codebooks[i][codes[:, i]]

        return decoded

    def train_pq(
        self, vectors: torch.Tensor, n_iterations: int = 50, verbose: bool = True
    ) -> None:
        """
        Train the product quantizer using k-means clustering.

        Args:
            vectors: Training vectors of shape [N, d]
            n_iterations: Number of k-means iterations per subspace
            verbose: Whether to print training progress
        """
        if vectors.shape[1] != self.d:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match expected {self.d}"
            )

        n_vectors = vectors.shape[0]

        if verbose:
            logger.info(f"Training PQ with {n_vectors} vectors")
            logger.info(f"Parameters: d={self.d}, m={self.m}, k={self.k}")

        # Split vectors into subvectors
        vectors_split = vectors.view(n_vectors, self.m, self.d_sub)

        with torch.no_grad():
            for i in range(self.m):
                if verbose:
                    logger.info(f"Training subquantizer {i+1}/{self.m}")

                sub_vectors = vectors_split[:, i]  # [N, d_sub]

                # Initialize centroids randomly from data points
                perm = torch.randperm(n_vectors)[: self.k]
                centroids = sub_vectors[perm].clone()  # [k, d_sub]

                # K-means iterations
                for iteration in range(n_iterations):
                    # Assign points to nearest centroids using broadcasting
                    sub_vectors_expanded = sub_vectors.unsqueeze(1)  # [N, 1, d_sub]
                    centroids_expanded = centroids.unsqueeze(0)  # [1, k, d_sub]
                    diff = sub_vectors_expanded - centroids_expanded  # [N, k, d_sub]
                    distances = torch.sum(diff * diff, dim=2)  # [N, k]
                    assignments = torch.argmin(distances, dim=1)  # [N]

                    # Update centroids
                    new_centroids = torch.zeros_like(centroids)
                    for j in range(self.k):
                        mask = assignments == j
                        if mask.sum() > 0:
                            new_centroids[j] = sub_vectors[mask].mean(dim=0)
                        else:
                            # If no points assigned, keep current centroid
                            new_centroids[j] = centroids[j]

                    # Check for convergence
                    centroid_shift = torch.norm(centroids - new_centroids).item()
                    centroids = new_centroids

                    if verbose and (iteration + 1) % 10 == 0:
                        logger.info(
                            f"  Iteration {iteration+1}: centroid shift = {centroid_shift:.6f}"
                        )

                    if centroid_shift < 1e-6:
                        if verbose:
                            logger.info(f"  Converged after {iteration+1} iterations")
                        break

                # Store learned centroids
                self.codebooks.data[i] = centroids

        self.is_trained = True
        if verbose:
            logger.info("PQ training completed")

    def compute_reconstruction_error(self, vectors: torch.Tensor) -> float:
        """
        Compute mean squared reconstruction error on a set of vectors.

        Args:
            vectors: Test vectors of shape [N, d]

        Returns:
            Mean squared error between original and reconstructed vectors
        """
        if not self.is_trained:
            raise RuntimeError(
                "ProductQuantizer must be trained before computing error"
            )

        with torch.no_grad():
            codes = self.forward(vectors)
            reconstructed = self.decode(codes)
            mse = torch.mean((vectors - reconstructed) ** 2).item()

        return mse

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

    def export_onnx(self, output_path: Path, batch_size: int = 1) -> None:
        """
        Export the trained quantizer to ONNX format for browser inference.

        Args:
            output_path: Path to save the ONNX model
            batch_size: Batch size for the exported model (use -1 for dynamic)
        """
        if not self.is_trained:
            raise RuntimeError("Cannot export untrained quantizer")

        self.eval()

        # Create dummy input
        if batch_size == -1:
            dummy_input = torch.randn(1, self.d)
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        else:
            dummy_input = torch.randn(batch_size, self.d)
            dynamic_axes = None

        # Export to ONNX
        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

        logger.info(f"PQ model exported to {output_path}")
