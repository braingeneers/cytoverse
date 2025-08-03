"""
Product Quantization (PQ) implementation for ONNX export.

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
from cytoverse.kmeans import export_kmeans_models, KMeansComplete
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import tempfile


class PQEncode(nn.Module):
    """
    Product Quantization encoding model.

    Given embeddings and codebooks, outputs PQ codes for each embedding.
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
    Product Quantization decoding model.

    Given PQ codes and codebooks, reconstructs approximate embeddings.
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
        _, k, d_sub = codebooks.shape
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


class PQDistanceBase(nn.Module):
    """
    Base class for asymmetric Product Quantization Distance computation.
    Only computes distances without top-k selection.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        reference_codes: torch.Tensor,
        codebooks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distances using asymmetric distance.

        Args:
            query: Query vector [d]
            reference_codes: PQ codes for reference vectors [N, m]
            codebooks: Codebooks [m, k_centroids, d_sub]

        Returns:
            distances: Distances to all reference vectors [N]
        """
        d = query.shape[0]
        N, m = reference_codes.shape
        _, k_centroids, d_sub = codebooks.shape

        # Reshape query into subvectors [m, d_sub]
        query_reshaped = query.view(m, d_sub)

        # Expand query for broadcasting: [m, 1, d_sub]
        query_expanded = query_reshaped.unsqueeze(1)

        # Compute distance tables for all subspaces at once
        # codebooks: [m, k_centroids, d_sub]
        # query_expanded: [m, 1, d_sub]
        diff = query_expanded - codebooks  # [m, k_centroids, d_sub]
        distance_tables = (diff * diff).sum(dim=2)  # [m, k_centroids]

        # Gather distances for each code
        # reference_codes: [N, m]
        # distance_tables: [m, k_centroids]

        # We need to gather from distance_tables using reference_codes
        # First, we'll flatten and gather, then reshape

        # Create indices for gathering
        batch_indices = (
            torch.arange(m, device=reference_codes.device).unsqueeze(0).expand(N, -1)
        )  # [N, m]

        # Gather distances for each reference vector
        # This is essentially distance_tables[i, reference_codes[n, i]] for each n, i
        gathered_distances = distance_tables[batch_indices, reference_codes]  # [N, m]

        # Sum across subspaces to get total distances
        total_distances = gathered_distances.sum(dim=1)  # [N]

        return total_distances


class PQDistance(nn.Module):
    """
    Asymmetric distance computation for Product Quantization.

    Computes distances between a query vector and PQ-encoded reference vectors
    using precomputed distance tables for efficiency.
    """

    def __init__(self, k: int = 50):
        super().__init__()
        self.k = k
        self.distance_computer = PQDistanceBase()

    def forward(
        self,
        query: torch.Tensor,
        reference_codes: torch.Tensor,
        codebooks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute top-k nearest neighbors using asymmetric distance.

        Args:
            query: Query vector [d]
            reference_codes: PQ codes for reference vectors [N, m]
            codebooks: Codebooks [m, k_centroids, d_sub]

        Returns:
            indices: Indices of k nearest neighbors [k]
        """
        # Compute distances
        total_distances = self.distance_computer(query, reference_codes, codebooks)

        # Get top-k indices
        N = reference_codes.shape[0]
        k_actual = self.k
        if N < self.k:
            k_actual = N
        _, indices = torch.topk(total_distances, k_actual, largest=False, sorted=True)

        return indices


def train_pq_codebooks(
    training_vectors: torch.Tensor,
    m: int = 8,
    k: int = 256,
    max_iterations: int = 100,
    output_dir: str | Path = ".",
    save_binary: bool = True,
) -> Dict:
    """
    Train Product Quantization codebooks using ONNX k-means.

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

    print(f"Training PQ codebooks: d={d}, m={m}, k={k}, d_sub={d_sub}")

    # Export k-means ONNX models to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        export_kmeans_models(tmpdir)

        # Load ONNX k-means models
        init_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_init.onnx"))
        iter_session = ort.InferenceSession(str(Path(tmpdir) / "kmeans_iteration.onnx"))

        # Train codebooks for each subspace
        codebooks = torch.zeros(m, k, d_sub)

        for i in range(m):
            print(f"Training subquantizer {i+1}/{m}")

            # Extract subvectors for this subspace
            start_idx = i * d_sub
            end_idx = start_idx + d_sub
            subvectors = training_vectors[:, start_idx:end_idx]

            # Convert to numpy for ONNX inference
            subvectors_np = subvectors.numpy().astype(np.float32)
            k_np = np.array(k, dtype=np.int64)
            seed_np = np.array(
                42 + i, dtype=np.int64
            )  # Different seed for each subspace

            # Initialize centroids using ONNX model
            centroids_np = init_session.run(
                None, {"embeddings": subvectors_np, "k": k_np, "seed": seed_np}
            )[0]

            # Run k-means iterations using ONNX model
            for iteration in range(max_iterations):
                outputs = iter_session.run(
                    None, {"embeddings": subvectors_np, "centroids": centroids_np}
                )
                centroids_np, assignments_np, converged_np = outputs

                if converged_np > 0.5:
                    print(f"  Subspace {i}: converged after {iteration + 1} iterations")
                    break

            # Convert back to torch tensor
            codebooks[i] = torch.from_numpy(centroids_np)

            # Check cluster utilization
            unique_assignments = np.unique(assignments_np)
            print(f"  Subspace {i}: {len(unique_assignments)}/{k} clusters used")

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
        "version": "1.0",
    }

    with open(output_dir / "pq_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if save_binary:
        # Save as binary file for web usage
        codebooks_np = codebooks.numpy().astype(np.float32)
        codebooks_np.tofile(output_dir / "pq_codebooks.bin")
        print(f"Saved binary codebooks to {output_dir / 'pq_codebooks.bin'}")

    # Also save as .npy for Python usage
    np.save(output_dir / "pq_codebooks.npy", codebooks.numpy())

    print(f"Training complete. Compression ratio: {metadata['compression_ratio']:.1f}x")
    print(f"Metadata saved to {output_dir / 'pq_metadata.json'}")

    return {"codebooks": codebooks, "metadata": metadata}


def optimize_pq_distance_onnx(onnx_path: Path, optimized_path: Path):
    """
    Optimize PQ distance ONNX model by replacing gather operations with GatherElements.

    Args:
        onnx_path: Path to the original ONNX model
        optimized_path: Path to save the optimized ONNX model
    """
    # Load the ONNX model
    model = onnx.load(str(onnx_path))
    graph = model.graph

    # Find the pattern where we gather from distance_tables using reference_codes
    # The pattern typically involves:
    # 1. A Reshape or Expand operation for batch_indices
    # 2. A Gather operation that uses both batch_indices and reference_codes

    nodes_to_remove = []
    new_nodes = []

    for i, node in enumerate(graph.node):
        if node.op_type == "Gather" and len(node.input) >= 2:
            # Check if this is the gather operation we want to optimize
            # Look for patterns that suggest this is the distance table gathering

            # Find the distance_tables input (should come from a squeeze operation)
            distance_tables_input = None
            reference_codes_input = None

            for input_name in node.input:
                # Check if input comes from a node that computes distance tables
                for prev_node in graph.node:
                    if prev_node.output[0] == input_name:
                        if prev_node.op_type == "Squeeze":
                            # This might be distance_tables
                            distance_tables_input = input_name
                        elif "reference_codes" in input_name:
                            reference_codes_input = input_name

            if distance_tables_input and reference_codes_input:
                # Create GatherElements node to replace the complex gather pattern
                gather_elements_node = helper.make_node(
                    "GatherElements",
                    inputs=[distance_tables_input, reference_codes_input],
                    outputs=[node.output[0] + "_gather_elements"],
                    axis=1,  # Gather along the k_centroids dimension
                )
                new_nodes.append(gather_elements_node)

                # Update subsequent nodes to use the new output
                for subsequent_node in graph.node[i + 1 :]:
                    for j, input_name in enumerate(subsequent_node.input):
                        if input_name == node.output[0]:
                            subsequent_node.input[j] = gather_elements_node.output[0]

                nodes_to_remove.append(node)

    # Remove old nodes and add new ones
    for node in nodes_to_remove:
        graph.node.remove(node)
    graph.node.extend(new_nodes)

    # Save the optimized model
    onnx.save(model, str(optimized_path))
    print(f"Optimized ONNX model saved to {optimized_path}")


def export_pq_models(
    codebooks: torch.Tensor, output_dir: str | Path = ".", k_nn: int = 50
):
    """
    Export PQ models to ONNX format.

    Args:
        codebooks: Trained codebooks [m, k, d_sub]
        output_dir: Directory to save ONNX models
        k_nn: Number of nearest neighbors for distance model (default: 50)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    m, k, d_sub = codebooks.shape
    d = m * d_sub

    print(f"Exporting PQ models with d={d}, m={m}, k={k}")

    # Export k-means models for future training
    print("Exporting k-means models...")
    export_kmeans_models(output_dir)

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

    # Export PQDistance with fixed k (legacy, includes top-k)
    distance_model = PQDistance(k=k_nn)
    distance_model.eval()

    dummy_query = torch.randn(d)
    dummy_reference_codes = torch.randint(0, k, (1000, m))

    torch.onnx.export(
        distance_model,
        (dummy_query, dummy_reference_codes, dummy_codebooks),
        output_dir / "pq_distance.onnx",
        input_names=["query", "reference_codes", "codebooks"],
        output_names=["indices"],
        dynamic_axes={"reference_codes": {0: "n_references"}, "indices": {0: "k_out"}},
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )

    # Export PQDistanceBase (distance computation only, no top-k)
    distance_base_model = PQDistanceBase()
    distance_base_model.eval()

    torch.onnx.export(
        distance_base_model,
        (dummy_query, dummy_reference_codes, dummy_codebooks),
        output_dir / "pq_distance_base.onnx",
        input_names=["query", "reference_codes", "codebooks"],
        output_names=["distances"],
        dynamic_axes={
            "reference_codes": {0: "n_references"},
            "distances": {0: "n_references"},
        },
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )

    # Attempt to optimize the distance base model with GatherElements
    try:
        optimize_pq_distance_onnx(
            output_dir / "pq_distance_base.onnx",
            output_dir / "pq_distance_base_optimized.onnx",
        )
    except Exception as e:
        print(f"Warning: Could not optimize ONNX model: {e}")
        print("The unoptimized model will still work correctly.")

    print(f"Exported PQ models to {output_dir}")
    print("- kmeans_init.onnx: K-means centroid initialization")
    print("- kmeans_iteration.onnx: K-means single iteration")
    print("- kmeans_complete.onnx: Complete k-means (for testing)")
    print("- pq_encode.onnx: Encode embeddings to PQ codes")
    print("- pq_decode.onnx: Decode PQ codes to embeddings")
    print(f"- pq_distance.onnx: Asymmetric distance computation with top-k (k={k_nn})")
    print("- pq_distance_base.onnx: Distance computation only (no top-k)")
    print(
        "- pq_distance_base_optimized.onnx: Optimized distance computation (if successful)"
    )


def create_pq_system(
    training_vectors: torch.Tensor,
    m: int = 8,
    k: int = 256,
    max_iterations: int = 100,
    output_dir: str | Path = ".",
) -> Dict:
    """
    Complete PQ system creation: train codebooks and export ONNX models.

    Args:
        training_vectors: Training vectors [N, d]
        m: Number of subquantizers
        k: Number of centroids per subquantizer
        max_iterations: Maximum k-means iterations
        output_dir: Directory to save all outputs

    Returns:
        Dictionary with codebooks and metadata
    """
    print("=== Creating Product Quantization System ===")

    # Train codebooks
    result = train_pq_codebooks(
        training_vectors, m=m, k=k, max_iterations=max_iterations, output_dir=output_dir
    )

    # Export ONNX models
    export_pq_models(result["codebooks"], output_dir)

    print("=== PQ System Creation Complete ===")
    return result
