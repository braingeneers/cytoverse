"""
Inverted File Index (IVF) implementation with Product Quantized (PQ) residual vectors for vector partitioning.

This module implements the IVF component of IVFPQ for partitioning large vector datasets into manageable chunks using residual vectors. Each vector is assigned to one of n_partitions partitions based on proximity to the partition centroids, and the residual (vector - centroid) is stored for PQ encoding.

The IVF index enables efficient approximate nearest neighbor search by:
1. Assigning query vectors to the most promising partitions
2. Computing query residuals and searching only within those partitions
3. Supporting asymmetric distance computation with residual vectors

Based on the IVFPQ approach with residual vectors for improved accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import logging
import json
from sklearn.cluster import KMeans
from tqdm import tqdm

from .pq import PQ, PQDistance, create_pq_system

logger = logging.getLogger(__name__)


def _train_ivfpq(
    vectors: torch.Tensor,
    output_dir: Path,
    n_partitions: int = None,
    n_training_vectors: int = None,
    sample_training_vectors: bool = False,
    max_iterations: int = 100,
    pq_m: int = 16,
    pq_k: int = 256,
) -> Dict:
    """
    Train IVFPQ index using residual vectors with scikit-learn KMeans and export

    Args:
        vectors: All vectors to be indexed [N, d]
        output_dir: Directory to save artifacts (pq, centroids, partitions)
        n_partitions: Number of partitions (default: sqrt(n_vectors))
        n_training_vectors: Number of training vectors to use for k-means (default: all vectors)
        sample_training_vectors: Randomly sample training vectors instead of taking first n (default: False)
        max_iterations: Maximum k-means iterations
        pq_m: Number of PQ subspaces
        pq_k: Number of centroids per PQ subspace

    Returns:
        Dictionary with training results and metadata
    """
    N, d = vectors.shape

    if n_partitions is None:
        n_partitions = int(np.sqrt(N))
        logger.info(
            f"n_partitions not specified, using default: {n_partitions} (sqrt(N))"
        )

    if n_training_vectors:
        if sample_training_vectors:
            # Randomly sample n_training_vectors from the entire dataset
            indices = torch.randperm(vectors.shape[0])[:n_training_vectors]
            training_vectors = vectors[indices]
        else:
            # Take the first n_training_vectors
            training_vectors = vectors[:n_training_vectors]
    else:
        training_vectors = vectors

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Training IVF with residual vectors: N={N}, d={d}, n_partitions={n_partitions}"
    )

    # Step 1: Run scikit-learn k-means on training vectors
    logger.info(
        f"Running scikit-learn k-means clustering on {training_vectors.shape[0]} training vectors..."
    )

    n_training_vectors = training_vectors.shape[0]
    assert (
        n_training_vectors > n_partitions
    ), "Training vectors must be greater than number of partitions"

    kmeans = KMeans(
        n_clusters=n_partitions,
        max_iter=max_iterations,
        random_state=42,
        n_init=1,  # Only run once since we set the random state
        verbose=0,
    )

    # Convert to numpy for sklearn - train on training_vectors
    training_vectors_np = training_vectors.cpu().numpy()
    kmeans.fit(training_vectors_np)

    # Predict assignments for all vectors
    assignments = kmeans.predict(vectors.cpu().numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()

    logger.info(f"K-means completed. Inertia: {kmeans.inertia_:.2f}")

    # Step 2: Compute residuals and organize by partition
    logger.info("Computing residuals and organizing by partition...")

    residuals_by_partition = {}
    vector_ids_by_partition = {}

    progress_bar = tqdm(range(n_partitions), desc="Processing partitions")

    for partition_id in progress_bar:
        # Find vectors assigned to this partition
        mask = assignments == partition_id
        partition_vectors = vectors[mask]
        partition_vector_ids = torch.nonzero(torch.from_numpy(mask)).squeeze(1)

        if len(partition_vectors) > 0:
            # Compute residuals: vector - centroid
            centroid = centroids[partition_id]
            residuals = partition_vectors - centroid.unsqueeze(0)

            residuals_by_partition[partition_id] = residuals
            vector_ids_by_partition[partition_id] = partition_vector_ids
        else:
            logger.warning(
                f"Partition {partition_id} has no vectors assigned. This may lead to empty partitions."
            )
            residuals_by_partition[partition_id] = torch.empty(0, d)
            vector_ids_by_partition[partition_id] = torch.empty(0, dtype=torch.long)

        progress_bar.set_postfix(vectors=len(partition_vectors))

    progress_bar.close()

    # Step 3: Train PQ on residual vectors and export partition files

    # Collect residuals for PQ training
    training_assignments = kmeans.predict(training_vectors.cpu().numpy())
    training_residuals_list = []
    for i in range(len(training_vectors)):
        partition_id = training_assignments[i]
        centroid = centroids[partition_id]
        residual = training_vectors[i] - centroid
        training_residuals_list.append(residual.unsqueeze(0))
    training_residuals = torch.cat(training_residuals_list, dim=0)

    assert len(training_residuals) > 0, "No training residuals available for PQ"

    # Train PQ codebooks on residuals
    pq_result = create_pq_system(
        training_residuals,
        m=pq_m,
        k=pq_k,
        max_iterations=max_iterations,
        output_dir=output_dir,
    )
    pq = PQ.load(output_dir)

    # Step 4: Encode residuals and export partition files
    logger.info("Encoding residuals with PQ and exporting partition files...")
    partitions_dir = output_dir / "partitions"
    partitions_dir.mkdir(exist_ok=True)

    partition_progress = tqdm(range(n_partitions), desc="Exporting partitions")

    assert (
        n_partitions < 10_000
    ), "n_partitions must be less than 10,000 for partition file naming"

    for partition_id in partition_progress:
        residuals = residuals_by_partition[partition_id]
        vector_ids = vector_ids_by_partition[partition_id]

        if len(residuals) > 0:
            # Encode residuals with PQ
            pq_codes = pq(residuals)

            # Export as binary file
            partition_file = partitions_dir / f"partition_{partition_id:04d}.bin"
            _export_partition_binary(partition_file, vector_ids, pq_codes)

            partition_progress.set_postfix(vectors=len(vector_ids))

    partition_progress.close()

    # Step 5: Export centroids as binary file
    centroids_file = output_dir / "ivf_centroids.bin"
    _export_centroids_binary(centroids_file, centroids)

    # Step 6: Export metadata
    partition_stats = {}
    for partition_id in range(n_partitions):
        partition_stats[partition_id] = len(vector_ids_by_partition[partition_id])

    metadata = {
        "d": d,
        "n_partitions": n_partitions,
        "pq_m": pq_m,
        "pq_k": pq_k,
        "total_vectors": N,
        "max_iterations": max_iterations,
        "inertia": float(kmeans.inertia_),
        "partition_sizes": partition_stats,
        "centroids_shape": list(centroids.shape),
        "version": "residual-1.1",
    }

    metadata_file = output_dir / "ivf_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Done!
    non_empty_partitions = sum(1 for size in partition_stats.values() if size > 0)
    logger.info(f"IVF training completed:")
    logger.info(f"  Total vectors: {N}")
    logger.info(f"  Non-empty partitions: {non_empty_partitions}/{n_partitions}")
    logger.info(f"  Average partition size: {N/non_empty_partitions:.1f}")
    logger.info(f"  Artifacts saved to: {output_dir}")

    return {
        "centroids": centroids,
        "assignments": assignments,
        "metadata": metadata,
        "residuals_by_partition": residuals_by_partition,
        "vector_ids_by_partition": vector_ids_by_partition,
    }


def _export_centroids_binary(file_path: Path, centroids: torch.Tensor) -> None:
    """Export centroids in binary format for efficient loading."""
    centroids_np = centroids.cpu().numpy().astype(np.float32)
    n_partitions, d = centroids_np.shape

    with open(file_path, "wb") as f:
        # Write header: [n_partitions:uint32, d:uint32]
        f.write(np.uint32(n_partitions).tobytes())
        f.write(np.uint32(d).tobytes())

        # Write centroids data
        f.write(centroids_np.tobytes())


def _load_centroids_binary(file_path: Path) -> torch.Tensor:
    """Load centroids from binary format."""
    with open(file_path, "rb") as f:
        # Read header
        n_partitions = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        d = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        # Read centroids data
        centroids_np = (
            np.frombuffer(f.read(), dtype=np.float32).reshape(n_partitions, d).copy()
        )

    return torch.from_numpy(centroids_np)


def _export_partition_binary(
    file_path: Path, vector_ids: torch.Tensor, pq_codes: torch.Tensor
) -> None:
    """Export partition data in binary format."""
    vector_ids_np = vector_ids.cpu().numpy().astype(np.int32)
    pq_codes_np = pq_codes.cpu().numpy().astype(np.uint8)

    num_vectors = len(vector_ids_np)
    m = pq_codes_np.shape[1]

    with open(file_path, "wb") as f:
        # Write header: [num_vectors:uint32, m:uint32]
        f.write(np.uint32(num_vectors).tobytes())
        f.write(np.uint32(m).tobytes())

        # Write data in interleaved format
        for i in range(num_vectors):
            f.write(vector_ids_np[i].tobytes())
            f.write(pq_codes_np[i].tobytes())


def _load_partition_binary(
    file_path: Path, m: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load partition data from binary format."""
    with open(file_path, "rb") as f:
        # Read header
        num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        m_file = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        assert m_file == m, f"Expected m={m}, got {m_file}"

        # Read interleaved data
        vector_ids = np.zeros(num_vectors, dtype=np.int32)
        pq_codes = np.zeros((num_vectors, m), dtype=np.uint8)

        for i in range(num_vectors):
            vector_ids[i] = np.frombuffer(f.read(4), dtype=np.int32)[0]
            pq_codes[i] = np.frombuffer(f.read(m), dtype=np.uint8)

    return torch.from_numpy(vector_ids), torch.from_numpy(pq_codes)


class IVFPQ(nn.Module):
    """
    Inverted File Index for dataset partitioning using residual vectors.

    This implementation uses scikit-learn for k-means clustering and works with
    residual vectors for improved accuracy in the IVFPQ pipeline.
    """

    def __init__(self, d: int, n_partitions: int):
        super().__init__()

        self.d = d
        self.n_partitions = n_partitions

        # Centroids will be loaded from trained model
        self.centroids = nn.Parameter(torch.randn(n_partitions, d), requires_grad=False)
        self.is_trained = False

        # Add PQ object for reuse
        self.pq = None

        logger.info(
            f"Initialized InvertedFileIndex: d={self.d}, n_partitions={self.n_partitions}"
        )

    def forward(
        self, query_vector: torch.Tensor, partition_centroids: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform coarse centroid search.

        Args:
            query_vector: Query vector of shape [d]
            partition_centroids: Partition centroids of shape [n_partitions, d]
            k: Number of nearest centroids to return

        Returns:
            Tuple of (top_k_indices, top_k_distances)
        """
        distances = torch.cdist(query_vector.unsqueeze(0), partition_centroids).squeeze(0)
        top_k_distances, top_k_indices = torch.topk(distances, k=k, largest=False)
        return top_k_indices, top_k_distances

    @classmethod
    def build(
        cls,
        vectors: torch.Tensor,
        output_dir: Path,
        n_partitions: int = None,
        num_training_vectors: Optional[int] = None,
        sample_training_vectors: bool = False,
        max_iterations: int = 100,
        pq_m: int = 16,
        pq_k: int = 256,
    ) -> "IVFPQ":
        """
        Build the IVFPQ index using residual vectors.

        Args:
            vectors: All vectors to be indexed [N, d]
            output_dir: Directory to save artifacts
            num_training_vectors: Number of training vectors to use for k-means (default: all vectors)
            sample_training_vectors: Randomly sample training vectors instead of taking first n (default: False)
            max_iterations: Number of k-means iterations
        """
        result = _train_ivfpq(
            vectors=vectors,
            output_dir=output_dir,
            n_partitions=n_partitions,
            n_training_vectors=num_training_vectors,
            sample_training_vectors=sample_training_vectors,
            max_iterations=max_iterations,
            pq_m=pq_m,
            pq_k=pq_k,
        )

        # Verify all vectors are assigned
        assignments = result["assignments"]
        assert len(assignments) == vectors.shape[0]
        assert np.all(assignments >= 0)

        ivf = IVFPQ.load(output_dir)

        # Export ONNX model for the forward method
        onnx_file = output_dir / "ivf_forward.onnx"
        dummy_query = torch.randn(vectors.shape[1])
        torch.onnx.export(
            ivf,
            (dummy_query, ivf.centroids.data, 4),  # Example inputs: query, partition_centroids, k
            onnx_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["query_vector", "partition_centroids", "k"],
            output_names=["top_k_indices", "top_k_distances"],
            dynamic_axes={
                "query_vector": {0: "d"},
                "partition_centroids": {0: "n_partitions", 1: "d"},
            },
        )

        assert np.all(assignments < ivf.n_partitions)

        return ivf

    def search(
        self,
        query_vector: torch.Tensor,
        model_path: Path,
        n_probe: int = 4,
        k: int = 50,
    ) -> Tuple[List[int], List[float]]:
        """
        Search for nearest neighbors using the trained IVF index.

        Args:
            query_vector: Query vector of shape [d]
            model_path: Path to model artifacts
            n_probe: Number of partitions to search
            k: Number of neighbors to return

        Returns:
            Tuple of (vector_ids, distances)
        """
        assert self.is_trained, "IVF index must be trained before searching"
        assert self.pq is not None, "PQ must be initialized before searching"

        # Step 1: Coarse centroid search
        top_partitions, _ = self.forward(query_vector, self.centroids.data, n_probe)

        # Step 2: Search each partition using PQ
        all_candidates = []
        partitions_dir = model_path / "partitions"
        pq_distance_partition = (
            PQDistance()
        )  # This k is now only used for initialization

        for partition_id in top_partitions:
            partition_id = partition_id.item()

            # Load partition data
            partition_file = partitions_dir / f"partition_{partition_id:04d}.bin"
            assert partition_file.exists(), f"Partition file {partition_file} not found"

            vector_ids, pq_codes = _load_partition_binary(partition_file, self.pq.m)

            assert len(vector_ids) > 0, f"Partition {partition_id} is empty"

            # Compute query residual for this partition
            query_residual = query_vector - self.centroids[partition_id]

            # Use asymmetric PQ distance computation - now returns all sorted results
            top_indices, top_distances = pq_distance_partition(
                query_residual, pq_codes, self.pq.codebooks
            )

            # Take only top k from this partition to limit memory usage
            # but we'll merge all partitions to get the global top k
            partition_k = min(k, len(top_indices))
            for i in range(partition_k):
                relative_idx = top_indices[i]
                dist = top_distances[i]
                if relative_idx < len(vector_ids):
                    vec_id = vector_ids[relative_idx.item()]
                    all_candidates.append((dist.item(), vec_id.item()))

        # Step 3: Sort all candidates from all partitions and return top k
        all_candidates.sort(key=lambda x: x[0])
        top_candidates = all_candidates[:k]

        distances = [dist for dist, _ in top_candidates]
        vector_ids = [vec_id for _, vec_id in top_candidates]

        return vector_ids, distances

    def save(self, path: Path) -> None:
        """Save the trained IVF index to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained IVF index")

        # Save centroids
        centroids_file = path / "ivf_centroids.bin"
        _export_centroids_binary(centroids_file, self.centroids.data)

        # Save metadata
        metadata = {
            "d": self.d,
            "n_partitions": self.n_partitions,
        }
        metadata_file = path / "ivf_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Export ONNX model for the forward method
        onnx_file = path / "ivf_forward.onnx"
        dummy_query = torch.randn(self.d)
        torch.onnx.export(
            self,
            (
                dummy_query,
                self.centroids.data,
                4,
            ),  # Example inputs: query, partition_centroids, k
            onnx_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["query_vector", "partition_centroids", "k"],
            output_names=["top_k_indices", "top_k_distances"],
            dynamic_axes={
                "query_vector": {0: "d"},
                "partition_centroids": {0: "n_partitions", 1: "d"},
            },
        )

    @classmethod
    def load(cls, path: Path) -> "IVFPQ":
        """Load a trained IVF index from disk."""

        metadata_file = path / "ivf_metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        ivf = cls(metadata["d"], metadata["n_partitions"])
        centroids_file = path / "ivf_centroids.bin"
        ivf.centroids.data = _load_centroids_binary(centroids_file)

        ivf.pq = PQ.load(path)

        ivf.is_trained = True

        return ivf
