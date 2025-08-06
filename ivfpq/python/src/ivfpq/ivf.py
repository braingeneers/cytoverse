"""
Inverted File Index (IVF) implementation with residual vectors for vector partitioning.

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
from typing import Optional, Tuple, List, Union, Dict
from pathlib import Path
import logging
import json
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle

from .pq import train_pq_codebooks, export_pq_models, ProductQuantizer

logger = logging.getLogger(__name__)


def train_ivf_residual(
    vectors: torch.Tensor,
    n_partitions: int,
    output_dir: Path,
    training_vectors: Optional[torch.Tensor] = None,
    max_iterations: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Train IVF index using residual vectors with scikit-learn KMeans.

    Args:
        vectors: All vectors to be indexed [N, d]
        n_partitions: Number of partitions
        output_dir: Directory to save artifacts
        training_vectors: Subset of vectors for k-means training [N_train, d] (default: use all vectors)
        max_iterations: Maximum k-means iterations
        verbose: Whether to show progress

    Returns:
        Dictionary with training results and metadata
    """
    N, d = vectors.shape
    
    # Use training_vectors for k-means if provided, otherwise use all vectors
    if training_vectors is None:
        training_vectors = vectors
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info(
            f"Training IVF with residual vectors: N={N}, d={d}, n_partitions={n_partitions}"
        )

    # Step 1: Run scikit-learn k-means on training vectors
    if verbose:
        logger.info(f"Running scikit-learn k-means clustering on {training_vectors.shape[0]} training vectors...")

    # Adjust n_partitions if we have fewer training vectors than partitions
    n_training_vectors = training_vectors.shape[0]
    n_partitions_actual = min(n_partitions, n_training_vectors)

    if n_partitions_actual < n_partitions and verbose:
        logger.info(
            f"Reducing partitions from {n_partitions} to {n_partitions_actual} due to insufficient training vectors"
        )

    kmeans = KMeans(
        n_clusters=n_partitions_actual,
        max_iter=max_iterations,
        random_state=42,
        n_init=1,  # Only run once since we set the random state
        verbose=0,
    )

    # Convert to numpy for sklearn - train on training_vectors
    training_vectors_np = training_vectors.cpu().numpy()
    kmeans.fit(training_vectors_np)
    
    # Predict assignments for all vectors
    vectors_np = vectors.cpu().numpy()
    assignments = kmeans.predict(vectors_np)
    centroids_actual = torch.from_numpy(kmeans.cluster_centers_).float()

    # If we had to reduce partitions, pad with zeros to maintain expected shape
    if n_partitions_actual < n_partitions:
        centroids = torch.zeros(n_partitions, vectors.shape[1])
        centroids[:n_partitions_actual] = centroids_actual
    else:
        centroids = centroids_actual

    if verbose:
        logger.info(f"K-means completed. Inertia: {kmeans.inertia_:.2f}")

    # Step 2: Compute residuals and organize by partition
    residuals_by_partition = {}
    vector_ids_by_partition = {}

    if verbose:
        logger.info("Computing residuals and organizing by partition...")

    progress_bar = tqdm(
        range(n_partitions), desc="Processing partitions", disable=not verbose
    )

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
            residuals_by_partition[partition_id] = torch.empty(0, d)
            vector_ids_by_partition[partition_id] = torch.empty(0, dtype=torch.long)

        progress_bar.set_postfix(vectors=len(partition_vectors))

    progress_bar.close()

    # Step 3: Train PQ on residual vectors and export partition files

    # Collect residuals for PQ training - use training vectors' residuals if available
    if training_vectors is not vectors:
        # Recompute residuals for training vectors only
        training_assignments = kmeans.predict(training_vectors.cpu().numpy())
        training_residuals_list = []
        for i in range(len(training_vectors)):
            partition_id = training_assignments[i]
            centroid = centroids[partition_id]
            residual = training_vectors[i] - centroid
            training_residuals_list.append(residual.unsqueeze(0))
        training_residuals = torch.cat(training_residuals_list, dim=0)
    else:
        # Use all residuals for PQ training
        all_residuals = []
        for residuals in residuals_by_partition.values():
            if len(residuals) > 0:
                all_residuals.append(residuals)
        if all_residuals:
            training_residuals = torch.cat(all_residuals, dim=0)
        else:
            training_residuals = torch.empty(0, d)
    
    if len(training_residuals) > 0:

        if verbose:
            logger.info(f"Training PQ on {len(training_residuals)} training residual vectors...")

        # Train PQ codebooks on residuals
        pq_result = train_pq_codebooks(
            training_residuals,
            m=16,  # Default number of subquantizers
            k=256,  # Default codebook size
            max_iterations=50,
            output_dir=output_dir,
            save_binary=True,
        )

        # Export PQ ONNX models
        export_pq_models(pq_result["codebooks"], output_dir, k_nn=50)

        # Load PQ for encoding residuals
        pq = ProductQuantizer(d=d, m=16, k=256)
        pq.codebooks.data = pq_result["codebooks"]
        pq.is_trained = True

        if verbose:
            logger.info("Encoding residuals with PQ and exporting partition files...")

        # Step 4: Encode residuals and export partition files
        partitions_dir = output_dir / "partitions"
        partitions_dir.mkdir(exist_ok=True)

        partition_progress = tqdm(
            range(n_partitions), desc="Exporting partitions", disable=not verbose
        )

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
    centroids_file = output_dir / "centroids.bin"
    _export_centroids_binary(centroids_file, centroids)

    # Step 6: Export metadata
    partition_stats = {}
    for partition_id in range(n_partitions):
        partition_stats[partition_id] = len(vector_ids_by_partition[partition_id])

    metadata = {
        "d": d,
        "n_partitions": n_partitions,
        "total_vectors": N,
        "max_iterations": max_iterations,
        "inertia": float(kmeans.inertia_),
        "partition_sizes": partition_stats,
        "centroids_shape": list(centroids.shape),
        "version": "residual-1.0",
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
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


def search_ivf_residual(
    query_vector: torch.Tensor,
    centroids: torch.Tensor,
    n_probe: int,
    model_path: Path,
    k: int = 50,
    verbose: bool = False,
) -> Tuple[List[int], List[float]]:
    """
    Search for nearest neighbors using IVF with residual vectors.

    Args:
        query_vector: Query vector of shape [d]
        centroids: Partition centroids of shape [n_partitions, d]
        n_probe: Number of partitions to search
        model_path: Path to model artifacts
        k: Number of neighbors to return
        verbose: Whether to show progress

    Returns:
        Tuple of (vector_ids, distances)
    """
    d = query_vector.shape[0]
    n_partitions = centroids.shape[0]

    # Step 1: Find nearest n_probe partitions using exact precision
    distances_to_centroids = torch.cdist(query_vector.unsqueeze(0), centroids).squeeze(
        0
    )
    _, top_partitions = torch.topk(
        distances_to_centroids, k=min(n_probe, n_partitions), largest=False
    )

    if verbose:
        logger.info(
            f"Searching {len(top_partitions)} partitions for {k} nearest neighbors"
        )

    # Step 2: Load PQ model for distance computation

    # Load PQ metadata
    pq_metadata_file = model_path / "pq_metadata.json"
    with open(pq_metadata_file, "r") as f:
        pq_metadata = json.load(f)

    # Load PQ codebooks
    codebooks_file = model_path / "pq_codebooks.npy"
    codebooks = torch.from_numpy(np.load(codebooks_file))

    pq = ProductQuantizer(d=pq_metadata["d"], m=pq_metadata["m"], k=pq_metadata["k"])
    pq.codebooks.data = codebooks
    pq.is_trained = True

    # Step 3: Search each partition using PQ distance on residuals
    all_candidates = []

    partitions_dir = model_path / "partitions"

    for partition_id in top_partitions:
        partition_id = partition_id.item()

        # Compute query residual
        query_residual = query_vector - centroids[partition_id]

        # Load partition data
        partition_file = partitions_dir / f"partition_{partition_id:04d}.bin"

        if partition_file.exists():
            vector_ids, pq_codes = _load_partition_binary(
                partition_file, pq_metadata["m"]
            )

            if len(vector_ids) > 0:
                # Use PQ distance computation on residuals
                # Decode PQ codes to get approximate residuals
                reconstructed_residuals = pq.decode(pq_codes)

                # Compute distances between query residual and reconstructed residuals
                query_expanded = query_residual.unsqueeze(0).expand(
                    len(reconstructed_residuals), -1
                )
                distances = torch.norm(reconstructed_residuals - query_expanded, dim=1)

                # Add candidates with their distances
                for i, (vec_id, dist) in enumerate(zip(vector_ids, distances)):
                    all_candidates.append((dist.item(), vec_id.item()))

    # Step 4: Sort all candidates and return top k
    all_candidates.sort(key=lambda x: x[0])
    top_candidates = all_candidates[:k]

    distances = [dist for dist, _ in top_candidates]
    vector_ids = [vec_id for _, vec_id in top_candidates]

    return vector_ids, distances


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


def load_centroids_binary(file_path: Path) -> torch.Tensor:
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


class InvertedFileIndex(nn.Module):
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

        logger.info(
            f"Initialized InvertedFileIndex: d={self.d}, n_partitions={self.n_partitions}"
        )

    def train_ivf(
        self,
        vectors: torch.Tensor,
        output_dir: Path,
        training_vectors: Optional[torch.Tensor] = None,
        max_iterations: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the IVF index using residual vectors.

        Args:
            vectors: All vectors to be indexed [N, d]
            output_dir: Directory to save artifacts
            training_vectors: Subset of vectors for k-means training [N_train, d] (default: use all vectors)
            max_iterations: Number of k-means iterations
            verbose: Whether to print progress
        """
        result = train_ivf_residual(
            vectors=vectors,
            n_partitions=self.n_partitions,
            output_dir=output_dir,
            training_vectors=training_vectors,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        self.centroids.data = result["centroids"]
        self.is_trained = True

        return result

    def search(
        self,
        query_vector: torch.Tensor,
        model_path: Path,
        n_probe: int = 4,
        k: int = 50,
        verbose: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        Search for nearest neighbors using the trained IVF index.

        Args:
            query_vector: Query vector of shape [d]
            model_path: Path to model artifacts
            n_probe: Number of partitions to search
            k: Number of neighbors to return
            verbose: Whether to show progress

        Returns:
            Tuple of (vector_ids, distances)
        """
        if not self.is_trained:
            # Load centroids from model_path
            centroids_file = model_path / "centroids.bin"
            self.centroids.data = load_centroids_binary(centroids_file)
            self.is_trained = True

        return search_ivf_residual(
            query_vector=query_vector,
            centroids=self.centroids.data,
            n_probe=n_probe,
            model_path=model_path,
            k=k,
            verbose=verbose,
        )

    def save(self, path: Path) -> None:
        """Save the trained IVF index to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained IVF index")

        save_dict = {
            "d": self.d,
            "n_partitions": self.n_partitions,
            "centroids": self.centroids.data.cpu(),
            "is_trained": self.is_trained,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: Path) -> "InvertedFileIndex":
        """Load a trained IVF index from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        ivf = cls(save_dict["d"], save_dict["n_partitions"])
        ivf.centroids.data = save_dict["centroids"]
        ivf.is_trained = save_dict["is_trained"]

        return ivf
