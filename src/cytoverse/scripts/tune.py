#!/usr/bin/env python3
"""
Ray AIR-based hyperparameter tuning for IVFPQ index on SCimilarity embeddings.

This script utilizes Ray Data, Train, and Tune to:
1. Stream embeddings from h5ad files through SCimilarity
2. Train IVFPQ models with hyperparameter search
3. Optimize for SCimilarity nearest neighbor overlap while minimizing search cost
"""

import ray
from ray import tune
from ray.data import Dataset
from ray.tune import Tuner, TuneConfig
from ray.air import session
from ray.air.config import RunConfig
import torch
import numpy as np
import anndata
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from scimilarity import CellAnnotation
from scimilarity.utils import align_dataset, lognorm_counts
from sklearn.model_selection import train_test_split

# Assume these are available - imported from the cytoverse package
# Add the project root to the path so we can import our modules
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.cytoverse.ivfpq.pq import ProductQuantizer
from src.cytoverse.ivfpq.ivf import InvertedFileIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class H5ADEmbeddingDataset:
    """Generate Ray Dataset that streams embeddings from h5ad file."""

    def __init__(self, h5ad_path: str, model_path: str, label_col: str = "CellType"):
        self.h5ad_path = h5ad_path
        self.model_path = model_path
        self.label_col = label_col

    def create_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train/val/test Ray datasets with 60/20/20 split."""

        # Load h5ad and SCimilarity model
        adata = anndata.read_h5ad(self.h5ad_path)
        ce = CellAnnotation(
            model_path=self.model_path, use_gpu=torch.cuda.is_available()
        )

        # Ensure counts layer exists
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

        # Get all indices and labels for stratification
        all_indices = np.arange(len(adata))
        all_labels = adata.obs[self.label_col].values

        # First split: 60% train, 40% temp (which will be split into 20% val, 20% test)
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            all_indices, all_labels, test_size=0.4, stratify=all_labels, random_state=42
        )

        # Second split: Split the 40% into 20% val and 20% test
        val_indices, test_indices, val_labels, test_labels = train_test_split(
            temp_indices,
            temp_labels,
            test_size=0.5,
            stratify=temp_labels,
            random_state=42,
        )

        # Process through SCimilarity in batches
        def process_batch(indices: List[int]) -> Dict[str, np.ndarray]:
            """Process a batch of cells through SCimilarity."""
            batch_adata = adata[indices].copy()

            # Align and normalize
            batch_adata = align_dataset(batch_adata, ce.gene_order)
            batch_adata = lognorm_counts(batch_adata)

            # Get embeddings
            if hasattr(batch_adata.X, "toarray"):
                expression_data = batch_adata.X.toarray().astype(np.float32)
            else:
                expression_data = batch_adata.X.astype(np.float32)

            embeddings = ce.get_embeddings(expression_data)

            return {"embeddings": embeddings, "indices": np.array(indices)}

        # Create Ray datasets - process in chunks to avoid memory issues
        batch_size = 1000

        def create_dataset_from_indices(indices: List[int]) -> Dataset:
            batches = []
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                batches.append(process_batch(batch_indices))
            return ray.data.from_items(batches)

        train_ds = create_dataset_from_indices(train_indices)
        val_ds = create_dataset_from_indices(val_indices)
        test_ds = create_dataset_from_indices(test_indices)

        return train_ds, val_ds, test_ds


def train_ivfpq(
    train_ds: Dataset, config: Dict[str, Any]
) -> Tuple[ProductQuantizer, InvertedFileIndex, torch.Tensor, torch.Tensor]:
    """Train PQ and IVF models on the training dataset."""

    # Collect all training embeddings
    all_embeddings = []

    for batch in train_ds.iter_batches(batch_size=None):
        # REMIND: Undo for real datasets
        # all_embeddings.append(batch["embeddings"])
        all_embeddings.append(batch["data"])

    embeddings = np.vstack(all_embeddings)

    embeddings_tensor = torch.from_numpy(embeddings).float()
    vector_ids = torch.arange(len(embeddings), dtype=torch.int32)

    d = embeddings_tensor.shape[1]

    # Train PQ
    logger.info(f"Training PQ with m={config['pq_m']}, k={config['pq_k']}")
    pq = ProductQuantizer(d=d, m=config["pq_m"], k=config["pq_k"])
    pq.train_pq(embeddings_tensor, n_iterations=20)  # Fewer iterations for tuning

    # Train IVF
    logger.info(f"Training IVF with {config['num_partitions']} partitions")
    ivf = InvertedFileIndex(d=d, n_partitions=config["num_partitions"])

    # Encode vectors with PQ
    pq_codes = pq(embeddings_tensor)

    # Build IVF index with PQ codes
    # ivf.train_ivf(embeddings_tensor, vector_ids, n_iterations=20)
    ivf.train_ivf(embeddings_tensor, vector_ids, n_iterations=4)

    # Add PQ codes to partitions
    # partitions = ivf.search_partitions(embeddings_tensor, n_probe=1)
    # for i, partition_ids in enumerate(partitions):
    #     if len(partition_ids) > 0:
    #         partition_id = partition_ids[0]  # Use closest partition
    #         ivf.add_to_partition(
    #             partition_id, vector_ids[i : i + 1], pq_codes[i : i + 1]
    #         )

    return pq, ivf, embeddings_tensor, pq_codes


def objective(
    config: Dict[str, Any], train_ds: Dataset, val_ds: Dataset, model_path: str
) -> None:
    """Objective function for Ray Tune."""

    # Train IVFPQ
    pq, ivf, train_embeddings, train_pq_codes = train_ivfpq(train_ds, config)

    # Evaluate on validation set
    total_overlap = 0
    total_samples = 0

    for batch in val_ds.iter_batches(batch_size=None):
        # REMIND: Undo for real datasets
        # val_embeddings = torch.from_numpy(batch["embeddings"]).float()
        val_embeddings = torch.from_numpy(batch["data"].copy()).float()

        for query in val_embeddings:
            # IVFPQ search
            selected_partitions = ivf.search_partitions(
                query.unsqueeze(0), n_probe=config["n_probe"]
            )[0]

            # Gather candidates from selected partitions
            candidates = []
            for partition_id in selected_partitions:
                partition_vectors = ivf.get_partition_vectors(partition_id)
                if len(partition_vectors) > 0:
                    # Get PQ codes for vectors in this partition
                    partition_codes = train_pq_codes[partition_vectors]

                    # Compute asymmetric distances using PQ
                    distances = pq.compute_asymmetric_distances(query, partition_codes)

                    # REMIND: Undo for real datasets
                    candidates = list(zip(partition_vectors, distances))
                    # for j, vec_id in enumerate(partition_vectors):
                    #     # candidates.append((vec_id.item(), distances[j].item()))
                    #     candidates.append((vec_id.item(), distances[j].item()))

            # Sort and get top k
            candidates.sort(key=lambda x: x[1])
            ivfpq_top_k = [c[0] for c in candidates[: config["probe_top_k"]]]

            # SCimilarity ground truth - find nearest neighbors
            query_batch = query.unsqueeze(0).numpy()

            # Compute distances to all training embeddings
            # NOTE: This is expensive - in production would use faiss or similar
            all_distances = np.linalg.norm(
                train_embeddings.numpy() - query_batch, axis=1
            )
            scimilarity_top_k = np.argsort(all_distances)[
                : config["probe_top_k"]
            ].tolist()

            # Calculate overlap
            overlap = len(set(ivfpq_top_k) & set(scimilarity_top_k))
            overlap_ratio = overlap / config["probe_top_k"]

            total_overlap += overlap_ratio
            total_samples += 1

    avg_overlap = total_overlap / total_samples if total_samples > 0 else 0

    # Calculate memory usage (approximate)
    # PQ: m * k * d_sub * 4 bytes (float32)
    # IVF: n_partitions * d * 4 bytes (centroids)
    d = train_embeddings.shape[1]
    pq_memory = config["pq_m"] * config["pq_k"] * (d // config["pq_m"]) * 4
    ivf_memory = config["num_partitions"] * d * 4
    total_memory_mb = (pq_memory + ivf_memory) / (1024 * 1024)

    # Report metrics to Ray Tune
    # We want to maximize overlap while minimizing memory
    session.report(
        {
            "overlap": avg_overlap,
            "memory_mb": total_memory_mb,
            "combined_score": avg_overlap
            - (total_memory_mb / 1000),  # Simple combined metric
        }
    )


def main(
    h5ad_path: str,
    model_path: str = "data/scimilarity/model_v1.1",
    num_samples: int = 1,
    max_concurrent_trials: int = 2,
):
    """Run hyperparameter tuning for IVFPQ."""

    # Initialize Ray
    ray.init(num_cpus=1, local_mode=True)
    # ray.init()

    # Create datasets
    # dataset_builder = H5ADEmbeddingDataset(h5ad_path, model_path)
    # train_ds, val_ds, _ = dataset_builder.create_dataset()

    # Create dataset from existing vector files
    # embeddings = np.load("data/scimilarity/vectors.npy")
    # np.save("data/scimilarity/train.npy", embeddings[0:100000])
    # np.save("data/scimilarity/val.npy", embeddings[100000:110000])
    train_ds = ray.data.from_numpy(np.load("data/scimilarity/train.npy"))
    val_ds = ray.data.from_numpy(np.load("data/scimilarity/val.npy"))

    # Define search space
    search_space = {
        "pq_m": tune.choice([4, 8]),
        "pq_k": tune.choice([64]),
        "num_partitions": tune.choice([8]),
        "n_probe": tune.choice([2]),
        "probe_top_k": tune.choice([10]),
    }

    # Create tuner
    tuner = Tuner(
        tune.with_parameters(
            objective, train_ds=train_ds, val_ds=val_ds, model_path=model_path
        ),
        param_space=search_space,
        # run_config=RunConfig(verbose=1),
        tune_config=TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            metric="combined_score",
            mode="max",
        ),
    )

    # Run tuning
    results = tuner.fit()

    # Print best results
    best_result = results.get_best_result()
    print("\nBest hyperparameters found:")
    print(f"Config: {best_result.config}")
    print(
        f"Metrics: overlap={best_result.metrics['overlap']:.3f}, "
        f"memory={best_result.metrics['memory_mb']:.1f}MB"
    )

    ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune IVFPQ hyperparameters for SCimilarity embeddings"
    )
    parser.add_argument("h5ad_path", type=str, help="Path to h5ad file")
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/scimilarity/model_v1.1",
        help="Path to SCimilarity model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of hyperparameter combinations to try",
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=2,
        help="Maximum number of concurrent trials",
    )

    args = parser.parse_args()

    main(args.h5ad_path, args.model_path, args.num_samples, args.max_concurrent_trials)
