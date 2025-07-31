#!/usr/bin/env python3
"""
Ray Tune hyperparameter tuning for IVFPQ index.

This script loads embeddings and labels from parquet files on disk
and uses Ray Tune to optimize IVFPQ hyperparameters.
"""

import ray
from ray import tune
from ray.data import Dataset
from ray.tune import Tuner, TuneConfig
from ray.air import session
from ray.air.config import RunConfig, CheckpointConfig
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import typer
from sklearn.model_selection import train_test_split
from collections import Counter

import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.cytoverse.ivfpq.pq import ProductQuantizer
from src.cytoverse.ivfpq.ivf import InvertedFileIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="IVFPQ hyperparameter tuning",
    add_completion=False,
)


def load_embeddings_and_labels(
    embeddings_path: str,
    labels_path: str,
    stratify_by: str,
    max_embeddings: Optional[int] = None,
) -> Tuple[np.ndarray, pd.Series, pd.DataFrame]:
    """Load embeddings and labels from parquet files using pyarrow backend."""
    # Use pyarrow to read only needed rows
    embeddings_file = pq.ParquetFile(embeddings_path)
    labels_file = pq.ParquetFile(labels_path)

    # Read labels first to get stratification column
    if max_embeddings is not None:
        # Calculate how many row groups we need to read to get max_embeddings rows
        rows_read = 0
        row_groups_to_read = []
        for i in range(labels_file.num_row_groups):
            row_group_size = labels_file.metadata.row_group(i).num_rows
            if rows_read + row_group_size <= max_embeddings:
                row_groups_to_read.append(i)
                rows_read += row_group_size
            else:
                # Read partial row group if needed
                if rows_read < max_embeddings:
                    row_groups_to_read.append(i)
                break
        labels_table = labels_file.read_row_groups(row_groups_to_read)
        labels_df = labels_table.to_pandas()
        if len(labels_df) > max_embeddings:
            labels_df = labels_df.iloc[:max_embeddings]
    else:
        labels_df = labels_file.read().to_pandas()

    # Get unique labels and create label to index mapping
    unique_labels = labels_df[stratify_by].unique()
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Convert labels to indices
    label_indices = labels_df[stratify_by].map(label_to_idx).astype(np.int16)

    # Read embeddings
    if max_embeddings is not None:
        # Calculate how many row groups we need to read to get max_embeddings rows
        rows_read = 0
        row_groups_to_read = []
        for i in range(embeddings_file.num_row_groups):
            row_group_size = embeddings_file.metadata.row_group(i).num_rows
            if rows_read + row_group_size <= max_embeddings:
                row_groups_to_read.append(i)
                rows_read += row_group_size
            else:
                # Read partial row group if needed
                if rows_read < max_embeddings:
                    row_groups_to_read.append(i)
                break
        embeddings_table = embeddings_file.read_row_groups(row_groups_to_read)
        embeddings_df = embeddings_table.to_pandas()
        if len(embeddings_df) > max_embeddings:
            embeddings_df = embeddings_df.iloc[:max_embeddings]
    else:
        embeddings_df = embeddings_file.read().to_pandas()

    # Extract embeddings as numpy array
    if "embedding" in embeddings_df.columns:
        embeddings = np.vstack(embeddings_df["embedding"].values)
    else:
        # Assume all columns are embedding dimensions
        embeddings = embeddings_df.values

    return embeddings, label_indices, labels_df


def train_ivfpq(
    train_embeddings: np.ndarray, config: Dict[str, Any]
) -> Tuple[ProductQuantizer, InvertedFileIndex, torch.Tensor, torch.Tensor]:
    """Train PQ and IVF models on the training embeddings."""
    embeddings_tensor = torch.from_numpy(train_embeddings.copy()).float()
    vector_ids = torch.arange(len(train_embeddings), dtype=torch.int32)
    d = embeddings_tensor.shape[1]

    # Train PQ
    logger.info(f"Training PQ with m={config['pq_m']}, k={config['pq_k']}")
    pq = ProductQuantizer(d=d, m=config["pq_m"], k=config["pq_k"])
    pq.train_pq(embeddings_tensor, n_iterations=20)

    # Train IVF
    logger.info(f"Training IVF with {config['num_partitions']} partitions")
    ivf = InvertedFileIndex(d=d, n_partitions=config["num_partitions"])

    # Encode vectors with PQ
    pq_codes = pq(embeddings_tensor)

    # Build IVF index
    ivf.train_ivf(embeddings_tensor, vector_ids, n_iterations=4)

    return pq, ivf, embeddings_tensor, pq_codes


def objective(
    config: Dict[str, Any],
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Objective function for Ray Tune."""
    # Train IVFPQ
    pq, ivf, train_embeddings_tensor, train_pq_codes = train_ivfpq(
        train_embeddings, config
    )

    # Evaluate on test set
    correct_predictions = 0
    total_samples = 0

    test_embeddings_tensor = torch.from_numpy(test_embeddings.copy()).float()

    for i, query in enumerate(test_embeddings_tensor):
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
                candidates.extend(zip(partition_vectors, distances.tolist()))

        # Sort and get top 50
        candidates.sort(key=lambda x: x[1])
        top_50_indices = [c[0] for c in candidates[:50]]

        if len(top_50_indices) > 0:
            # Get label indices for top 50 nearest neighbors
            top_50_labels = y_train[top_50_indices]

            # Compute consensus label (most common)
            label_counts = Counter(top_50_labels)
            consensus_label = label_counts.most_common(1)[0][0]

            # Check if consensus matches actual label
            actual_label = y_test[i]
            if consensus_label == actual_label:
                correct_predictions += 1

        total_samples += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # Calculate estimated wire memory for full dataset (MB per query)
    N_full = 23000000
    avg_vectors_per_partition = N_full / config["num_partitions"]
    bytes_per_vector = 4 + config["pq_m"]  # ID (4 bytes) + PQ codes (m bytes)
    wire_bytes_per_query = (
        config["n_probe"] * avg_vectors_per_partition * bytes_per_vector
    )
    wire_mb = wire_bytes_per_query / (1024 * 1024)

    # Report metrics to Ray Tune
    session.report(
        {
            "accuracy": accuracy,
            "wire_mb": wire_mb,
            "combined_score": accuracy - (wire_mb / 1000),
        }
    )


@app.command()
def tune_ivfpq(
    embeddings_path: str = typer.Argument(help="Path to embeddings parquet file"),
    labels_path: str = typer.Argument(help="Path to labels parquet file"),
    stratify_by: str = typer.Option("prediction", help="Column to stratify by"),
    test_size: float = typer.Option(0.2, help="Ratio for test set"),
    num_samples: int = typer.Option(
        20, help="Number of hyperparameter configurations to try"
    ),
    max_concurrent_trials: int = typer.Option(4, help="Maximum concurrent trials"),
    max_embeddings: Optional[int] = typer.Option(
        100000, help="Maximum number of embeddings to load (None for all)"
    ),
) -> None:
    """
    Use Ray Tune to find optimal IVFPQ hyperparameters.

    Loads embeddings and labels from parquet files and tunes IVFPQ parameters.
    """
    logger.info("Loading embeddings and labels from parquet files")

    # Load data
    embeddings, label_indices, labels_df = load_embeddings_and_labels(
        embeddings_path, labels_path, stratify_by, max_embeddings
    )

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        label_indices.values,
        test_size=test_size,
        # stratify=label_indices.values,
        random_state=42,
    )

    logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")

    # Initialize Ray
    # ray.init(ignore_reinit_error=True, local_mode=True)
    ray.init(ignore_reinit_error=True, logging_level="ERROR")

    logging.getLogger("ray.train").setLevel(logging.CRITICAL)
    logging.getLogger("ray.tune").setLevel(logging.CRITICAL)

    # Define search space
    search_space = {
        "pq_m": tune.choice([16, 32, 64]),  # Number of subquantizers
        "pq_k": tune.choice([256]),  # Codebook size
        "num_partitions": tune.choice([64, 128, 256, 512]),
        "n_probe": tune.choice([1, 2, 4, 8]),
    }

    # Create tuner
    tuner = Tuner(
        tune.with_parameters(
            objective,
            train_embeddings=X_train,
            test_embeddings=X_test,
            y_train=y_train,
            y_test=y_test,
        ),
        param_space=search_space,
        tune_config=TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            metric="combined_score",
            mode="max",
        ),
        run_config=RunConfig(
            name="ivfpq_tune",
            log_to_file=False,
            checkpoint_config=CheckpointConfig(num_to_keep=None),
        ),
    )

    # Run tuning
    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result()
    logger.info(f"Best configuration found:")
    logger.info(f"  Config: {best_result.config}")
    logger.info(
        f"  Metrics: accuracy={best_result.metrics['accuracy']:.3f}, "
        f"wire_mb={best_result.metrics['wire_mb']:.1f}MB"
    )

    ray.shutdown()


if __name__ == "__main__":
    app()
