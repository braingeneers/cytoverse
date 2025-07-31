#!/usr/bin/env python3
"""
Ray Tune hyperparameter tuning for IVFPQ index.

This script loads embeddings and labels from parquet files on disk
and uses Ray Tune to optimize IVFPQ hyperparameters.
"""

import ray
from ray import tune
from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import typer
from sklearn.model_selection import train_test_split

# Note: Consider installing src.cytoverse as a package or setting PYTHONPATH
# to avoid runtime sys.path modifications, which can bloat serialized environment.
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

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
    size_to_sample_from: int = 2_000_000,
    num_embeddings: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings and labels from parquet files with stratified sampling.
    Supports large datasets by only reading groups from the embeddings file
    after stratifying from the labels file, which we assume is small,
    can fit into memory, and is in the same order as the embeddings, and is
    categorical.
    """
    logger.info(
        f"Reading first {size_to_sample_from} labels for stratified splitting..."
    )
    stratify_labels_df = pd.read_parquet(labels_path)[:size_to_sample_from]
    stratify_label_indices = stratify_labels_df[stratify_by].cat.codes.values

    train_indices, test_indices = train_test_split(
        stratify_labels_df.index.values,
        train_size=int(num_embeddings * (80 / 100)),
        test_size=int(num_embeddings * (20 / 100)),
        stratify=stratify_label_indices,
        random_state=42,
    )

    # Combine train and test indices
    selected_indices = np.concatenate([train_indices, test_indices])
    selected_indices.sort()

    logger.info(
        f"Selected {len(train_indices)} train and {len(test_indices)} test samples"
    )

    # Load embeddings using the selected indices without reading entire file
    logger.info("Loading selected embeddings from parquet...")

    # Create sets for fast lookup
    train_set = set(train_indices)
    test_set = set(test_indices)

    # Storage for train and test data
    train_embeddings = []
    test_embeddings = []
    train_labels = []
    test_labels = []

    # Read embeddings row group by row group (only from first 2M samples)
    embeddings_file = pq.ParquetFile(embeddings_path)
    current_row = 0
    for i in range(embeddings_file.num_row_groups):
        row_group = embeddings_file.read_row_group(i)
        row_group_df = row_group.to_pandas()
        row_group_size = len(row_group_df)

        # Check which rows we need from this group
        row_end = current_row + row_group_size

        for j in range(row_group_size):
            global_idx = current_row + j

            # Skip if we're beyond the first 2M samples
            if global_idx >= size_to_sample_from:
                break

            if global_idx in train_set:
                # Extract embedding
                if "embedding" in row_group_df.columns:
                    embedding = row_group_df.iloc[j]["embedding"]
                else:
                    embedding = row_group_df.iloc[j].values
                train_embeddings.append(embedding)
                train_labels.append(stratify_label_indices[global_idx])
            elif global_idx in test_set:
                # Extract embedding
                if "embedding" in row_group_df.columns:
                    embedding = row_group_df.iloc[j]["embedding"]
                else:
                    embedding = row_group_df.iloc[j].values
                test_embeddings.append(embedding)
                test_labels.append(stratify_label_indices[global_idx])

        current_row = row_end

        # Early exit if we've collected all needed samples or passed 2M
        if (
            len(train_embeddings) == len(train_indices)
            and len(test_embeddings) == len(test_indices)
        ) or current_row >= size_to_sample_from:
            break

    # Convert to numpy arrays
    X_train = np.vstack(train_embeddings)
    X_test = np.vstack(test_embeddings)
    y_train = np.array(train_labels, dtype=np.int32)
    y_test = np.array(train_labels, dtype=np.int32)

    logger.info(
        f"Loaded embeddings - Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test


def train_ivfpq(
    train_embeddings: torch.Tensor, config: Dict[str, Any]
) -> Tuple[Any, Any, torch.Tensor, torch.Tensor]:
    """Train PQ and IVF models on the training embeddings."""
    vector_ids = torch.arange(len(train_embeddings), dtype=torch.int32)
    d = train_embeddings.shape[1]

    from src.cytoverse.ivfpq.pq import ProductQuantizer
    from src.cytoverse.ivfpq.ivf import InvertedFileIndex

    # Train PQ
    logger.info(f"Training PQ with m={config['pq_m']}, k={config['pq_k']}")
    pq = ProductQuantizer(d=d, m=config["pq_m"], k=config["pq_k"])
    pq.train_pq(train_embeddings, n_iterations=20)

    # Train IVF
    logger.info(f"Training IVF with {config['num_partitions']} partitions")
    ivf = InvertedFileIndex(d=d, n_partitions=config["num_partitions"])

    # Encode vectors with PQ
    pq_codes = pq(train_embeddings)

    # Build IVF index
    ivf.train_ivf(train_embeddings, vector_ids, n_iterations=4)

    return pq, ivf, train_embeddings, pq_codes


class IVFPQTrainable(tune.Trainable):
    def setup(self, config):
        # Retrieve data from object store
        self.train_embeddings = ray.get(config["train_embeddings_ref"])
        self.test_embeddings = ray.get(config["test_embeddings_ref"])
        self.y_train = ray.get(config["y_train_ref"])
        self.y_test = ray.get(config["y_test_ref"])
        self.config = config

    def step(self):
        from collections import Counter
        from src.cytoverse.ivfpq.pq import ProductQuantizer
        from src.cytoverse.ivfpq.ivf import InvertedFileIndex

        pq, ivf, _, train_pq_codes = train_ivfpq(self.train_embeddings, self.config)

        correct_predictions = 0
        total_samples = len(self.test_embeddings)
        unique_partitions = set()

        logger.info(
            f"Evaluating IVFPQ with n_probe={self.config['n_probe']}, "
            f"num_partitions={self.config['num_partitions']}, "
            f"pq_m={self.config['pq_m']}, pq_k={self.config['pq_k']}"
        )
        
        # Batch process all queries
        all_selected_partitions = ivf.search_partitions(
            self.test_embeddings, n_probe=self.config["n_probe"]
        )
        
        # Process each query's results
        for i, (query, selected_partitions) in enumerate(zip(self.test_embeddings, all_selected_partitions)):
            unique_partitions.update(selected_partitions)
            
            # Collect all codes from selected partitions
            all_partition_vectors = []
            all_partition_codes = []
            partition_offsets = [0]
            
            for partition_id in selected_partitions:
                partition_vectors = ivf.get_partition_vectors(partition_id)
                if len(partition_vectors) > 0:
                    all_partition_vectors.extend(partition_vectors)
                    all_partition_codes.append(train_pq_codes[partition_vectors])
                    partition_offsets.append(len(all_partition_vectors))
            
            if len(all_partition_vectors) > 0:
                # Compute distances for all candidates at once
                all_codes = torch.cat(all_partition_codes)
                distances = pq.compute_asymmetric_distances(query, all_codes)
                
                # Find top 50
                top_50_mask = torch.topk(distances, min(50, len(distances)), largest=False)
                top_50_local_indices = top_50_mask.indices
                top_50_indices = [all_partition_vectors[idx] for idx in top_50_local_indices]
                
                # Consensus prediction
                top_50_labels = self.y_train[top_50_indices]
                label_counts = Counter(top_50_labels)
                consensus_label = label_counts.most_common(1)[0][0]
                actual_label = self.y_test[i]
                if consensus_label == actual_label:
                    correct_predictions += 1

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        N_full = 23000000
        avg_vectors_per_partition = N_full / self.config["num_partitions"]
        bytes_per_vector = 4 + self.config["pq_m"]
        total_wire_bytes = (
            len(unique_partitions) * avg_vectors_per_partition * bytes_per_vector
        )
        total_wire_mb = total_wire_bytes / (1024 * 1024)

        return {
            "accuracy": accuracy,
            "total_wire_mb": total_wire_mb,
            "unique_partitions": len(unique_partitions),
            "combined_score": accuracy - (total_wire_mb / 1000),
            "done": True,
        }

    def save_checkpoint(self, checkpoint_dir: str):
        return None

    def load_checkpoint(self, checkpoint_path: str):
        pass


@app.command()
def tune_ivfpq(
    embeddings_path: str = typer.Argument(help="Path to embeddings parquet file"),
    labels_path: str = typer.Argument(help="Path to labels parquet file"),
    stratify_by: str = typer.Option("prediction", help="Column to stratify by"),
    num_samples: int = typer.Option(
        20, help="Number of hyperparameter configurations to try"
    ),
    max_concurrent_trials: int = typer.Option(4, help="Maximum concurrent trials"),
    debug: bool = typer.Option(
        False, help="Run in debug mode with local Ray and small search space"
    ),
) -> None:
    """
    Use Ray Tune to find optimal IVFPQ hyperparameters.

    Loads embeddings and labels from parquet files and tunes IVFPQ parameters.
    """
    logger.info("Loading embeddings and labels from parquet files")

    # Load data
    X_train, X_test, y_train, y_test = load_embeddings_and_labels(
        embeddings_path, labels_path, stratify_by
    )

    logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")

    # Initialize Ray
    if debug:
        ray.init(ignore_reinit_error=True, local_mode=True)
    else:
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    logging.getLogger("ray.train").setLevel(logging.CRITICAL)
    logging.getLogger("ray.tune").setLevel(logging.CRITICAL)

    # Place data in object store
    train_embeddings_ref = ray.put(torch.from_numpy(X_train))
    test_embeddings_ref = ray.put(torch.from_numpy(X_test))
    y_train_ref = ray.put(y_train)
    y_test_ref = ray.put(y_test)

    # Define search space with ObjectRefs
    if debug:
        search_space = {
            "pq_m": tune.choice([4]),
            "pq_k": tune.choice([64]),
            "num_partitions": tune.choice([64]),
            "n_probe": tune.choice([1]),
            "train_embeddings_ref": train_embeddings_ref,
            "test_embeddings_ref": test_embeddings_ref,
            "y_train_ref": y_train_ref,
            "y_test_ref": y_test_ref,
        }
    else:
        search_space = {
            "pq_m": tune.choice([16, 32, 64]),
            "pq_k": tune.choice([128, 256, 512]),
            "num_partitions": tune.choice([128, 256, 512]),
            "n_probe": tune.choice([1, 2, 4]),
            "train_embeddings_ref": train_embeddings_ref,
            "test_embeddings_ref": test_embeddings_ref,
            "y_train_ref": y_train_ref,
            "y_test_ref": y_test_ref,
        }

    # Create tuner
    tuner = Tuner(
        IVFPQTrainable,
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
        f"total_wire_mb={best_result.metrics['total_wire_mb']:.1f}MB, "
        f"unique_partitions={best_result.metrics['unique_partitions']}"
    )

    ray.shutdown()


if __name__ == "__main__":
    app()
