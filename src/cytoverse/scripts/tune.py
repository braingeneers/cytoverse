#!/usr/bin/env python3
"""
Ray AIR-based hyperparameter tuning for IVFPQ index on SCimilarity embeddings.

This script utilizes Ray Data, Train, and Tune to:
1. Generate embeddings from h5ad files using SCimilarity
2. Split data into train/val/test sets
3. Train IVFPQ models with hyperparameter search using Ray Tune
4. Optimize for SCimilarity nearest neighbor overlap while minimizing search cost
"""

import ray
from ray import tune
from ray.data import Dataset
from ray.tune import Tuner, TuneConfig
from ray.air import session
from ray.air.config import RunConfig
import torch
import numpy as np
import pandas as pd
import anndata
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
from scimilarity import CellAnnotation
from scimilarity.utils import align_dataset, lognorm_counts
from sklearn.model_selection import train_test_split
import typer
import boto3
import tempfile

# Assume these are available - imported from the cytoverse package
# Add the project root to the path so we can import our modules
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.cytoverse.ivfpq.pq import ProductQuantizer
from src.cytoverse.ivfpq.ivf import InvertedFileIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="IVFPQ training and tuning for SCimilarity embeddings",
    add_completion=False,
)


@app.command()
def embed(
    h5ad_path: Path = typer.Argument(
        exists=True, file_okay=True, dir_okay=False, help="Path to local h5ad file"
    ),
    output_path: Path = typer.Argument(help="Path for output parquet file"),
    obs_keys: List[str] = typer.Option(
        ["CellType"], help="Observation keys to include as categorical columns"
    ),
    model_path: str = typer.Option(
        "~/data/scimilarity/model_v1.1", help="Path to SCimilarity model"
    ),
    batch_size: int = typer.Option(1024, help="Batch size for embedding generation"),
    num_samples: Optional[int] = typer.Option(
        None, help="Limit the number of cells to process (None = all cells)"
    ),
    num_workers: int = typer.Option(
        4, help="Number of Ray workers for parallel processing"
    ),
) -> None:
    """
    Generate SCimilarity embeddings from a local h5ad file and save to parquet.

    Uses Ray Datasets to process cells in parallel batches from the h5ad file, generates 128d embeddings
    using SCimilarity, and stores them along with specified observation metadata to a local parquet file.
    """
    logger.info(f"Processing h5ad file from {h5ad_path}")

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=num_workers)

    # Read the h5ad file in read-only mode to get metadata
    logger.info("Reading h5ad file metadata...")
    adata = anndata.read_h5ad(h5ad_path, backed="r")

    n_obs = adata.n_obs
    n_vars = adata.n_vars
    gene_names = adata.var_names.tolist()

    logger.info(f"File contains {n_obs} cells with {n_vars} genes")

    # Determine total cells to process
    total_cells = min(n_obs, num_samples) if num_samples is not None else n_obs
    logger.info(f"Will process {total_cells} cells in batches of {batch_size}")

    # Create a Ray Dataset that yields batches of expression data and metadata
    def create_h5ad_batches():
        """Generator that yields batches from AnnData object."""
        # Open h5ad file in backed mode for memory efficiency
        adata_full = anndata.read_h5ad(h5ad_path, backed="r")

        for start_idx in range(0, total_cells, batch_size):
            end_idx = min(start_idx + batch_size, total_cells)

            # Use AnnData slicing - it handles sparse matrices automatically
            batch_adata = adata_full[start_idx:end_idx]

            # Get expression matrix - AnnData handles sparse/dense conversion
            if hasattr(batch_adata.X, "toarray"):
                expression = batch_adata.X.toarray().astype(np.float32)
            else:
                expression = batch_adata.X.astype(np.float32)

            # Extract metadata using AnnData's obs
            metadata = {}
            for key in obs_keys:
                if key in batch_adata.obs:
                    metadata[key] = batch_adata.obs[key].tolist()
                else:
                    logger.warning(f"Key '{key}' not found in obs, filling with None")
                    metadata[key] = [None] * (end_idx - start_idx)

            yield {
                "expression": expression,
                "metadata": metadata,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "gene_names": gene_names,
            }

    # Create batch items for Ray Dataset
    logger.info("Creating Ray Dataset from h5ad batches...")
    batch_items = list(create_h5ad_batches())

    # Create Ray Dataset from the batch items
    expression_ds = ray.data.from_items(batch_items)

    # Define embedding function that will be applied to each batch
    class SCimilarityEmbedder:
        def __init__(self, model_path: str):
            self.model_path = model_path
            self.ca = None

        def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            # Initialize model lazily (once per worker)
            if self.ca is None:
                self.ca = CellAnnotation(model_path=self.model_path)

            # Get data from batch
            expression = batch["expression"]
            gene_names = batch["gene_names"]
            metadata = batch["metadata"]

            # Create AnnData object for preprocessing
            adata = anndata.AnnData(X=expression, var=pd.DataFrame(index=gene_names))

            # Use SCimilarity preprocessing pipeline
            adata = align_dataset(adata, self.ca.gene_order)
            adata = lognorm_counts(adata)

            # Get preprocessed expression data
            if hasattr(adata.X, "toarray"):
                expression_data = adata.X.toarray().astype(np.float32)
            else:
                expression_data = adata.X.astype(np.float32)

            # Generate embeddings
            embeddings = self.ca.get_embeddings(expression_data)

            # Prepare output records
            output_records = []
            for i in range(len(embeddings)):
                record = {"embedding": embeddings[i].tolist()}
                # Add metadata for this cell
                for key in metadata:
                    record[key] = metadata[key][i]
                output_records.append(record)

            return {"records": output_records}

    # Apply embedding transformation with parallel processing
    embedder = SCimilarityEmbedder(model_path)
    embedded_ds = expression_ds.map_batches(
        embedder,
        batch_format="dict",
        zero_copy_batch=False,
        num_cpus=1,  # Each batch uses 1 CPU
    )

    # Convert records to individual rows for streaming
    def records_to_rows(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert batch of records to individual rows."""
        return batch["records"]

    # Stream individual records
    rows_ds = embedded_ds.flat_map(records_to_rows)

    # Convert individual records to DataFrame format for parquet writing
    def format_for_parquet(batch: List[Dict[str, Any]]) -> pd.DataFrame:
        """Format batch of records as DataFrame for parquet writing."""
        df = pd.DataFrame(batch)
        # Convert metadata columns to categorical for better compression
        for key in obs_keys:
            if key in df.columns:
                df[key] = pd.Categorical(df[key])
        return df

    # Apply DataFrame formatting
    parquet_ds = rows_ds.map_batches(
        format_for_parquet,
        batch_format="python",
        batch_size=batch_size,  # Process in same batch sizes
    )

    # Write to local parquet file
    logger.info(f"Writing embeddings to {output_path}")
    parquet_ds.write_parquet(
        str(output_path),
        compression="snappy",
    )

    logger.info(f"Embeddings saved to {output_path}")

    # Shutdown Ray
    ray.shutdown()


@app.command()
def split(
    s3_embeddings_path: str = typer.Argument(
        ..., help="S3 path to embeddings parquet file"
    ),
    split_key: str = typer.Argument(
        ..., help="Observation key to use for stratified splitting"
    ),
    train_ratio: float = typer.Option(0.8, help="Ratio for training set"),
    val_ratio: float = typer.Option(0.1, help="Ratio for validation set"),
    test_ratio: float = typer.Option(0.1, help="Ratio for test set"),
    random_state: int = typer.Option(42, help="Random state for reproducibility"),
) -> None:
    """
    Split embeddings into stratified train/val/test sets.

    Creates three parquet files with -train, -val, and -test suffixes,
    stratified on the specified observation key.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    logger.info(f"Loading embeddings from {s3_embeddings_path}")

    # Parse S3 path
    s3 = boto3.client("s3")
    bucket_name = s3_embeddings_path.split("/")[2]
    key_name = "/".join(s3_embeddings_path.split("/")[3:])

    # Download parquet file
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
        s3.download_file(bucket_name, key_name, tmp_file.name)

        # Load dataframe
        df = pd.read_parquet(tmp_file.name)
        logger.info(f"Loaded {len(df)} embeddings")

        if split_key not in df.columns:
            raise ValueError(
                f"Split key '{split_key}' not found in dataframe columns: {df.columns.tolist()}"
            )

        # First split into train and temp (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df[split_key],
            random_state=random_state,
        )

        # Then split temp into val and test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_test_ratio,
            stratify=temp_df[split_key],
            random_state=random_state,
        )

        logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

        # Save splits
        base_path = s3_embeddings_path.rsplit(".", 1)[0]

        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            output_path = f"{base_path}-{split_name}.parquet"

            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_split:
                split_df.to_parquet(tmp_split.name, engine="pyarrow")

                output_bucket = output_path.split("/")[2]
                output_key = "/".join(output_path.split("/")[3:])
                s3.upload_file(tmp_split.name, output_bucket, output_key)

            logger.info(f"Saved {split_name} split to {output_path}")


def _load_embeddings_from_s3(s3_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata from S3 parquet file."""
    s3 = boto3.client("s3")
    bucket_name = s3_path.split("/")[2]
    key_name = "/".join(s3_path.split("/")[3:])

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
        s3.download_file(bucket_name, key_name, tmp_file.name)
        df = pd.read_parquet(tmp_file.name)

        # Extract embeddings as numpy array
        embeddings = np.vstack(df["embedding"].values)

        return embeddings, df


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


@app.command()
def tune(
    s3_train_path: str = typer.Argument(
        ..., help="S3 path to training embeddings parquet"
    ),
    s3_val_path: str = typer.Argument(
        ..., help="S3 path to validation embeddings parquet"
    ),
    s3_test_path: str = typer.Argument(..., help="S3 path to test embeddings parquet"),
    num_samples: int = typer.Option(
        10, help="Number of hyperparameter configurations to try"
    ),
    max_concurrent_trials: int = typer.Option(4, help="Maximum concurrent trials"),
) -> None:
    """
    Use Ray Tune to find optimal IVFPQ hyperparameters.

    Tunes over PQ parameters (m, k) and IVF parameters (n_partitions) to optimize
    the trade-off between compression, search efficiency, and reconstruction quality.
    """
    logger.info("Initializing Ray Tune for IVFPQ hyperparameter search")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Load embeddings from S3 into Ray datasets
    train_embeddings, _ = _load_embeddings_from_s3(s3_train_path)
    val_embeddings, _ = _load_embeddings_from_s3(s3_val_path)

    train_ds = ray.data.from_numpy(train_embeddings)
    val_ds = ray.data.from_numpy(val_embeddings)

    # Define search space
    search_space = {
        "pq_m": tune.choice([16, 32, 64]),  # Number of subquantizers
        "pq_k": tune.choice([256]),  # Codebook size
        "num_partitions": tune.choice([64, 128, 256, 512]),
        "n_probe": tune.choice([1, 2, 4, 8]),
        "probe_top_k": tune.choice([50]),  # Top k for evaluation
    }

    # Create tuner
    tuner = Tuner(
        tune.with_parameters(
            objective, train_ds=train_ds, val_ds=val_ds, model_path=""
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
            local_dir="./ray_results",
            log_to_file=True,
        ),
    )

    # Run tuning
    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result()
    logger.info(f"Best configuration found:")
    logger.info(f"  Config: {best_result.config}")
    logger.info(
        f"  Metrics: overlap={best_result.metrics['overlap']:.3f}, "
        f"memory={best_result.metrics['memory_mb']:.1f}MB"
    )

    # Test on test set
    logger.info("Evaluating best configuration on test set...")
    test_embeddings, _ = _load_embeddings_from_s3(s3_test_path)

    # You can add test evaluation here if needed

    ray.shutdown()


if __name__ == "__main__":
    app()
