#!/usr/bin/env python3

import logging
import os
import typer
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from typing import Union

import onnxruntime as ort
from tqdm import tqdm
import s3fs

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _load_h5ad_from_path_or_s3(path: Union[str, Path]) -> sc.AnnData:
    """
    Load h5ad file from local path or S3 URI.

    Args:
        path: Local file path or S3 URI (s3://bucket/key)

    Returns:
        Loaded AnnData object
    """
    path_str = str(path)

    if path_str.startswith("s3://"):
        logger.info("Loading from S3: %s", path_str)
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path_str, "rb") as f:
            adata = sc.read_h5ad(f)
        logger.info("  S3 file loaded successfully")
    else:
        logger.info("Loading from local path: %s", path_str)
        adata = sc.read_h5ad(path_str)

    return adata


def _validate_and_convert_to_csr(adata: sc.AnnData) -> sc.AnnData:
    """
    Validate that .X is in CSR format for efficient cell-by-cell access.

    CSR (Compressed Sparse Row) format enables efficient row slicing,
    which is critical for processing cells in batches during embedding.

    Args:
        adata: AnnData object to validate

    Returns:
        AnnData with .X in CSR format (converts if necessary)
    """
    from scipy.sparse import issparse, isspmatrix_csr, csr_matrix

    if not issparse(adata.X):
        logger.warning("  .X is dense array, converting to CSR for efficiency")
        adata.X = csr_matrix(adata.X)
    elif not isspmatrix_csr(adata.X):
        current_format = type(adata.X).__name__
        logger.warning(
            "  .X is in %s format, converting to CSR for efficient row access",
            current_format,
        )
        adata.X = adata.X.tocsr()
    else:
        logger.info("  ✅ .X is already in CSR format")

    return adata


def _random_sample_cells(adata: sc.AnnData, max_cells: int, seed: int) -> sc.AnnData:
    """
    Randomly sample cells from AnnData object.

    Uses stratified sampling when possible to preserve label distributions.

    Args:
        adata: AnnData object to sample from
        max_cells: Maximum number of cells to keep
        seed: Random seed for reproducibility

    Returns:
        Sampled AnnData object
    """
    n_cells = adata.n_obs

    assert max_cells > 0, f"max_cells must be positive, got {max_cells}"
    assert (
        max_cells <= n_cells
    ), f"max_cells ({max_cells}) exceeds dataset size ({n_cells})"

    logger.info("Randomly sampling %d cells from %d total", max_cells, n_cells)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Random sample without replacement
    indices = np.random.choice(n_cells, size=max_cells, replace=False)
    indices.sort()  # Sort for consistent ordering

    sampled = adata[indices, :].copy()
    logger.info("  Sample indices: min=%d, max=%d", indices.min(), indices.max())

    return sampled


def _load_model_genes(genes_path: Path) -> list[str]:
    """
    Load model gene vocabulary from genes.txt.

    Args:
        genes_path: Path to genes.txt file (one gene per line)

    Returns:
        List of gene names in model order
    """
    with open(genes_path, "r") as f:
        genes = [line.strip() for line in f if line.strip()]

    logger.info("Loaded %d genes from %s", len(genes), genes_path)
    return genes


def _create_inflation_indices(
    sample_genes: list[str], model_genes: list[str]
) -> np.ndarray:
    """
    Create inflation index mapping from sample genes to model genes.

    Python port of worker.ts precomputeInflationIndices().

    Args:
        sample_genes: Gene names from h5ad file
        model_genes: Gene names from model genes.txt

    Returns:
        Array of shape [n_sample_genes] where:
        - inflation_indices[i] = j means sample gene i maps to model gene j
        - inflation_indices[i] = -1 means sample gene i not in model
    """
    # Create lookup dict for O(1) access
    model_gene_to_idx = {gene: idx for idx, gene in enumerate(model_genes)}

    inflation_indices = np.array(
        [model_gene_to_idx.get(gene, -1) for gene in sample_genes], dtype=np.int32
    )

    n_missing = np.sum(inflation_indices == -1)
    n_present = len(inflation_indices) - n_missing

    logger.info(
        "Gene mapping: %d/%d genes present in model", n_present, len(sample_genes)
    )
    logger.info("  Missing genes: %d", n_missing)

    assert n_present > 0, "No genes from sample found in model vocabulary!"

    return inflation_indices


def _inflate_batch_to_model_space(
    batch_data: np.ndarray, inflation_indices: np.ndarray, n_model_genes: int
) -> np.ndarray:
    """
    Inflate batch from sample gene space to model gene space.

    Python port of worker.ts fillBatchData().

    Args:
        batch_data: Expression data in sample gene space [batch_size, n_sample_genes]
        inflation_indices: Mapping from sample to model genes
        n_model_genes: Number of genes in model

    Returns:
        Expression data in model gene space [batch_size, n_model_genes]
    """
    batch_size = batch_data.shape[0]

    # Initialize with zeros
    inflated = np.zeros((batch_size, n_model_genes), dtype=np.float32)

    # Map non-zero values to correct positions
    for sample_idx in range(len(inflation_indices)):
        model_idx = inflation_indices[sample_idx]
        if model_idx != -1:
            inflated[:, model_idx] = batch_data[:, sample_idx]

    return inflated


def _extract_raw_counts_batch(
    adata: sc.AnnData, start_idx: int, batch_size: int
) -> np.ndarray:
    """
    Extract raw counts for a batch of cells, handling sparse/dense formats.

    Args:
        adata: AnnData object with raw counts
        start_idx: Starting cell index
        batch_size: Number of cells to extract

    Returns:
        Dense array [batch_size, n_genes] with raw counts
    """
    end_idx = min(start_idx + batch_size, adata.n_obs)

    # Extract batch (handles both sparse and dense)
    batch = adata.X[start_idx:end_idx]

    # Convert to dense if sparse
    if hasattr(batch, "toarray"):
        batch = batch.toarray()

    return batch.astype(np.float32)


def _compute_embeddings_onnx(
    adata: sc.AnnData,
    onnx_session: ort.InferenceSession,
    inflation_indices: np.ndarray,
    n_model_genes: int,
    batch_size: int = 1000,
) -> np.ndarray:
    """
    Compute embeddings using ONNX model with gene inflation.

    Args:
        adata: AnnData object with raw counts in .X
        onnx_session: ONNX Runtime session for model.onnx (combined model)
        inflation_indices: Gene mapping from sample to model space
        n_model_genes: Number of genes in model
        batch_size: Batch size for inference

    Returns:
        Embeddings array [n_cells, embedding_dim]
    """
    n_cells = adata.n_obs
    embeddings_list = []

    logger.info(
        "Computing embeddings for %d cells in batches of %d", n_cells, batch_size
    )

    # Get input/output names from ONNX model
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    for start_idx in tqdm(range(0, n_cells, batch_size), desc="Embedding batches"):
        # Extract raw counts batch
        batch_counts = _extract_raw_counts_batch(adata, start_idx, batch_size)

        # Inflate to model gene space
        batch_inflated = _inflate_batch_to_model_space(
            batch_counts, inflation_indices, n_model_genes
        )

        # Run ONNX inference
        # Note: model.onnx does preprocessing internally (normalize + log1p)
        batch_embeddings = onnx_session.run(
            [output_name], {input_name: batch_inflated}
        )[0]

        embeddings_list.append(batch_embeddings)

    # Concatenate all batches
    embeddings: np.ndarray = np.concatenate(
        [np.asarray(e) for e in embeddings_list], axis=0
    )

    logger.info("Generated embeddings: %s", embeddings.shape)
    return embeddings.astype(np.float32)


# Create Typer app
app = typer.Typer(
    help="Ingest h5ad file as reference by processing through SCimilarity",
    add_completion=False,
)


def _validate_outputs(
    h5ad_path: Union[str, Path],
    output_dir: Path,
    onnx_model_path: Path,
    genes_path: Path,
    n_samples: int,
    batch_size: int,
    max_cells: int | None = None,
    seed: int = 42,
) -> None:
    """
    Validate that output files maintain correct row ordering and embeddings match.

    This function:
    1. Loads h5ad file and applies same sampling as during export (if any)
    2. Verifies labels match in labels.parquet
    3. Verifies embeddings match when regenerated through ONNX model

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Directory containing embeddings.npy and labels.parquet
        onnx_model_path: Path to ONNX model
        genes_path: Path to genes.txt
        n_samples: Number of random samples to validate
        batch_size: Batch size for ONNX inference
        max_cells: If specified, must match the sampling used during export
        seed: Random seed (must match the seed used during export)
    """

    logger.info("Validating output files against h5ad: %s", h5ad_path)

    # Load the h5ad file (supports S3)
    logger.info("Loading h5ad file...")
    adata = _load_h5ad_from_path_or_s3(h5ad_path)
    logger.info("  Original shape: %s", adata.shape)

    # Validate and convert to CSR (same as during export)
    adata = _validate_and_convert_to_csr(adata)

    # Apply the SAME sampling as was used during export
    if max_cells is not None and adata.n_obs > max_cells:
        logger.info("Applying same sampling as export (max_cells=%d, seed=%d)", max_cells, seed)
        adata = _random_sample_cells(adata, max_cells, seed)
        logger.info("  Sampled shape: %s", adata.shape)

    # Load output files
    embeddings_path = output_dir / "embeddings.npy"
    labels_path = output_dir / "labels.parquet"

    assert embeddings_path.exists(), f"Embeddings file not found: {embeddings_path}"

    logger.info("Loading embeddings from: %s", embeddings_path)
    embeddings_array = np.load(embeddings_path)
    logger.info("  Embeddings shape: %s", embeddings_array.shape)

    # Check if labels file exists
    has_labels = labels_path.exists()
    if has_labels:
        logger.info("Loading labels from: %s", labels_path)
        labels_df = pd.read_parquet(labels_path)
        logger.info("  Labels shape: %s", labels_df.shape)
        logger.info("  Label columns: %s", list(labels_df.columns))

    # Select random indices to validate
    n_cells = min(adata.n_obs, embeddings_array.shape[0])
    n_samples = min(n_samples, n_cells)
    random_indices = np.random.choice(n_cells, size=n_samples, replace=False)
    random_indices.sort()

    logger.info("Validating %d random samples: %s", n_samples, random_indices)

    # Validate labels (if present)
    if has_labels:
        logger.info("=== Validating Labels ===")
        all_labels_match = True

        for idx in random_indices:
            logger.info("Row %d:", idx)

            # Check each label column
            for col in labels_df.columns:
                if col in adata.obs.columns:
                    h5ad_value = adata.obs.iloc[idx][col]
                    parquet_value = labels_df.iloc[idx][col]

                    match = str(h5ad_value) == str(parquet_value)
                    status = "✅" if match else "❌"
                    logger.info(
                        "  %s: %s h5ad='%s' parquet='%s'",
                        col,
                        status,
                        h5ad_value,
                        parquet_value,
                    )

                    if not match:
                        all_labels_match = False

        if all_labels_match:
            logger.info("✅ All labels match!")
        else:
            logger.info("❌ Some labels do not match!")

    # Validate embeddings
    logger.info("=== Validating Embeddings ===")

    # Load ONNX model
    logger.info("Loading ONNX model from: %s", onnx_model_path)
    onnx_session = ort.InferenceSession(str(onnx_model_path))

    logger.info("Loading model genes from: %s", genes_path)
    model_genes = _load_model_genes(genes_path)

    # Process subset through ONNX
    logger.info("Processing subset through ONNX model...")
    subset_adata = adata[random_indices, :].copy()

    # Create gene inflation mapping
    inflation_indices = _create_inflation_indices(
        subset_adata.var_names.tolist(), model_genes
    )

    # Generate embeddings via ONNX
    subset_embeddings = _compute_embeddings_onnx(
        subset_adata, onnx_session, inflation_indices, len(model_genes), batch_size
    )

    # Compare embeddings
    all_embeddings_match = True
    tolerance = 1e-5

    for i, idx in enumerate(random_indices):
        regenerated = subset_embeddings[i]
        stored = embeddings_array[idx]

        # Calculate max absolute difference
        max_diff = np.max(np.abs(regenerated - stored))
        match = max_diff < tolerance

        status = "✅" if match else "❌"
        logger.info("Row %d: %s max_diff=%.2e", idx, status, max_diff)

        if not match:
            all_embeddings_match = False
            # Show first few values for debugging
            logger.info("  Regenerated[:5]: %s", regenerated[:5])
            logger.info("  Stored[:5]: %s", stored[:5])

    if all_embeddings_match:
        logger.info("✅ All embeddings match within tolerance (%s)!", tolerance)
    else:
        logger.info("❌ Some embeddings do not match within tolerance (%s)!", tolerance)

    # Summary
    logger.info("=== Validation Summary ===")
    if has_labels:
        logger.info("Labels: %s", "✅ PASS" if all_labels_match else "❌ FAIL")
    logger.info("Embeddings: %s", "✅ PASS" if all_embeddings_match else "❌ FAIL")

    if (not has_labels or all_labels_match) and all_embeddings_match:
        logger.info("✅ Validation PASSED!")
    else:
        logger.info("❌ Validation FAILED!")
        raise typer.Exit(1)


def _compute_label_intersection(
    h5ad_paths: list[Union[str, Path]], requested_labels: list[str]
) -> list[str]:
    """
    Compute intersection of label columns across multiple h5ad files.

    Args:
        h5ad_paths: List of paths to h5ad files
        requested_labels: User-requested label columns

    Returns:
        List of label columns present in ALL files
    """
    if not h5ad_paths:
        return []

    logger.info("Computing label intersection across %d files...", len(h5ad_paths))

    # Load obs columns from each file
    all_obs_columns = []
    for h5ad_path in h5ad_paths:
        adata = _load_h5ad_from_path_or_s3(h5ad_path)
        obs_columns = set(adata.obs.columns)
        all_obs_columns.append(obs_columns)
        logger.info("  %s: %d obs columns", Path(str(h5ad_path)).name, len(obs_columns))

    # Find intersection
    common_columns = set.intersection(*all_obs_columns)
    logger.info("  Common columns across all files: %d", len(common_columns))

    # Intersect with requested labels
    valid_labels = [label for label in requested_labels if label in common_columns]

    # Warn about missing labels
    for label in requested_labels:
        if label not in common_columns:
            logger.warning("Label '%s' not found in all files, will be excluded", label)

    logger.info("  ✅ Valid labels across all files: %s", valid_labels)
    return valid_labels


def _process_single_h5ad_file(
    h5ad_path: Union[str, Path],
    onnx_session: ort.InferenceSession,
    model_genes: list[str],
    labels: list[str],
    max_cells: int | None,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """
    Process a single h5ad file to generate embeddings and extract labels.

    Args:
        h5ad_path: Path or S3 URI to h5ad file
        onnx_session: ONNX Runtime inference session
        model_genes: List of model gene names
        labels: Label columns to extract
        max_cells: Maximum cells to sample (None for all)
        seed: Random seed for sampling
        batch_size: Batch size for embedding inference

    Returns:
        Tuple of (embeddings array, labels dataframe or None)
    """
    # Load h5ad file
    adata = _load_h5ad_from_path_or_s3(h5ad_path)
    logger.info("  Loaded data shape: %s", adata.shape)

    # Validate and convert to CSR
    adata = _validate_and_convert_to_csr(adata)

    # Sample cells if needed
    if max_cells is not None and adata.n_obs > max_cells:
        adata = _random_sample_cells(adata, max_cells, seed)
        logger.info("  Sampled data shape: %s", adata.shape)

    # Create gene inflation mapping
    inflation_indices = _create_inflation_indices(adata.var_names.tolist(), model_genes)

    # Compute embeddings
    embeddings = _compute_embeddings_onnx(
        adata, onnx_session, inflation_indices, len(model_genes), batch_size
    )

    # Extract labels if specified
    labels_df = None
    if labels:
        labels_df = pd.DataFrame()
        for label in labels:
            if label in adata.obs.columns:
                labels_df[label] = adata.obs[label].values
                labels_df[label] = labels_df[label].astype("category")

    return embeddings, labels_df


def _process_multiple_h5ad_files(
    h5ad_paths: list[Union[str, Path]],
    onnx_session: ort.InferenceSession,
    model_genes: list[str],
    requested_labels: list[str],
    max_cells: int | None,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """
    Process multiple h5ad files with label intersection.

    Args:
        h5ad_paths: List of paths or S3 URIs to h5ad files
        onnx_session: ONNX Runtime inference session
        model_genes: List of model gene names
        requested_labels: User-requested label columns
        max_cells: Maximum cells per file (None for all)
        seed: Base random seed
        batch_size: Batch size for embedding inference

    Returns:
        Tuple of (concatenated embeddings, concatenated labels or None)
    """
    # Compute label intersection
    valid_labels = _compute_label_intersection(h5ad_paths, requested_labels)

    # Process each file
    all_embeddings: list[np.ndarray] = []
    all_labels: list[pd.DataFrame] | None = [] if valid_labels else None

    for i, h5ad_path in enumerate(h5ad_paths):
        logger.info("Processing file %d/%d: %s", i + 1, len(h5ad_paths), h5ad_path)

        # Use per-file seed for reproducibility
        file_seed = seed + i

        embeddings, labels_df = _process_single_h5ad_file(
            h5ad_path=h5ad_path,
            onnx_session=onnx_session,
            model_genes=model_genes,
            labels=valid_labels,
            max_cells=max_cells,
            seed=file_seed,
            batch_size=batch_size,
        )

        all_embeddings.append(embeddings)
        if all_labels is not None and labels_df is not None:
            all_labels.append(labels_df)

        logger.info("  ✅ File %d complete: %d cells processed", i + 1, len(embeddings))

    # Concatenate results
    logger.info("Concatenating results from %d files...", len(h5ad_paths))
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    final_labels = None
    if all_labels:
        final_labels = pd.concat(all_labels, axis=0, ignore_index=True)

    logger.info("  Total cells: %d", len(final_embeddings))

    return final_embeddings, final_labels


@app.command()
def ingest(
    h5ad_paths: list[str] = typer.Argument(
        help="Path(s) or S3 URI(s) to h5ad file(s). Multiple files will be processed and concatenated.",
    ),
    onnx_model_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to ONNX embedding model file (model.onnx)",
    ),
    genes_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to model genes vocabulary file (genes.txt)",
    ),
    output_path: Path = typer.Argument(
        file_okay=False,
        dir_okay=True,
        help="Output directory path",
    ),
    labels: list[str] = typer.Option(
        [],
        help="Label columns from adata.obs to extract (e.g., --labels cell_type --labels tissue)",
    ),
    max_cells: int = typer.Option(
        None,
        help="Maximum number of cells per file to sample (default: use all cells)",
    ),
    validate: bool = typer.Option(
        True,
        help="Validate the output files after processing (default: True)",
    ),
    seed: int = typer.Option(
        42,
        help="Random seed for reproducibility",
    ),
    batch_size: int = typer.Option(
        1000,
        help="Batch size for embedding inference",
    ),
) -> None:
    """
    Process h5ad file(s) through ONNX embedding model to generate embeddings and extract labels.

    Supports both local files and S3 URIs (s3://bucket/key). Multiple files will be processed
    with label intersection (only labels present in ALL files are extracted).

    This script:
    1. Loads h5ad file(s) (from local path or S3)
    2. Computes label intersection for multi-file processing
    3. Validates/converts to CSR format for efficient processing
    4. Optionally samples random cells with seed
    5. Processes raw counts through ONNX embedding model
    6. Extracts specified label columns from adata.obs
    7. Saves embeddings to embeddings.npy and labels to labels.parquet
    """
    # Load ONNX model and genes (needed for both single and multi-file)
    logger.info("Loading ONNX model from: %s", onnx_model_path)
    onnx_session = ort.InferenceSession(str(onnx_model_path))

    logger.info("Loading model genes from: %s", genes_path)
    model_genes = _load_model_genes(genes_path)

    # Dispatch to single or multi-file processing
    if len(h5ad_paths) == 1:
        logger.info("Processing single h5ad file: %s", h5ad_paths[0])
        embeddings, labels_df = _process_single_h5ad_file(
            h5ad_path=h5ad_paths[0],
            onnx_session=onnx_session,
            model_genes=model_genes,
            labels=labels,
            max_cells=max_cells,
            seed=seed,
            batch_size=batch_size,
        )
    else:
        logger.info("Processing %d h5ad files...", len(h5ad_paths))
        embeddings, labels_df = _process_multiple_h5ad_files(
            h5ad_paths=h5ad_paths,
            onnx_session=onnx_session,
            model_genes=model_genes,
            requested_labels=labels,
            max_cells=max_cells,
            seed=seed,
            batch_size=batch_size,
        )

    # Create output directory
    logger.info("Creating output directory: %s", output_path)
    os.makedirs(output_path, exist_ok=True)

    # Save embeddings
    embeddings_path = output_path / "embeddings.npy"
    logger.info("Saving embeddings to: %s", embeddings_path)
    np.save(embeddings_path, embeddings.astype(np.float32))

    # Save labels if available
    labels_path = None
    if labels_df is not None:
        labels_path = output_path / "labels.parquet"
        logger.info("Saving labels to: %s", labels_path)
        labels_df.to_parquet(labels_path, compression="snappy", index=None)
        logger.info("Labels saved with shape: %s", labels_df.shape)

    logger.info("✅ Processing complete!")
    logger.info("   Embeddings: %s (shape: %s)", embeddings_path, embeddings.shape)
    if labels_path:
        logger.info("   Labels: %s (shape: %s)", labels_path, labels_df.shape)
    logger.info("   Output directory: %s", output_path)

    # Run validation if requested (only for single file)
    if validate and len(h5ad_paths) == 1:
        logger.info("=" * 60)
        logger.info("Running validation...")
        logger.info("=" * 60)
        try:
            _validate_outputs(
                h5ad_path=h5ad_paths[0],
                output_dir=output_path,
                onnx_model_path=onnx_model_path,
                genes_path=genes_path,
                n_samples=10,
                batch_size=batch_size,
                max_cells=max_cells,
                seed=seed,
            )
        except Exception as e:
            logger.error("Validation failed: %s", e)
            raise typer.Exit(1)
    elif validate and len(h5ad_paths) > 1:
        logger.info("Validation skipped for multi-file processing")


if __name__ == "__main__":
    app()
