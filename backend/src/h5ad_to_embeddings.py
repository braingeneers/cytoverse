#!/usr/bin/env python3

import os
import typer
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc

import onnxruntime as ort
from tqdm import tqdm


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

    print(f"Loaded {len(genes)} genes from {genes_path}")
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

    print(f"Gene mapping: {n_present}/{len(sample_genes)} genes present in model")
    print(f"  Missing genes: {n_missing}")

    if n_present == 0:
        raise ValueError("No genes from sample found in model vocabulary!")

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

    print(f"Computing embeddings for {n_cells} cells in batches of {batch_size}...")

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
        batch_embeddings = onnx_session.run([output_name], {input_name: batch_inflated})[
            0
        ]

        embeddings_list.append(batch_embeddings)

    # Concatenate all batches
    embeddings = np.concatenate(embeddings_list, axis=0)

    print(f"Generated embeddings: {embeddings.shape}")
    return embeddings.astype(np.float32)


# Create Typer app
app = typer.Typer(
    help="Ingest h5ad file as reference by processing through SCimilarity",
    add_completion=False,
)


def _validate_outputs(
    h5ad_path: Path,
    output_dir: Path,
    onnx_model_path: Path,
    genes_path: Path,
    n_samples: int,
    batch_size: int,
) -> None:
    """
    Validate that output files maintain correct row ordering and embeddings match.

    This function:
    1. Loads random rows from the original h5ad
    2. Verifies labels match at those indices in labels.parquet
    3. Verifies embeddings match when regenerated through ONNX model
    """

    print(f"Validating output files against h5ad: {h5ad_path}")

    # Load the h5ad file
    print("\nLoading h5ad file...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    # Load output files
    embeddings_path = output_dir / "embeddings.npy"
    labels_path = output_dir / "labels.parquet"

    if not embeddings_path.exists():
        print(f"❌ Error: {embeddings_path} not found")
        raise typer.Exit(1)

    print(f"\nLoading embeddings from: {embeddings_path}")
    embeddings_array = np.load(embeddings_path)
    print(f"  Embeddings shape: {embeddings_array.shape}")

    # Check if labels file exists
    has_labels = labels_path.exists()
    if has_labels:
        print(f"\nLoading labels from: {labels_path}")
        labels_df = pd.read_parquet(labels_path)
        print(f"  Labels shape: {labels_df.shape}")
        print(f"  Label columns: {list(labels_df.columns)}")

    # Select random indices to validate
    n_cells = min(adata.n_obs, embeddings_array.shape[0])
    n_samples = min(n_samples, n_cells)
    random_indices = np.random.choice(n_cells, size=n_samples, replace=False)
    random_indices.sort()

    print(f"\nValidating {n_samples} random samples: {random_indices}")

    # Validate labels (if present)
    if has_labels:
        print("\n=== Validating Labels ===")
        all_labels_match = True

        for idx in random_indices:
            print(f"\nRow {idx}:")

            # Check each label column
            for col in labels_df.columns:
                if col in adata.obs.columns:
                    h5ad_value = adata.obs.iloc[idx][col]
                    parquet_value = labels_df.iloc[idx][col]

                    match = str(h5ad_value) == str(parquet_value)
                    status = "✅" if match else "❌"
                    print(
                        f"  {col}: {status} h5ad='{h5ad_value}' parquet='{parquet_value}'"
                    )

                    if not match:
                        all_labels_match = False

        if all_labels_match:
            print("\n✅ All labels match!")
        else:
            print("\n❌ Some labels do not match!")

    # Validate embeddings
    print("\n=== Validating Embeddings ===")

    # Load ONNX model
    print(f"\nLoading ONNX model from: {onnx_model_path}")
    onnx_session = ort.InferenceSession(str(onnx_model_path))

    print(f"Loading model genes from: {genes_path}")
    model_genes = _load_model_genes(genes_path)

    # Process subset through ONNX
    print("\nProcessing subset through ONNX model...")
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
        print(f"\nRow {idx}: {status} max_diff={max_diff:.2e}")

        if not match:
            all_embeddings_match = False
            # Show first few values for debugging
            print(f"  Regenerated[:5]: {regenerated[:5]}")
            print(f"  Stored[:5]: {stored[:5]}")

    if all_embeddings_match:
        print(f"\n✅ All embeddings match within tolerance ({tolerance})!")
    else:
        print(f"\n❌ Some embeddings do not match within tolerance ({tolerance})!")

    # Summary
    print("\n=== Validation Summary ===")
    if has_labels:
        print(f"Labels: {'✅ PASS' if all_labels_match else '❌ FAIL'}")
    print(f"Embeddings: {'✅ PASS' if all_embeddings_match else '❌ FAIL'}")

    if (not has_labels or all_labels_match) and all_embeddings_match:
        print("\n✅ Validation PASSED!")
    else:
        print("\n❌ Validation FAILED!")
        raise typer.Exit(1)


@app.command()
def ingest(
    h5ad_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        default="fixtures/allen-celltypes+human-cortex+various-cortical-areas-1000.h5ad",
        help="Path to h5ad file to process",
    ),
    onnx_model_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        default="public/models/scimilarity/embedding/model.onnx",
        help="Path to ONNX embedding model file (model.onnx)",
    ),
    genes_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        default="public/models/scimilarity/embedding/genes.txt",
        help="Path to model genes vocabulary file (genes.txt)",
    ),
    output_path: Path = typer.Argument(
        file_okay=False,
        dir_okay=True,
        default="data/references/dev",
        help="Output directory path",
    ),
    labels: list[str] = typer.Option(
        ["cortical_layer_label", "CellType"],
        help="Label columns from adata.obs to extract (e.g., --labels cell_type --labels tissue)",
    ),
    max_cells: int = typer.Option(
        None,
        help="Maximum number of cells to export (default: export all cells)",
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
    Process h5ad file through SCimilarity to generate embeddings and extract labels.

    This script:
    1. Loads the h5ad file
    2. Processes raw counts through SCimilarity (align_dataset, lognorm_counts, get_embeddings)
    3. Extracts specified label columns from adata.obs
    4. Saves embeddings to embeddings.parquet and labels to labels.parquet
    """

    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded data shape: {adata.shape}")
    print(f"  Available obs columns: {list(adata.obs.columns)}")

    # Subset cells if max_cells is specified
    if max_cells is not None and adata.n_obs > max_cells:
        print(f"\nSubsetting to first {max_cells} cells (from {adata.n_obs} total)")
        adata = adata[:max_cells, :]
        print(f"  Subset data shape: {adata.shape}")

    # Validate label columns
    if labels:
        print(f"\nValidating label columns: {labels}")
        valid_labels = []
        for label in labels:
            if label not in adata.obs.columns:
                print(f"  ❌ Warning: Column '{label}' not found in adata.obs")
                continue

            unique_count = adata.obs[label].nunique()
            print(f"  Column '{label}': {unique_count} unique values")

            if unique_count > 1000:
                print(
                    f"  ❌ Warning: Column '{label}' has {unique_count} unique values, exceeding limit of 1000. Skipping."
                )
                continue

            valid_labels.append(label)

        labels = valid_labels
        print(f"  ✅ Valid label columns: {labels}")

    if not labels:
        print(
            "  ⚠️ No valid label columns specified or found. Only embeddings will be exported."
        )

    # Load ONNX model and genes
    print(f"\nLoading ONNX model from: {onnx_model_path}")
    onnx_session = ort.InferenceSession(str(onnx_model_path))

    print(f"Loading model genes from: {genes_path}")
    model_genes = _load_model_genes(genes_path)

    # Process through ONNX embedding pipeline
    print("\nProcessing data through ONNX embedding model...")

    # Step 1: Create gene inflation mapping
    print("  1. Creating gene inflation mapping...")
    inflation_indices = _create_inflation_indices(
        adata.var_names.tolist(), model_genes
    )

    # Step 2: Generate embeddings via ONNX
    # Note: ONNX model handles preprocessing internally (normalize + log1p)
    embeddings = _compute_embeddings_onnx(
        adata, onnx_session, inflation_indices, len(model_genes), batch_size
    )

    # Create output directory
    print(f"\nCreating output directory: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Save embeddings to embeddings.npy
    embeddings_path = output_path / "embeddings.npy"
    print(f"Saving embeddings to: {embeddings_path}")

    # Save embeddings as numpy array
    np.save(embeddings_path, embeddings.astype(np.float32))

    # Extract and save labels if specified
    if labels:
        print(f"Extracting labels: {labels}")

        # Create labels dataframe
        labels_df = pd.DataFrame()

        # Add each label column
        for label in labels:
            labels_df[label] = adata.obs[label].values
            # Convert to categorical for efficient storage
            labels_df[label] = labels_df[label].astype("category")
            print(f"  Added '{label}': {labels_df[label].nunique()} unique values")

        # Save labels to parquet without index
        labels_path = output_path / "labels.parquet"
        print(f"Saving labels to: {labels_path}")
        labels_df.to_parquet(
            labels_path,
            compression="snappy",
            index=None,
        )

        print(f"Labels saved with shape: {labels_df.shape}")

    print("\n✅ Processing complete!")
    print(f"   Embeddings: {embeddings_path} (shape: {embeddings.shape})")
    if labels:
        print(f"   Labels: {labels_path} (shape: {labels_df.shape})")
    print(f"   Output directory: {output_path}")

    # Run validation if requested
    if validate:
        print("\n" + "=" * 60)
        print("Running validation...")
        print("=" * 60)
        try:
            _validate_outputs(
                h5ad_path=h5ad_path,
                output_dir=output_path,
                onnx_model_path=onnx_model_path,
                genes_path=genes_path,
                n_samples=10,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
