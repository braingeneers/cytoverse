#!/usr/bin/env python3

import os
import typer
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import scanpy as sc
from typing import List

import scimilarity
from scimilarity import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts


# Create Typer app
app = typer.Typer(
    help="Ingest h5ad file as reference by processing through SCimilarity",
    add_completion=False,
)


@app.command()
def ingest(
    h5ad_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to h5ad file to process",
    ),
    model_path: Path = typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to SCimilarity model directory",
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
    max_labels_per_column: int = typer.Option(
        1000,
        help="Maximum number of unique labels allowed per column",
    ),
    max_cells: int = typer.Option(
        None,
        help="Maximum number of cells to export (default: export all cells)",
    ),
) -> None:
    """
    Process h5ad file through SCimilarity to generate embeddings and extract labels.

    This script:
    1. Loads the h5ad file
    2. Processes raw counts through SCimilarity (align_dataset, lognorm_counts, get_embeddings)
    3. Extracts specified label columns from adata.obs
    4. Saves embeddings to vectors.npy and labels to labels.parquet
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

            if unique_count > max_labels_per_column:
                print(
                    f"  ❌ Warning: Column '{label}' has {unique_count} unique values, exceeding limit of {max_labels_per_column}. Skipping."
                )
                continue

            valid_labels.append(label)

        labels = valid_labels
        print(f"  ✅ Valid label columns: {labels}")

    if not labels:
        print(
            "  ⚠️ No valid label columns specified or found. Only embeddings will be exported."
        )

    # Load SCimilarity model
    print(f"\nLoading SCimilarity model from: {model_path}")
    ce = scimilarity.CellEmbedding(str(model_path))

    # Process through SCimilarity pipeline
    print("\nProcessing data through SCimilarity...")

    # Step 1: Align dataset
    print("  1. Aligning dataset...")
    aligned_adata = align_dataset(adata, ce.gene_order)
    print(f"     Aligned data shape: {aligned_adata.shape}")

    # Step 2.1: Check for raw counts
    if "counts" in adata.layers:
        print("  Counts found in adata.layers['counts']")
    elif "raw_counts" in adata.layers:
        print("  Raw counts found in adata.layers['raw_counts']")
        aligned_adata.layers["counts"] = aligned_adata.layers["raw_counts"]
    elif "raw" in adata.layers:
        print("  Raw counts found in adata.layers['raw']")
        aligned_adata.layers["counts"] = aligned_adata.layers["raw"]
    else:
        print("  No raw counts found in adata.layers. Using adata.X as counts.")
        aligned_adata.layers["counts"] = aligned_adata.X

    # Step 2.2: Log-normalize counts
    print("  2. Log-normalizing counts...")
    lognorm_adata = lognorm_counts(aligned_adata)
    print(f"     Log-normalized data shape: {lognorm_adata.shape}")

    # Step 3: Get embeddings
    print("  3. Generating embeddings...")
    embeddings = ce.get_embeddings(lognorm_adata.X)
    print(f"     Embeddings shape: {embeddings.shape}")

    # Create output directory
    print(f"\nCreating output directory: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Save embeddings to vectors.npy
    vectors_path = output_path / "vectors.npy"
    print(f"Saving embeddings to: {vectors_path}")
    np.save(vectors_path, embeddings.astype(np.float32))

    # Extract and save labels if specified
    if labels:
        print(f"Extracting labels: {labels}")

        # Create labels dataframe with cell indices
        labels_df = pd.DataFrame(index=range(len(adata)))
        labels_df.index = labels_df.index.astype("int32")

        # Add each label column
        for label in labels:
            labels_df[label] = adata.obs[label].values
            # Convert to categorical for efficient storage
            labels_df[label] = labels_df[label].astype("category")
            print(f"  Added '{label}': {labels_df[label].nunique()} unique values")

        # Save labels to parquet
        labels_path = output_path / "labels.parquet"
        print(f"Saving labels to: {labels_path}")
        labels_df.to_parquet(
            labels_path,
            compression="snappy",
            index=True,
        )

        print(f"Labels saved with shape: {labels_df.shape}")

    print(f"\n✅ Processing complete!")
    print(f"   Embeddings: {vectors_path} (shape: {embeddings.shape})")
    if labels:
        print(f"   Labels: {labels_path} (shape: {labels_df.shape})")
    print(f"   Output directory: {output_path}")


if __name__ == "__main__":
    app()
