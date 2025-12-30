#!/usr/bin/env python3

import os
import typer
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc

import scimilarity
from scimilarity import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts


# Create Typer app
app = typer.Typer(
    help="Ingest h5ad file as reference by processing through SCimilarity",
    add_completion=False,
)


def _validate_outputs(
    h5ad_path: Path,
    output_dir: Path,
    model_path: Path,
    n_samples: int,
    ce: CellEmbedding,
) -> None:
    """
    Validate that output files maintain correct row ordering and embeddings match.

    This function:
    1. Loads random rows from the original h5ad
    2. Verifies labels match at those indices in labels.parquet
    3. Verifies embeddings match when regenerated through SCimilarity
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

    # Load SCimilarity model if not provided
    if ce is None:
        print(f"\nLoading SCimilarity model from: {model_path}")
        ce = scimilarity.CellEmbedding(str(model_path))
    else:
        print("\nUsing provided SCimilarity model")

    # Process subset through SCimilarity
    print("\nProcessing subset through SCimilarity...")
    subset_adata = adata[random_indices, :].copy()

    # Align dataset
    aligned_subset = align_dataset(subset_adata, ce.gene_order)

    # Check for raw counts
    if "counts" in subset_adata.layers:
        aligned_subset.layers["counts"] = aligned_subset.layers["counts"]
    elif "raw_counts" in subset_adata.layers:
        aligned_subset.layers["counts"] = aligned_subset.layers["raw_counts"]
    elif "raw" in subset_adata.layers:
        aligned_subset.layers["counts"] = aligned_subset.layers["raw"]
    else:
        aligned_subset.layers["counts"] = aligned_subset.X

    # Log-normalize
    lognorm_subset = lognorm_counts(aligned_subset)

    # Generate embeddings
    subset_embeddings = ce.get_embeddings(lognorm_subset.X)

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
    model_path: Path = typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        default="data/models/scimilarity/model_v1.1",
        help="Path to SCimilarity model directory",
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
                model_path=model_path,
                n_samples=10,
                ce=ce,  # Pass the already loaded model
            )
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
