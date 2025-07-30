#!/usr/bin/env python3

import os
import typer
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from typing import List

import tiledb
import scimilarity


# Create Typer app
app = typer.Typer(
    help="Export SCimilarity TileDB embeddings and labels to parquet format",
    add_completion=False,
)


@app.command()
def export(
    model_path: Path = typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to SCimilarity directory (ex: model_v1.1/)",
    ),
    output_path: Path = typer.Argument(
        file_okay=False,
        dir_okay=True,
        help="Output directory path",
    ),
    labels: list[str] = typer.Option(
        [],
        help="Label columns from metadata to extract (e.g., --labels prediction --labels tissue)",
    ),
    max_cells: int = typer.Option(
        None,
        help="Maximum number of cells to export (default: export all cells, None means no limit)",
    ),
    validate: bool = typer.Option(
        True,
        help="Validate the output files after processing (default: True)",
    ),
) -> None:
    """
    Export embeddings and labels from SCimilarity TileDB to parquet format.

    This script:
    1. Loads embeddings from TileDB
    2. Extracts specified label columns from cell_metadata
    3. Saves embeddings to embeddings.parquet and labels to labels.parquet
    """

    print("Opening metadata TileDB...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:

        # Get all available schema fields for reference
        schema_fields = [f.name for f in metadata_db.schema]
        available_labels = [f for f in schema_fields if f != "index"]
        print(f"  Available label columns: {available_labels}")

        # Get total number of cells without loading data
        total_cells = (
            metadata_db.nonempty_domain()[0][1] + 1
        )  # TileDB uses 0-based indexing
        print(f"Total cells in TileDB: {total_cells}")

    # Validate label columns
    if labels:
        print(f"\nValidating label columns: {labels}")
        valid_labels = []
        for label in labels:
            if label not in available_labels:
                print(f"  ❌ Warning: Column '{label}' not found in metadata")
                continue
            valid_labels.append(label)
        labels = valid_labels
        print(f"  ✅ Valid label columns: {labels}")

    if not labels:
        print(
            "  ⚠️ No valid label columns specified or found. Only embeddings will be exported."
        )

    # Determine how many cells to export
    if max_cells is None:
        print("\nExporting ALL cells...")
        num_to_export = total_cells
    else:
        print(f"\nSubsetting to first {max_cells} cells (from {total_cells} total)")
        num_to_export = min(max_cells, total_cells)

    # Create output directory
    print(f"\nCreating output directory: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Load embeddings
    print("\nLoading embeddings from TileDB...")
    embeddings_list = []
    batch_size = 100_000
    num_batches = (num_to_export + batch_size - 1) // batch_size

    with tiledb.open(str(model_path / "cell_embedding"), "r") as embedding_array:
        with tqdm(total=num_to_export, desc="Loading embeddings", unit="cells") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_to_export)

                try:
                    # Query the specific range using direct TileDB
                    embeddings_result = embedding_array[start_idx:end_idx]

                    # Extract the actual embeddings from OrderedDict
                    if embeddings_result is None or "vals" not in embeddings_result:
                        print(
                            f"Warning: No embeddings found for indices {start_idx} to {end_idx-1}"
                        )
                    else:
                        embeddings = embeddings_result["vals"]
                        embeddings_list.append(embeddings)

                    # Update progress
                    pbar.update(end_idx - start_idx)

                except Exception as e:
                    print(f"❌ Error loading batch {start_idx}:{end_idx} - {e}")
                    raise

    # Concatenate all embeddings
    embeddings = np.vstack(embeddings_list)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings to embeddings.parquet
    embeddings_path = output_path / "embeddings.parquet"
    print(f"Saving embeddings to: {embeddings_path}")

    # Convert embeddings to DataFrame
    embeddings_df = pd.DataFrame(embeddings.astype(np.float32))

    # Save to parquet without index
    embeddings_df.to_parquet(
        embeddings_path,
        compression=None,
        index=None,
    )

    # Extract and save labels if specified
    if labels:
        print(f"\nExtracting labels: {labels}")

        # Load metadata
        with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
            # Create labels dataframe
            labels_df = pd.DataFrame()

            # Load each label column
            for label in labels:
                print(f"  Loading '{label}'...")
                label_data = metadata_db.query(attrs=[label]).df[0 : num_to_export - 1]
                labels_df[label] = label_data[label].values

                # Check unique values
                unique_count = labels_df[label].nunique()
                print(f"  Column '{label}': {unique_count} unique values")

                if unique_count > 1000:
                    print(
                        f"  ❌ Warning: Column '{label}' has {unique_count} unique values, exceeding limit of 1000. Skipping."
                    )
                    labels_df = labels_df.drop(columns=[label])
                    labels.remove(label)
                    continue

                # Convert to categorical for efficient storage
                labels_df[label] = labels_df[label].astype("category")

        if labels:
            # Save labels to parquet without index
            labels_path = output_path / "labels.parquet"
            print(f"Saving labels to: {labels_path}")
            labels_df.to_parquet(
                labels_path,
                compression="snappy",
                index=None,
            )
            print(f"Labels saved with shape: {labels_df.shape}")

    print(f"\n✅ Processing complete!")
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
                model_path=model_path,
                parquet_dir=output_path,
                n_samples=10,
                max_cells=num_to_export,
            )
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            raise typer.Exit(1)


def _validate_outputs(
    model_path: Path,
    parquet_dir: Path,
    n_samples: int = 10,
    max_cells: int = None,
) -> None:
    """
    Validate that parquet files match data in TileDB.

    This function:
    1. Selects random indices from the exported data
    2. Verifies labels match at those indices between TileDB and labels.parquet
    3. Verifies embeddings match between TileDB and embeddings.parquet
    """

    print(f"Validating parquet files against TileDB: {model_path}")

    # Load parquet files
    embeddings_path = parquet_dir / "embeddings.parquet"
    labels_path = parquet_dir / "labels.parquet"

    if not embeddings_path.exists():
        print(f"❌ Error: {embeddings_path} not found")
        raise typer.Exit(1)

    print(f"\nLoading embeddings from: {embeddings_path}")
    embeddings_df = pd.read_parquet(embeddings_path)
    print(f"  Embeddings shape: {embeddings_df.shape}")

    # Check if labels file exists
    has_labels = labels_path.exists()
    if has_labels:
        print(f"\nLoading labels from: {labels_path}")
        labels_df = pd.read_parquet(labels_path)
        print(f"  Labels shape: {labels_df.shape}")
        print(f"  Label columns: {list(labels_df.columns)}")

    # Select random indices to validate
    n_cells = embeddings_df.shape[0]
    if max_cells:
        n_cells = min(n_cells, max_cells)
    n_samples = min(n_samples, n_cells)
    random_indices = np.random.choice(n_cells, size=n_samples, replace=False)
    random_indices.sort()

    print(f"\nValidating {n_samples} random samples: {random_indices}")

    # Validate labels (if present)
    if has_labels:
        print("\n=== Validating Labels ===")
        all_labels_match = True

        # Load original labels from TileDB for validation indices
        with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
            # Get label columns from parquet file
            label_cols = list(labels_df.columns)

            for idx in random_indices:
                print(f"\nRow {idx}:")

                # Load this specific row from TileDB
                tiledb_row = metadata_db.query(attrs=label_cols).df[idx : idx + 1]

                # Check each label column
                for col in label_cols:
                    tiledb_value = tiledb_row[col].values[0]
                    parquet_value = labels_df.iloc[idx][col]

                    match = str(tiledb_value) == str(parquet_value)
                    status = "✅" if match else "❌"
                    print(
                        f"  {col}: {status} tiledb='{tiledb_value}' parquet='{parquet_value}'"
                    )

                    if not match:
                        all_labels_match = False

        if all_labels_match:
            print("\n✅ All labels match!")
        else:
            print("\n❌ Some labels do not match!")

    # Validate embeddings
    print("\n=== Validating Embeddings ===")
    all_embeddings_match = True
    tolerance = 1e-5

    # Load embeddings from TileDB for validation indices
    original_embeddings = scimilarity.utils.embedding_from_tiledb(
        random_indices.tolist(), str(model_path / "cell_embedding")
    )

    for i, idx in enumerate(random_indices):
        tiledb_embedding = original_embeddings[i]
        parquet_embedding = embeddings_df.iloc[idx].values

        # Calculate max absolute difference
        max_diff = np.max(np.abs(tiledb_embedding - parquet_embedding))
        match = max_diff < tolerance

        status = "✅" if match else "❌"
        print(f"\nRow {idx}: {status} max_diff={max_diff:.2e}")

        if not match:
            all_embeddings_match = False
            # Show first few values for debugging
            print(f"  TileDB[:5]: {tiledb_embedding[:5]}")
            print(f"  Parquet[:5]: {parquet_embedding[:5]}")

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


if __name__ == "__main__":
    app()
