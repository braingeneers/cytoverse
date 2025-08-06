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
    help="Export SCimilarity embeddings and labels",
    add_completion=False,
)


@app.command()
def embeddings(
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
        ["prediction", "tissue"],
        help="Labels to export",
    ),
    num_embeddings: int = typer.Option(
        None,
        "--num-embeddings",
        help="Number of embeddings to export (default: export all embeddings)",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Validate exported files by checking 10 random samples against TileDB",
    ),
) -> None:
    """
    Export embeddings and optionally associated expression data from a SCimilarity model.

    Export behavior:
    - If --num-embeddings is not specified: export all cells
    - If --num-embeddings is specified: export first N cells (indices 0 to N-1)
    """

    print("Opening metadata TileDB...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:

        # Get all available schema fields for reference
        schema_fields = [f.name for f in metadata_db.schema]
        print("Available fields:", schema_fields)

        # Get total number of cells without loading data
        total_cells = (
            metadata_db.nonempty_domain()[0][1] + 1
        )  # TileDB uses 0-based indexing
        print(f"Total cells in TileDB: {total_cells}")

        # Step 1: Determine how many embeddings to export
        if num_embeddings is None:
            # Export all cells
            print("Exporting ALL cells...")
            num_to_export = total_cells
        else:
            # Export first num_embeddings cells (indices 0 to num_embeddings-1)
            print(f"Exporting first {num_embeddings} embeddings...")
            num_to_export = min(num_embeddings, total_cells)

    print(f"Outputting {num_to_export} embeddings to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    # Step 2: Load metadata and embeddings for first num_to_export indices
    print("Loading metadata...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
        metadata_df = metadata_db.query(attrs=labels).df[0 : num_to_export - 1]
    print(f"  Loaded metadata shape: {metadata_df.shape}")

    # Ensure we have exactly the number of rows we expect
    if len(metadata_df) > num_to_export:
        metadata_df = metadata_df.iloc[:num_to_export]
        print(f"  Trimmed metadata to {num_to_export} rows")

    for label in labels:
        metadata_df[label] = metadata_df[label].astype("category")

    # Export metadata as Parquet file with compression
    labels_path = output_path / "labels.parquet"
    metadata_df.to_parquet(
        labels_path,
        compression="snappy",
        index=None,
    )

    print("Loading embeddings...")

    # Get embedding dimensions by loading a small sample first using direct TileDB
    with tiledb.open(str(model_path / "cell_embedding"), "r") as embedding_array:
        sample_result = embedding_array[0:1]
        if sample_result is None or "vals" not in sample_result:
            raise ValueError("Could not determine embedding dimensions")
        embedding_dim = sample_result["vals"].shape[1]
        print(f"  Embedding dimensions: {embedding_dim}")

    # Create proper numpy file with header
    embeddings_path = output_path / "embeddings.npy"

    # Create full-size array and save it to establish proper .npy format
    print(f"Creating embeddings file with shape ({num_to_export}, {embedding_dim})")
    full_array = np.empty((num_to_export, embedding_dim), dtype=np.float32)
    np.save(embeddings_path, full_array)
    del full_array

    # Now open as memory-mapped for writing
    embeddings_mmap = np.load(embeddings_path, mmap_mode="r+")

    # Load embeddings in batches using direct TileDB access
    batch_size = 100_000
    num_batches = (num_to_export + batch_size - 1) // batch_size
    successful_count = 0

    try:
        with tiledb.open(str(model_path / "cell_embedding"), "r") as embedding_array:
            with tqdm(
                total=num_to_export, desc="Loading embeddings", unit="cells"
            ) as pbar:
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
                            batch_size_actual = embeddings.shape[0]

                            # Write batch to memory-mapped array at the correct position
                            embeddings_mmap[
                                start_idx : start_idx + batch_size_actual
                            ] = embeddings
                            successful_count += batch_size_actual

                        # Update progress
                        pbar.update(end_idx - start_idx)

                    except Exception as e:
                        print(f"❌ Error loading batch {start_idx}:{end_idx} - {e}")
                        # Continue with next batch
                        pbar.update(end_idx - start_idx)

        # Flush all changes to disk
        embeddings_mmap.flush()
        print(f"  Successfully loaded embeddings: {successful_count}/{num_to_export}")

    finally:
        # Clean up memory map
        del embeddings_mmap

    print(f"Labels saved to {labels_path} (shape: {metadata_df.shape})")
    print(
        f"Embeddings saved to {embeddings_path} (shape: ({successful_count}, {embedding_dim}))"
    )
    print(
        f"Exported indices: 0 to {num_to_export-1} (requested {num_to_export}, successful {successful_count})"
    )

    # Validation if requested
    if validate:
        print("\n🔍 Validating exported files...")
        # Select random sample for validation
        validation_count = min(10, num_to_export)
        validation_indices = np.random.choice(
            num_to_export, size=validation_count, replace=False
        ).tolist() + [
            0,
            num_to_export - 1,
        ]  # Include first and last indices
        validation_indices.sort()  # Sort for cleaner display
        print(
            f"  Validating random sample of {validation_count} indices: {validation_indices}"
        )

        _validate_exports(
            model_path,
            labels_path,
            embeddings_path,
            labels,
            validation_indices,
        )

    print(f"Exported {num_to_export} cells to {output_path}")


def _validate_exports(
    model_path: Path,
    labels_path: Path,
    embeddings_path: Path,
    labels: list[str],
    validation_indices: list[int],
) -> None:
    """Validate exported Parquet and numpy files against original TileDB data."""

    # Read exported files - select rows at validation_indices positions
    print("  Reading from exported files...")
    labels_df_exported = pd.read_parquet(labels_path)
    embeddings_exported = np.load(embeddings_path, mmap_mode="r")

    # Select the specific rows we want to validate
    labels_df_validation = labels_df_exported.iloc[validation_indices]
    embeddings_validation = embeddings_exported[validation_indices]

    # Get embedding IDs from the labels file index for the validation rows
    embedding_ids_exported = labels_df_validation.index.values

    # Read validation samples from original TileDB using the same indices
    print("  Reading from original TileDB...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
        original_metadata = metadata_db.query(attrs=["index"] + labels).df[
            validation_indices
        ]
        # Set the "index" column as the dataframe index with int32 dtype
        original_metadata = original_metadata.set_index("index", drop=True)

    original_embeddings = scimilarity.utils.embedding_from_tiledb(
        validation_indices, str(model_path / "cell_embedding")
    )

    # Validate labels using DataFrame comparison
    print("  Validating labels...")

    if labels_df_validation.astype(str).equals(original_metadata.astype(str)):
        labels_match = True
        print("    ✅ All label values match")
    else:
        labels_match = False
        print(f"    ❌ Label mismatch")

    # Validate embeddings
    print("  Validating embeddings...")
    embeddings_match = np.allclose(
        embeddings_validation,
        original_embeddings,
        rtol=1e-10,
        atol=1e-10,
    )

    if embeddings_match:
        print("    ✅ All embedding values match")
    else:
        max_diff = np.max(np.abs(embeddings_validation - original_embeddings))
        print(f"    ❌ Embedding mismatch: max diff = {max_diff:.2e}")

    # Validate embedding IDs
    print("  Validating embedding IDs...")
    embedding_ids_match = np.array_equal(embedding_ids_exported, validation_indices)

    if embedding_ids_match:
        print("    ✅ All embedding ID values match")
    else:
        print(f"    ❌ Embedding ID mismatch")

    if labels_match and embeddings_match and embedding_ids_match:
        print("  🎉 Validation passed! All exported data matches original TileDB.")
    else:
        print("  ⚠️ Validation found mismatches. Please check the export process.")


if __name__ == "__main__":
    app()
