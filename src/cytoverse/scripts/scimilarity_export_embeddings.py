#!/usr/bin/env python3

import os
import typer
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from typing import List

import tiledb
import scimilarity
from sklearn.model_selection import StratifiedShuffleSplit


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
        help="Labels to export (First used for stratification)",
    ),
    num_embeddings: int = typer.Option(
        None,
        "--num-embeddings",
        help="Number of embeddings to export (default: export all embeddings in TileDB order)",
    ),
    stratify: bool = typer.Option(
        False,
        "--stratify",
        help="Use stratified sampling based on first label (default: export in TileDB order)",
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
    - If --num-cells is not specified: export all cells in TileDB order
    - If --num-cells is specified without --stratify: export first N cells in TileDB order
    - If --num-cells is specified with --stratify: use stratified sampling based on first label
    """

    np.random.seed(42)  # For reproducibility

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

        # Step 1: Generate the list of indices to export
        if num_embeddings is None:
            # Export all cells in TileDB order
            print("Exporting ALL cells in TileDB order...")
            sampled_indices = list(range(total_cells))
        elif not stratify:
            # Export random num_embeddings in TileDB order (no stratification)
            print(f"Exporting random {num_embeddings} embeddings in TileDB order...")
            sampled_indices = np.random.choice(
                total_cells, size=num_embeddings, replace=False
            ).astype(np.int32)
        else:
            # Stratified sampling for subset
            print(f"Stratified sampling {num_embeddings} embeddings...")
            print("Reading metadata for stratification...")

            # Read all of the label we're stratifying on
            df_strat: pd.DataFrame = metadata_db.query(attrs=labels[0:1]).df[:]

            # Stratify on the first label
            strata_labels: pd.Series = df_strat[labels[0]].astype("category")

            # Use stratified sampling
            print("Stratifying on:", labels[0])
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=num_embeddings, random_state=42
            )
            strat_sample_indices, _ = next(sss.split(df_strat.index, strata_labels))

            # Convert stratified sample indices back to original TileDB indices
            sampled_indices = (
                df_strat.index[strat_sample_indices].astype(np.int32).values
            )

    sampled_indices.sort()  # TileDB requires sorted indices

    print(f"Outputting {len(sampled_indices)} embeddings to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    # Step 2: Load metadata and embeddings for sampled indices only
    print("Loading metadata...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
        metadata_df = metadata_db.query(attrs=["index"] + labels).df[
            sampled_indices
        ]
    print(f"  Loaded metadata shape: {metadata_df.shape}")

    # Set the "index" column as the dataframe index with int32 dtype
    metadata_df = metadata_df.set_index("index")
    metadata_df.index = metadata_df.index.astype("int32")

    for label in labels:
        metadata_df[label] = metadata_df[label].astype("category")

    # Export metadata as Parquet file with compression
    labels_path = output_path / "labels.parquet"
    metadata_df.to_parquet(
        labels_path,
        compression="snappy",
        index=True,
    )

    print("Loading embeddings...")
    embeddings = scimilarity.utils.embedding_from_tiledb(
        sampled_indices, str(model_path / "cell_embedding")
    )
    print(f"  Loaded embeddings shape: {embeddings.shape}")

    # Export embeddings as numpy array
    vectors_path = output_path / "vectors.npy"
    np.save(vectors_path, embeddings)

    print(f"Labels saved to {labels_path} (shape: {metadata_df.shape})")
    print(f"Vectors (Embeddings) saved to {vectors_path} (shape: {embeddings.shape})")
    print(f"Sampled indices: {sampled_indices[:10]}... (total {len(sampled_indices)})")

    # Validation if requested
    if validate:
        print("\n🔍 Validating exported files...")
        _validate_exports(
            model_path,
            labels_path,
            vectors_path,
            labels,
            sampled_indices[0:10],
        )

    print(f"Exported {len(sampled_indices)} cells to {output_path}")


def _validate_exports(
    model_path: Path,
    labels_path: Path,
    embeddings_path: Path,
    labels: list[str],
    original_indices: list[int],
) -> None:
    """Validate exported Parquet and numpy files against original TileDB data."""

    # Read exported files
    print("  Reading from exported files...")
    labels_df_exported = pd.read_parquet(labels_path)[: len(original_indices)]
    embeddings_exported = np.load(embeddings_path)[: len(original_indices)]
    # Get vector IDs from the labels file index
    vector_ids_exported = labels_df_exported.index.values[: len(original_indices)]

    # Read validation samples from original TileDB
    print("  Reading from original TileDB...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
        original_metadata = metadata_db.query(
            attrs=["index"] + labels
        ).df[original_indices]
        # Set the "index" column as the dataframe index with int32 dtype
        original_metadata = original_metadata.set_index("index")
        original_metadata.index = original_metadata.index.astype("int32")

    original_embeddings = scimilarity.utils.embedding_from_tiledb(
        original_indices, str(model_path / "cell_embedding")
    )

    # Validate labels using DataFrame comparison
    print("  Validating labels...")

    if labels_df_exported.astype(str).equals(original_metadata.astype(str)):
        labels_match = True
        print("    ✅ All label values match")
    else:
        labels_match = False
        print(f"    ❌ Label mismatch")

    # Validate embeddings
    print("  Validating embeddings...")
    embeddings_match = np.allclose(
        embeddings_exported,
        original_embeddings,
        rtol=1e-10,
        atol=1e-10,
    )

    if embeddings_match:
        print("    ✅ All embedding values match")
    else:
        max_diff = np.max(
            np.abs(embeddings_exported[: len(original_indices)] - original_embeddings)
        )
        print(f"    ❌ Embedding mismatch: max diff = {max_diff:.2e}")

    # Validate vector IDs
    print("  Validating vector IDs...")
    vector_ids_match = np.array_equal(vector_ids_exported, original_indices)

    if vector_ids_match:
        print("    ✅ All vector ID values match")
    else:
        print(f"    ❌ Vector ID mismatch")

    if labels_match and embeddings_match and vector_ids_match:
        print("  🎉 Validation passed! All exported data matches original TileDB.")
    else:
        print("  ⚠️ Validation found mismatches. Please check the export process.")


if __name__ == "__main__":
    app()
