#!/usr/bin/env python3

import os
import typer
from pathlib import Path
import pandas as pd
import pyarrow.feather as feather
import numpy as np
import tiledb
import scimilarity
from sklearn.model_selection import StratifiedShuffleSplit


def main(
    model_path: Path = typer.Option(
        "data/scimilarity/model_v1.1/cellsearch",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to model directory",
    ),
    output_path: Path = typer.Option(
        "web/public/models/scimilarity",
        file_okay=False,
        dir_okay=True,
        help="Output directory path",
    ),
    labels: list[str] = typer.Option(
        ["tissue"],
        help="labels to use for stratification",
    ),
    count: int = typer.Option(1000, "-c", "--count", help="Number of cells to export"),
    export_h5ad: bool = typer.Option(
        False, "--export_h5ad", help="Export an h5ad file"
    ),
) -> None:

    np.random.seed(42)  # For reproducibility

    print("Opening metadata TileDB and stratified sampling cells...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:

        # Get all available schema fields for reference
        schema_fields = [f.name for f in metadata_db.schema]
        print("Available fields:", schema_fields)

        print("Reading metadata for stratification...")

        # Read all rows for stratification
        # df_strat: pd.DataFrame = metadata_db.query(attrs=labels).df[:]

        # Read a subset of rows to then stratify from
        df_strat: pd.DataFrame = metadata_db.query(attrs=labels).df[0:1000000]

        # Create compound label for stratification
        strata_labels: pd.Series = df_strat[labels].agg("|".join, axis=1)

        # Use stratified sampling
        print("Stratifying on:", labels)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=42)
        sampled_indices, _ = next(sss.split(df_strat.index, strata_labels))

    print(f"Outputing {count} cells to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    print("Exporting metadata for sampled cells...")
    metadata_df: pd.DataFrame = df_strat.loc[sampled_indices].copy()
    metadata_df[labels] = metadata_df[labels].astype("category")
    feather.write_feather(metadata_df, output_path / "labels.feather")

    print("Exporting corresponding cell embeddings...")
    embeddings = scimilarity.utils.embedding_from_tiledb(
        sampled_indices, str(model_path / "cell_embedding")
    )
    np.save(output_path / "embeddings.npy", embeddings)

    if export_h5ad:
        print("Exporting h5ad with all of the cells expression...")
        adata = scimilarity.utils.adata_from_tiledb(
            sampled_indices[0:2],
            str(model_path / "cell_expression"),
        )
        adata.write(output_path / "expression.h5ad")

    print(f"Exported {len(sampled_indices)} cells to {output_path}")


if __name__ == "__main__":
    typer.run(main)
