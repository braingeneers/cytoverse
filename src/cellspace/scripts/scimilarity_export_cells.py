#!/usr/bin/env python3

import typer
from pathlib import Path
import pandas as pd
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
        "data/scimilarity",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Output directory path",
    ),
    field: str = typer.Option(
        "prediction",
        help="Field to use for stratification (e.g., 'tissue', 'study')",
    ),
    count: int = typer.Option(250, "-c", "--count", help="Number of cells to export"),
    export_h5ad: bool = typer.Option(
        False, "--export_h5ad", help="Export an h5ad file"
    ),
) -> None:

    print("Opening metadata TileDB and stratified sampling cells...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:

        fields = [f.name for f in metadata_db.schema]
        print("Meta data fieilds:", fields)

        # Read only required columns for stratification
        df_strat: pd.DataFrame = metadata_db.query(attrs=[field]).df[:]

        # Create stratification labels by combining study and tissue
        # strata_labels: pd.Series = df_strat["study"] + "_" + df_strat["tissue"]
        strata_labels: pd.Series = df_strat[field]

        # Use stratified sampling
        sss = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=42)
        sampled_indices, _ = next(sss.split(df_strat.index, strata_labels))

        # Create metadata DataFrame with sampled indices
        metadata_df: pd.DataFrame = df_strat.loc[sampled_indices].copy()
        # metadata_df["study"] = metadata_df["study"].astype("category")
        metadata_df[field] = metadata_df[field].astype("category")

        # Export metadata
        metadata_df.to_parquet(output_path / "metadata.parquet")

    if export_h5ad:
        print("Exporting h5ad with all of the cells expression...")
        adata = scimilarity.utils.adata_from_tiledb(
            sampled_indices[0:2],
            str(model_path / "cell_expression"),
        )
        adata.write(output_path / "expression.h5ad")

    print("Exporting corresponding cell embeddings...")
    embeddings = scimilarity.utils.embedding_from_tiledb(
        sampled_indices, str(model_path / "cell_embedding")
    )
    np.save(output_path / "embeddings.npy", embeddings)

    print(f"Exported {len(sampled_indices)} cells to {output_path}")


if __name__ == "__main__":
    typer.run(main)
