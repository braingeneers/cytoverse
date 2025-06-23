#!/usr/bin/env python3

import click
from pathlib import Path
import pandas as pd
import numpy as np
import tiledb
import scimilarity
from sklearn.model_selection import StratifiedShuffleSplit


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="data/scimilarity/model_v1.1/cellsearch",
    help="Path to model directory",
)
@click.option(
    "--output_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="data/scimilarity",
    help="Output directory path",
)
@click.option("-c", "--count", type=int, default=1000, help="Number of cells to export")
def main(model_path: Path, output_path: Path, count: int) -> None:

    print("Opening metadata TileDB and stratified sampling cells...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
        # Read only required columns for stratification
        df_strat: pd.DataFrame = metadata_db.query(attrs=["study", "tissue"]).df[:]

        # Create stratification labels by combining study and tissue
        strata_labels: pd.Series = df_strat["study"] + "_" + df_strat["tissue"]

        # Use stratified sampling
        sss = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=42)
        sampled_indices, _ = next(sss.split(df_strat.index, strata_labels))

        # Create metadata DataFrame with sampled indices
        metadata_df: pd.DataFrame = df_strat.loc[sampled_indices].copy()
        metadata_df["study"] = metadata_df["study"].astype("category")
        metadata_df["tissue"] = metadata_df["tissue"].astype("category")

        # Export metadata
        metadata_df.to_parquet(output_path / "metadata.parquet")

    print("Exporting corresponding cell embeddings...")
    embeddings = scimilarity.utils.embedding_from_tiledb(
        sampled_indices, str(model_path / "cell_embedding")
    )
    np.save(output_path / "embeddings.npy", embeddings)

    print(f"Exported {len(sampled_indices)} cells to {output_path}")


if __name__ == "__main__":
    main()
