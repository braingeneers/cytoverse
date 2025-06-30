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

import torch
import onnx
import onnxruntime as ort
from scimilarity import CellEmbedding
import cellxgene_census
import anndata as ad

# Create Typer app
app = typer.Typer(
    help="Export embeddings, expression and embedding model from SCimilarity",
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
        ["prediction", "tissue", "study"],
        help="Labels to use for stratification (First used for stratification, others exported as well)",
    ),
    num_cells: int = typer.Option(
        1000, "--num-cells", help="Number of cells to export"
    ),
) -> None:
    """
    Export embeddings and optionally associated expression data from a SCimilarity model.
    """

    np.random.seed(42)  # For reproducibility

    print("Opening metadata TileDB and stratified sampling cells...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:

        # Get all available schema fields for reference
        schema_fields = [f.name for f in metadata_db.schema]
        print("Available fields:", schema_fields)

        print("Reading metadata for stratification...")

        # Read all rows for stratification
        df_strat: pd.DataFrame = metadata_db.query(attrs=labels).df[:]

        # Read a subset of rows to then stratify from
        # df_strat: pd.DataFrame = metadata_db.query(attrs=labels).df[0:1000000]

        # Create compound label for stratification
        # strata_labels: pd.Series = df_strat[labels].agg("|".join, axis=1)

        # Stratify on the first label
        strata_labels: pd.Series = df_strat[labels[0]].astype("category")

        # Use stratified sampling
        print("Stratifying on:", labels[0])
        sss = StratifiedShuffleSplit(n_splits=1, train_size=num_cells, random_state=42)
        sampled_indices, _ = next(sss.split(df_strat.index, strata_labels))

    print(f"Outputing {num_cells} cells to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    print("Exporting metadata for sampled cells...")
    metadata_df: pd.DataFrame = df_strat.loc[sampled_indices].copy()
    metadata_df[labels] = metadata_df[labels].astype("category")
    metadata_df.to_parquet(
        str(output_path / "labels.parquet"),
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    print("Exporting corresponding cell embeddings...")
    embeddings = scimilarity.utils.embedding_from_tiledb(
        sampled_indices, str(model_path / "cell_embedding")
    )

    # Create DataFrame with cell IDs as index and embeddings as columns
    embeddings_df = pd.DataFrame(
        embeddings,
        index=sampled_indices,
        columns=[f"embedding_{i}" for i in range(embeddings.shape[1])],
    )
    embeddings_df.index.name = "cell_id"

    embeddings_df.to_parquet(
        str(output_path / "embeddings.parquet"),
        engine="pyarrow",
        compression="snappy",
        index=True,
    )

    print(f"Exported {len(sampled_indices)} cells to {output_path}")


@app.command()
def model(
    model_path: Path = typer.Argument(
        exists=True,
        help="Path to scimilarity model checkpoint directory",
    ),
    output_path: Path = typer.Argument(
        help="Path to save the ONNX model",
    ),
) -> None:
    """
    Convert a scimilarity encoder model checkpoint to ONNX format.
    """
    print(f"Loading model from {model_path}...")
    ce = CellEmbedding(model_path)

    print("Converting to ONNX format...")
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedder_path = output_path / "embedder.onnx"
    torch.onnx.export(
        ce.model,
        torch.zeros(1, ce.n_genes),
        embedder_path,
        export_params=True,
        opset_version=14,  # Use version 14 for better transformer support
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,  # Reduce verbosity
    )

    print(f"ONNX model saved to {embedder_path}")

    print("Validating ONNX model...")
    onnx_model = onnx.load(str(embedder_path))
    onnx.checker.check_model(onnx_model)
    print("  ✅ ONNX model validation passed")

    print("Checking concordance with scimilarity model...")

    # Create ONNX runtime session
    ort_session = ort.InferenceSession(str(embedder_path))

    # Generate random test data (batch of 5 samples)
    batch_size = 5
    np.random.seed(42)  # For reproducible results
    test_input = np.random.lognormal(
        mean=0, sigma=1, size=(batch_size, ce.n_genes)
    ).astype(np.float32)

    # Run through original PyTorch model
    ce.model.eval()
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_output = ce.model(torch_input).numpy()

    # Run through ONNX model
    onnx_output = ort_session.run(None, {"input": test_input})[0]

    # Compare outputs
    max_diff = np.max(np.abs(torch_output - onnx_output))
    mean_diff = np.mean(np.abs(torch_output - onnx_output))

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✅ Models are concordant (differences < 1e-5)")
    elif max_diff < 1e-3:
        print("  ⚠️ Models have small differences (< 1e-3)")
    else:
        print(f"  ❌ Models have significant differences (max diff: {max_diff:.2e})")

    print("Exporting genes...")

    # Export genes file alongside the ONNX model
    genes_path = output_path / "embedder.genes"
    with open(genes_path, "w") as f:
        f.write("\n".join(map(str, ce.gene_order)))

    print(f"Genes saved to {genes_path}")

    print("🎉 Export complete!")


@app.command()
def h5ad(
    model_path: Path = typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to SCimilarity directory (ex: model_v1.1/)",
    ),
    output_path: Path = typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory path where labels.parquet and embeddings.parquet exist",
    ),
    num_cells: int = typer.Option(
        2, "--num-cells", help="Number of cells to export from CELLxGENE Census"
    ),
) -> None:
    """
    Extract expression data from CELLxGENE Census for cells in exported embeddings.
    Requires: pip install cellxgene-census
    """
    print(f"Loading metadata from {output_path}...")

    # Check if required files exist
    labels_file = output_path / "labels.parquet"
    embeddings_file = output_path / "embeddings.parquet"

    if not labels_file.exists():
        print(f"Error: {labels_file} not found. Run 'export' command first.")
        return

    if not embeddings_file.exists():
        print(f"Error: {embeddings_file} not found. Run 'export' command first.")
        return

    # Load the exported labels to get cell indices and metadata
    labels_df = pd.read_parquet(labels_file)
    embeddings_df = pd.read_parquet(embeddings_file)

    # Get the first num_cells from the embeddings (which has cell_id as index)
    selected_cell_ids = embeddings_df.index[:num_cells].tolist()

    print(f"Selected {len(selected_cell_ids)} cells to extract from Census...")
    print(f"Cell IDs: {selected_cell_ids}")

    # Open the SCimilarity TileDB to get study_accession and cell_id for each cell
    print("Opening SCimilarity metadata TileDB...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:

        # Query for the specific cell indices we need
        census_metadata = metadata_db.query(
            attrs=["study", "sample"], coords=selected_cell_ids
        ).df[:]

    print("Census metadata for selected cells:")
    print(census_metadata)

    # Group cells by study for efficient Census queries
    study_groups = census_metadata.groupby("study")

    all_adatas = []

    print("Connecting to CELLxGENE Census...")
    try:
        with cellxgene_census.open_soma(census_version="2025-01-30") as census:

            for study, group in study_groups:
                print(f"Fetching data for study: {study}")
                cell_ids_for_study = group["cell_id"].tolist()

                try:
                    # Query the Census for this study and these specific cells
                    adata_study = cellxgene_census.get_anndata(
                        census,
                        organism="homo_sapiens",  # Assuming human data
                        obs_value_filter=f"dataset_id == '{study}'",
                        obs_coords=cell_ids_for_study,  # Filter to specific cells
                        var_value_filter=None,  # Get all genes
                    )

                    if adata_study.n_obs > 0:
                        # Add study information to obs
                        adata_study.obs["study"] = study
                        # Add original cell index for mapping back to embeddings
                        adata_study.obs["original_cell_id"] = group.index
                        all_adatas.append(adata_study)
                        print(
                            f"  Retrieved {adata_study.n_obs} cells, {adata_study.n_vars} genes"
                        )
                    else:
                        print(f"  No cells found for study {study}")

                except Exception as e:
                    print(f"  Error fetching data for study {study}: {e}")
                    continue

    except Exception as e:
        print(f"Error connecting to CELLxGENE Census: {e}")
        print(
            "Make sure you have internet connection and the census package is properly installed."
        )
        return

    if not all_adatas:
        print("No expression data could be retrieved from Census")
        return

    # Concatenate all study data
    print("Concatenating expression data from all studies...")
    combined_adata = ad.concat(all_adatas, join="outer", fill_value=0)

    # Ensure we have the right number of cells (or as many as we could get)
    if combined_adata.n_obs > num_cells:
        combined_adata = combined_adata[:num_cells]

    # Save the combined expression data
    output_file = output_path / "census_expression.h5ad"
    combined_adata.write(output_file)

    print(f"✅ Saved expression data for {combined_adata.n_obs} cells to {output_file}")
    print(f"Expression matrix shape: {combined_adata.X.shape}")
    print(f"Available studies: {combined_adata.obs['study_accession'].unique()}")

    # Also save a mapping file for validation purposes
    mapping_df = pd.DataFrame(
        {
            "cell_id": selected_cell_ids[: combined_adata.n_obs],
            "census_obs_index": range(combined_adata.n_obs),
        }
    )
    mapping_df.to_csv(output_path / "cell_mapping.csv", index=False)
    print(f"Saved cell mapping to {output_path / 'cell_mapping.csv'}")


if __name__ == "__main__":
    app()
