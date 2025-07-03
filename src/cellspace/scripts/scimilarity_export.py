#!/usr/bin/env python3

import os
import typer
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet
import numpy as np
import tiledb
import scimilarity
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import onnx
import onnxruntime as ort
from scimilarity import CellEmbedding

# import cellxgene_census
import anndata as ad

# Create Typer app
app = typer.Typer(
    help="Export SCimilarity model, embeddings and labels",
    add_completion=False,
)


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
        opset_version=14,  # Use version 14 for better transformer support, 17 latest
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

    # Create CPU ONNX runtime session
    ort_session_cpu = ort.InferenceSession(str(embedder_path))

    # Try to create GPU ONNX runtime session
    gpu_available = False
    try:
        providers = ort.get_available_providers()
        print(f"  Available ONNX Runtime providers: {providers}")

        if "CUDAExecutionProvider" in providers:
            ort_session_gpu = ort.InferenceSession(
                str(embedder_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            gpu_available = True
            print("  GPU (CUDA) provider available for ONNX Runtime")
        else:
            print("  No compatible GPU providers available for high-dimensional input")
            if "CoreMLExecutionProvider" in providers:
                print(
                    "  ⚠️  Note: CoreML has input dimension limit of 16,384, but model requires 28,231 genes"
                )
            print("  Using CPU execution only")
    except Exception as e:
        print(f"  Could not create GPU session: {e}")
        gpu_available = False

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

    # Run through CPU ONNX model
    onnx_output_cpu = ort_session_cpu.run(None, {"input": test_input})[0]

    # Compare CPU ONNX with PyTorch
    max_diff_cpu = np.max(np.abs(torch_output - onnx_output_cpu))
    mean_diff_cpu = np.mean(np.abs(torch_output - onnx_output_cpu))

    print(f"  PyTorch vs CPU ONNX:")
    print(f"    Max absolute difference: {max_diff_cpu:.2e}")
    print(f"    Mean absolute difference: {mean_diff_cpu:.2e}")

    if max_diff_cpu < 1e-5:
        print("    ✅ Models are concordant (differences < 1e-5)")
    elif max_diff_cpu < 1e-3:
        print("    ⚠️ Models have small differences (< 1e-3)")
    else:
        print(
            f"    ❌ Models have significant differences (max diff: {max_diff_cpu:.2e})"
        )

    # Run through GPU ONNX model if available
    if gpu_available:
        onnx_output_gpu = ort_session_gpu.run(None, {"input": test_input})[0]

        # Compare GPU ONNX with PyTorch
        max_diff_gpu = np.max(np.abs(torch_output - onnx_output_gpu))
        mean_diff_gpu = np.mean(np.abs(torch_output - onnx_output_gpu))

        print(f"  PyTorch vs GPU ONNX:")
        print(f"    Max absolute difference: {max_diff_gpu:.2e}")
        print(f"    Mean absolute difference: {mean_diff_gpu:.2e}")

        if max_diff_gpu < 1e-5:
            print("    ✅ Models are concordant (differences < 1e-5)")
        elif max_diff_gpu < 1e-3:
            print("    ⚠️ Models have small differences (< 1e-3)")
        else:
            print(
                f"    ❌ Models have significant differences (max diff: {max_diff_gpu:.2e})"
            )

        # Compare CPU vs GPU ONNX
        max_diff_cpu_gpu = np.max(np.abs(onnx_output_cpu - onnx_output_gpu))
        mean_diff_cpu_gpu = np.mean(np.abs(onnx_output_cpu - onnx_output_gpu))

        print(f"  CPU ONNX vs GPU ONNX:")
        print(f"    Max absolute difference: {max_diff_cpu_gpu:.2e}")
        print(f"    Mean absolute difference: {mean_diff_cpu_gpu:.2e}")

        if max_diff_cpu_gpu < 1e-5:
            print("    ✅ CPU and GPU outputs are concordant (differences < 1e-5)")
        elif max_diff_cpu_gpu < 1e-3:
            print("    ⚠️ CPU and GPU outputs have small differences (< 1e-3)")
        else:
            print(
                f"    ❌ CPU and GPU outputs have significant differences (max diff: {max_diff_cpu_gpu:.2e})"
            )

    print("Exporting genes...")

    # Export genes file alongside the ONNX model
    # REMIND: Try exporting as gzip
    genes_path = output_path / "embedder.genes"
    with open(genes_path, "w") as f:
        f.write("\n".join(map(str, ce.gene_order)))

    print(f"Genes saved to {genes_path}")

    print("🎉 Export complete!")


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
        help="Labels to export (First used for stratification, others exported as well)",
    ),
    num_embeddings: int = typer.Option(
        None,
        "--num-embeddings",
        help="Number of embeddings to export (default: export all embeddings)",
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
            )
        else:
            # Stratified sampling for subset
            print(f"Stratified sampling {num_embeddings} embeddings...")
            print("Reading metadata for stratification...")

            # Read all rows for stratification (this is unavoidable for stratification)
            df_strat: pd.DataFrame = metadata_db.query(attrs=labels).df[:]

            # Stratify on the first label
            strata_labels: pd.Series = df_strat[labels[0]].astype("category")

            # Use stratified sampling
            print("Stratifying on:", labels[0])
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=num_embeddings, random_state=42
            )
            strat_sample_indices, _ = next(sss.split(df_strat.index, strata_labels))

            # Convert stratified sample indices back to original TileDB indices
            sampled_indices = df_strat.index[strat_sample_indices].tolist()

    sampled_indices.sort()  # TileDB requires sorted indices

    print(f"Outputting {len(sampled_indices)} embeddings to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    # Step 2: Load metadata and embeddings for sampled indices only
    print("Loading metadata...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
        metadata_df = metadata_db.query(attrs=labels).df[sampled_indices]
    print(f"  Loaded metadata shape: {metadata_df.shape}")

    print("Loading embeddings...")
    embeddings = scimilarity.utils.embedding_from_tiledb(
        sampled_indices, str(model_path / "cell_embedding")
    )
    print(f"  Loaded embeddings shape: {embeddings.shape}")

    for label in labels:
        metadata_df[label] = metadata_df[label].astype("category")

    # Export metadata as Parquet file with compression
    labels_path = output_path / "labels.parquet"
    table = pa.Table.from_pandas(metadata_df, preserve_index=False)
    pa.parquet.write_table(
        table,
        labels_path,
        compression="snappy",
        use_dictionary=True,  # Leverage categorical columns
        write_statistics=True,
    )

    # Export embeddings as numpy array
    embeddings_path = output_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    print(f"Labels saved to {labels_path} (shape: {metadata_df.shape})")
    print(f"Embeddings saved to {embeddings_path} (shape: {embeddings.shape})")

    # Validation if requested
    if validate:
        print("\n🔍 Validating exported files...")
        _validate_exports(
            model_path, labels_path, embeddings_path, labels, sampled_indices[0:10]
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

    # Read validation samples from original TileDB
    print("  Reading from original TileDB...")
    with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:
        original_metadata = metadata_db.query(attrs=labels).df[original_indices]

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

    if labels_match and embeddings_match:
        print("  🎉 Validation passed! All exported data matches original TileDB.")
    else:
        print("  ⚠️ Validation found mismatches. Please check the export process.")


# Not funcqtional yet, requires cellxgene-census package with dependency
# conflicts
# @app.command()
# def h5ad(
#     model_path: Path = typer.Argument(
#         exists=True,
#         file_okay=False,
#         dir_okay=True,
#         help="Path to SCimilarity directory (ex: model_v1.1/)",
#     ),
#     output_path: Path = typer.Argument(
#         exists=True,
#         file_okay=False,
#         dir_okay=True,
#         help="Directory path where labels.parquet and embeddings.parquet exist",
#     ),
#     num_cells: int = typer.Option(
#         2, "--num-cells", help="Number of cells to export from CELLxGENE Census"
#     ),
# ) -> None:
#     """
#     Extract expression data from CELLxGENE Census for cells in exported embeddings.
#     Requires: pip install cellxgene-census
#     """
#     print(f"Loading metadata from {output_path}...")

#     # Check if required files exist
#     labels_file = output_path / "labels.parquet"
#     embeddings_file = output_path / "embeddings.parquet"

#     if not labels_file.exists():
#         print(f"Error: {labels_file} not found. Run 'export' command first.")
#         return

#     if not embeddings_file.exists():
#         print(f"Error: {embeddings_file} not found. Run 'export' command first.")
#         return

#     # Load the exported labels to get cell indices and metadata
#     labels_df = pd.read_parquet(labels_file)
#     embeddings_df = pd.read_parquet(embeddings_file)

#     # Get the first num_cells from the embeddings (which has cell_id as index)
#     selected_cell_ids = embeddings_df.index[:num_cells].tolist()

#     print(f"Selected {len(selected_cell_ids)} cells to extract from Census...")
#     print(f"Cell IDs: {selected_cell_ids}")

#     # Open the SCimilarity TileDB to get study_accession and cell_id for each cell
#     print("Opening SCimilarity metadata TileDB...")
#     with tiledb.open(str(model_path / "cell_metadata"), "r") as metadata_db:

#         # Query for the specific cell indices we need
#         census_metadata = metadata_db.query(
#             attrs=["study", "sample"], coords=selected_cell_ids
#         ).df[:]

#     print("Census metadata for selected cells:")
#     print(census_metadata)

#     # Group cells by study for efficient Census queries
#     study_groups = census_metadata.groupby("study")

#     all_adatas = []

#     print("Connecting to CELLxGENE Census...")
#     try:
#         with cellxgene_census.open_soma(census_version="2025-01-30") as census:

#             for study, group in study_groups:
#                 print(f"Fetching data for study: {study}")
#                 cell_ids_for_study = group["cell_id"].tolist()

#                 try:
#                     # Query the Census for this study and these specific cells
#                     adata_study = cellxgene_census.get_anndata(
#                         census,
#                         organism="homo_sapiens",  # Assuming human data
#                         obs_value_filter=f"dataset_id == '{study}'",
#                         obs_coords=cell_ids_for_study,  # Filter to specific cells
#                         var_value_filter=None,  # Get all genes
#                     )

#                     if adata_study.n_obs > 0:
#                         # Add study information to obs
#                         adata_study.obs["study"] = study
#                         # Add original cell index for mapping back to embeddings
#                         adata_study.obs["original_cell_id"] = group.index
#                         all_adatas.append(adata_study)
#                         print(
#                             f"  Retrieved {adata_study.n_obs} cells, {adata_study.n_vars} genes"
#                         )
#                     else:
#                         print(f"  No cells found for study {study}")

#                 except Exception as e:
#                     print(f"  Error fetching data for study {study}: {e}")
#                     continue

#     except Exception as e:
#         print(f"Error connecting to CELLxGENE Census: {e}")
#         print(
#             "Make sure you have internet connection and the census package is properly installed."
#         )
#         return

#     if not all_adatas:
#         print("No expression data could be retrieved from Census")
#         return

#     # Concatenate all study data
#     print("Concatenating expression data from all studies...")
#     combined_adata = ad.concat(all_adatas, join="outer", fill_value=0)

#     # Ensure we have the right number of cells (or as many as we could get)
#     if combined_adata.n_obs > num_cells:
#         combined_adata = combined_adata[:num_cells]

#     # Save the combined expression data
#     output_file = output_path / "census_expression.h5ad"
#     combined_adata.write(output_file)

#     print(f"✅ Saved expression data for {combined_adata.n_obs} cells to {output_file}")
#     print(f"Expression matrix shape: {combined_adata.X.shape}")
#     print(f"Available studies: {combined_adata.obs['study_accession'].unique()}")

#     # Also save a mapping file for validation purposes
#     mapping_df = pd.DataFrame(
#         {
#             "cell_id": selected_cell_ids[: combined_adata.n_obs],
#             "census_obs_index": range(combined_adata.n_obs),
#         }
#     )
#     mapping_df.to_csv(output_path / "cell_mapping.csv", index=False)
#     print(f"Saved cell mapping to {output_path / 'cell_mapping.csv'}")


if __name__ == "__main__":
    app()
