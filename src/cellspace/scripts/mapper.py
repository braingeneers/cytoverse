#!/usr/bin/env python

from pathlib import Path
from typing import Optional, Dict
import json

import numpy as np
import torch
import typer
import onnxruntime as ort
from umap_pytorch import PUMAP
import anndata as ad
import scanpy as sc
from tqdm import tqdm

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Create Typer app
app = typer.Typer(
    help="Train a parametric UMAP model from sample embeddings and export it to ONNX format, then map embeddings using the trained model.",
    add_completion=True,
)


@app.command()
def train(
    sample_embeddings_path: Path = typer.Argument(
        ..., help="Path to embeddings.npy file"
    ),
    output_path: Path = typer.Argument(..., help="Path to output the trained model"),
    num_embeddings: Optional[int] = typer.Option(
        None,
        help="Limit the total number of embeddings used for training, use all if None",
    ),
) -> None:
    """
    Train a parametric UMAP model from sample embeddings and export it to ONNX format.
    """
    typer.echo(
        f"Training parametric UMAP model from embeddings {sample_embeddings_path}"
    )

    # Load the embeddings
    embeddings = np.load(sample_embeddings_path)

    # Determine number of embeddings to use
    num_embeddings = (
        num_embeddings if num_embeddings is not None else embeddings.shape[0]
    )
    if num_embeddings is not None:
        num_embeddings = min(embeddings.shape[0], num_embeddings)

    embeddings = embeddings[:num_embeddings]

    # Create and fit the model
    model = PUMAP(
        encoder=None,  # nn.Module, None for default
        decoder=None,  # nn.Module, True for default, None for encoder only
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,  # How much to weigh reconstruction loss for decoder
        reconstruction_loss=torch.nn.functional.binary_cross_entropy_with_logits,  # pass in custom reconstruction loss functions
        random_state=None,
        lr=1e-3,
        epochs=10,
        num_workers=1,
        num_gpus=1,
        match_nonparametric_umap=False,  # Train network to match embeddings from non parametric umap
    )

    model.fit(torch.from_numpy(embeddings).to("mps"))

    # Export to ONNX
    model_output_path = output_path / "mapper.onnx"
    torch.onnx.export(
        model.model.encoder.encoder,
        torch.zeros(1, embeddings.shape[1]),
        model_output_path,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=12,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    typer.echo(f"Saved clustering model to {model_output_path}")


def save_mappings_png(output_path: Path, mappings: np.ndarray) -> None:
    """
    Save raw mappings as a PNG scatter plot for comparison.

    Args:
        output_path: Directory where to save the PNG file
        mappings: Raw float32 mappings array of shape (num_embeddings, 2)
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(mappings[:, 0], mappings[:, 1], alpha=0.6, s=1)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Raw Mappings Scatter Plot")
    plt.axis("equal")

    # Save the plot
    png_path = output_path / "mappings.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    typer.echo(f"Saved raw mappings plot to {png_path}")


@app.command()
def map(
    model_path: Path = typer.Argument(..., help="Path to mapper.onnx"),
    sample_embeddings_path: Path = typer.Argument(
        ..., help="Path to sample-embeddings.npy file"
    ),
    output_path: Path = typer.Argument(
        ..., help="Path to output directory for the mapped data"
    ),
    batch_size: int = typer.Option(
        32, help="Number of samples to process in each batch"
    ),
    export_png: bool = typer.Option(
        False, "--export-png", help="Export a reference PNG file"
    ),
    num_embeddings: Optional[int] = typer.Option(
        None, help="Limit the total number of inputs processed, process all if None"
    ),
) -> None:
    """
    Map embeddings to a lower-dimensional space using a trained ONNX model.
    """
    typer.echo(
        f"Mapping samples from {sample_embeddings_path} using model {model_path}"
    )

    # Load the cluster model
    onnx_session = ort.InferenceSession(str(model_path))
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Load the embeddings
    embeddings = np.load(sample_embeddings_path)

    # Determine number of embeddings to process
    num_embeddings = (
        num_embeddings if num_embeddings is not None else embeddings.shape[0]
    )
    if num_embeddings is not None:
        num_embeddings = min(embeddings.shape[0], num_embeddings)

    typer.echo(f"Processing {num_embeddings} embeddings with batch size {batch_size}")

    # Initialize list to store mappings
    all_mappings = []

    # Process in batches
    with tqdm(total=num_embeddings) as pbar:
        for batch_start in range(0, num_embeddings, batch_size):
            batch_end = min(batch_start + batch_size, num_embeddings)
            batch_size_actual = batch_end - batch_start

            # Get batch of embeddings
            batch_embeddings = embeddings[batch_start:batch_end]

            # Run the model
            model_input = {input_name: batch_embeddings.astype(np.float32)}
            batch_mappings = onnx_session.run([output_name], model_input)[0]

            # Store the mappings
            all_mappings.append(batch_mappings)

            pbar.update(batch_size_actual)

    # Combine all batches
    mappings = np.concatenate(all_mappings)  # Shape: (num_embeddings, 2)

    # Save raw mappings as a numpy file
    mappings_path = output_path / "mappings.npy"
    np.save(mappings_path, mappings)
    typer.echo(f"Saved raw float mappings to {mappings_path}")

    # Save raw mappings as PNG for comparison
    if export_png:
        save_mappings_png(output_path, mappings)


@app.command()
def export(
    input_path: Path = typer.Argument(
        ..., help="Path to input directory containing mappings.npy and labels.parquet"
    ),
    output_path: Path = typer.Argument(
        ..., help="Path to output directory for Arrow files and metadata"
    ),
) -> None:
    """
    Export mappings and labels to Arrow columnar format files.

    Reads mappings.npy and labels.parquet from input_path and outputs:
    - x.arrow and y.arrow: Coordinate columns in Arrow format
    - {column_name}.arrow: Category indices for each label column as int16
    - metadata.json: Contains categories mapping for label columns
    """
    typer.echo(f"Exporting data from {input_path} to {output_path}")

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Load mappings.npy
    mappings_file = input_path / "mappings.npy"
    if not mappings_file.exists():
        typer.echo(f"Error: {mappings_file} not found", err=True)
        raise typer.Exit(1)

    typer.echo("Loading mappings...")
    mappings = np.load(mappings_file)  # Shape: (num_embeddings, 2)

    # Normalize coordinates to [-1, 1] range while preserving proportions and centering
    typer.echo("Normalizing coordinates to [-1, 1] range with centering...")

    # Find min and max for each dimension separately
    x_min, x_max = np.min(mappings[:, 0]), np.max(mappings[:, 0])
    y_min, y_max = np.min(mappings[:, 1]), np.max(mappings[:, 1])

    # Calculate ranges for each dimension
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Use the larger range to preserve aspect ratio
    max_range = max(x_range, y_range)

    # Calculate centers for each dimension
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    # Normalize to [-1, 1] while preserving proportions and centering both dimensions
    if max_range > 0:
        normalized_mappings = np.zeros_like(mappings)
        normalized_mappings[:, 0] = (mappings[:, 0] - x_center) / (max_range / 2)
        normalized_mappings[:, 1] = (mappings[:, 1] - y_center) / (max_range / 2)
    else:
        # Handle edge case where all values are the same
        normalized_mappings = np.zeros_like(mappings)

    typer.echo(
        f"Original X range: [{x_min:.3f}, {x_max:.3f}], Y range: [{y_min:.3f}, {y_max:.3f}]"
    )
    typer.echo(
        f"Normalized X range: [{np.min(normalized_mappings[:, 0]):.3f}, {np.max(normalized_mappings[:, 0]):.3f}]"
    )
    typer.echo(
        f"Normalized Y range: [{np.min(normalized_mappings[:, 1]):.3f}, {np.max(normalized_mappings[:, 1]):.3f}]"
    )

    # Export x and y coordinates as separate Arrow files
    typer.echo("Exporting coordinate columns...")

    # X coordinates
    x_array = pa.array(normalized_mappings[:, 0], type=pa.float32())
    x_table = pa.table([x_array], names=["x"])
    x_path = output_path / "x.arrow"
    with pa.ipc.new_file(x_path, x_table.schema) as writer:
        writer.write_table(x_table)
    typer.echo(f"Saved X coordinates to {x_path}")

    # Y coordinates
    y_array = pa.array(normalized_mappings[:, 1], type=pa.float32())
    y_table = pa.table([y_array], names=["y"])
    y_path = output_path / "y.arrow"
    with pa.ipc.new_file(y_path, y_table.schema) as writer:
        writer.write_table(y_table)
    typer.echo(f"Saved Y coordinates to {y_path}")

    # Load labels.parquet
    labels_file = input_path / "labels.parquet"
    if not labels_file.exists():
        typer.echo(f"Error: {labels_file} not found", err=True)
        raise typer.Exit(1)

    typer.echo("Loading labels...")
    labels_df = pd.read_parquet(labels_file)

    # Process each label column
    typer.echo("Processing label columns...")
    categories_dict = {}

    for col in labels_df.columns:
        typer.echo(f"  Processing column: {col}")

        if labels_df[col].dtype.name == "category":
            # For categorical columns, get categories and codes
            categories = labels_df[col].cat.categories.tolist()
            indices = labels_df[col].cat.codes.astype(np.int16)
        else:
            # For non-categorical columns, create categories from unique values
            unique_values = labels_df[col].unique()
            # Sort to ensure consistent ordering
            categories = sorted([str(val) for val in unique_values if pd.notna(val)])

            # Create mapping from values to indices
            value_to_idx = {val: idx for idx, val in enumerate(categories)}
            indices = (
                labels_df[col]
                .map(lambda x: value_to_idx.get(str(x), -1) if pd.notna(x) else -1)
                .astype(np.int16)
            )

        # Store categories for metadata
        categories_dict[col] = categories

        # Export indices as Arrow file
        indices_array = pa.array(indices, type=pa.int16())
        indices_table = pa.table([indices_array], names=[col])
        col_path = output_path / f"{col}.arrow"
        with pa.ipc.new_file(col_path, indices_table.schema) as writer:
            writer.write_table(indices_table)
        typer.echo(f"    Saved {len(categories)} categories to {col_path}")

    # Create metadata.json
    typer.echo("Creating metadata...")
    metadata = {
        "categories": categories_dict,
        "xCenter": float(x_center),
        "yCenter": float(y_center),
        "maxRange": float(max_range)
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    typer.echo(f"Saved metadata to {metadata_path}")

    # Summary
    typer.echo("\n=== Export Summary ===")
    typer.echo(f"Exported {len(mappings)} data points")
    typer.echo(f"X/Y coordinates: {x_path}, {y_path}")
    typer.echo(f"Label columns: {len(categories_dict)}")
    for col, cats in categories_dict.items():
        typer.echo(f"  {col}: {len(cats)} categories")
    typer.echo(f"Metadata: {metadata_path}")
    typer.echo("Export complete! ✓")


if __name__ == "__main__":
    app()
