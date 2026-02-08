#!/usr/bin/env python

from pathlib import Path
from typing import Optional, Dict
import json

import numpy as np
import torch
import typer
import onnxruntime as ort
import umap_pytorch
import anndata as ad
import scanpy as sc
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create Typer app
app = typer.Typer(
    help="Train a parametric UMAP model from vectors and export it to ONNX format, then map vectors using the trained model.",
    add_completion=True,
)


@app.command()
def train(
    vectors_path: Path = typer.Argument(..., help="Path to embeddings.npy file"),
    labels_path: Path = typer.Argument(
        ..., help="Path to labels.parquet file for stratified sampling"
    ),
    output_path: Path = typer.Argument(..., help="Path to output the trained model"),
    num_vectors: Optional[int] = typer.Option(
        None,
        help="Limit the total number of vectors used for training, use all if None",
    ),
    stratify_label: str = typer.Option(
        "prediction",
        help="Label column name to stratify on",
    ),
) -> None:
    """
    Train a parametric UMAP model from vectors and export it to ONNX format.
    """
    typer.echo(f"Training parametric UMAP model from vectors {vectors_path}")

    # Load the vectors
    vectors = np.load(vectors_path)

    # Initialize random state for reproducibility
    random_state = np.random.RandomState(42)

    # Track training vector indices for consistency across train/map/export
    training_vector_ids = None

    # Select subset of vectors if num_vectors is specified
    if num_vectors is not None:
        if labels_path is not None:
            # Stratified sampling based on labels
            typer.echo(f"Loading labels from {labels_path}")
            labels_df = pd.read_parquet(labels_path)

            if stratify_label not in labels_df.columns:
                typer.echo(
                    f"Error: Column '{stratify_label}' not found in labels file",
                    err=True,
                )
                raise typer.Exit(1)

            if len(labels_df) != vectors.shape[0]:
                typer.echo(
                    f"Error: Number of labels ({len(labels_df)}) doesn't match number of vectors ({vectors.shape[0]})",
                    err=True,
                )
                raise typer.Exit(1)

            # Check for NaN values in the stratify column
            stratify_values = labels_df[stratify_label].values
            nan_mask = pd.isna(stratify_values)
            nan_count = nan_mask.sum()
            nan_percentage = (nan_count / len(stratify_values)) * 100

            if nan_count > 0:
                typer.echo(
                    f"⚠️ NaN values in '{stratify_label}' column: {nan_count} ({nan_percentage:.2f}%)"
                )

            if nan_percentage >= 10:
                typer.echo(
                    f"❌ Error: Too many NaN values ({nan_percentage:.2f}%) in stratify column '{stratify_label}'. Maximum allowed is 10%.",
                    err=True,
                )
                raise typer.Exit(1)

            # If there are NaN values but less than 10%, filter them out
            if nan_count > 0:
                typer.echo(
                    f"Filtering out {nan_count} rows with NaN values for stratification"
                )
                valid_mask = ~nan_mask
                valid_indices = np.arange(vectors.shape[0])[valid_mask]
                valid_stratify_values = stratify_values[valid_mask]

                # Adjust num_vectors to account for filtered rows
                adjusted_num_vectors = min(num_vectors, len(valid_indices))
                typer.echo(
                    f"Adjusted selection size: {adjusted_num_vectors} vectors from {len(valid_indices)} valid vectors"
                )

                # Use train_test_split for stratified sampling on valid indices
                selected_valid_indices, _ = train_test_split(
                    valid_indices,
                    test_size=1.0 - (adjusted_num_vectors / len(valid_indices)),
                    stratify=valid_stratify_values,
                    random_state=42,
                )
                selected_indices = selected_valid_indices
            else:
                # Clamp num_vectors to available vectors
                if num_vectors >= vectors.shape[0]:
                    if num_vectors > vectors.shape[0]:
                        typer.echo(
                            f"Requested {num_vectors} vectors but only {vectors.shape[0]} available, using all vectors"
                        )
                    selected_indices = np.arange(vectors.shape[0])
                else:
                    print(
                        f"Selecting {num_vectors} stratified vectors from {vectors.shape[0]} total vectors based on '{stratify_label}' column"
                    )

                    # Create indices array for stratified splitting
                    indices = np.arange(vectors.shape[0])

                    # Use train_test_split for stratified sampling
                    selected_indices, _ = train_test_split(
                        indices,
                        test_size=1.0 - (num_vectors / vectors.shape[0]),
                        stratify=stratify_values,
                        random_state=42,
                    )

            training_vector_ids = selected_indices
            vectors = vectors[selected_indices]
        else:
            # Random sampling (original behavior)
            print(
                f"Selecting {num_vectors} random vectors from {vectors.shape[0]} total vectors"
            )
            selected_indices = random_state.choice(
                vectors.shape[0], num_vectors, replace=False
            )
            training_vector_ids = selected_indices
            vectors = vectors[selected_indices]
    else:
        # Use all vectors
        training_vector_ids = np.arange(vectors.shape[0])

    # Determine number of vectors to use
    num_vectors = num_vectors if num_vectors is not None else vectors.shape[0]
    if num_vectors is not None:
        num_vectors = min(vectors.shape[0], num_vectors)

    vectors = vectors[:num_vectors]
    training_vector_ids = training_vector_ids[:num_vectors]

    # Create and fit the model
    model = umap_pytorch.PUMAP(
        encoder=None,  # nn.Module, None for default
        decoder=None,  # nn.Module, True for default, None for encoder only
        n_neighbors=10,
        min_dist=0.5,
        metric="euclidean",
        n_components=2,
        beta=1.0,  # How much to weigh reconstruction loss for decoder
        reconstruction_loss=torch.nn.functional.binary_cross_entropy_with_logits,  # pass in custom reconstruction loss functions
        random_state=random_state,  # For reproducibility
        lr=1e-3,
        epochs=15,
        num_workers=4,
        num_gpus=1,
        match_nonparametric_umap=False,  # Train network to match vectors from non parametric umap
    )

    model.fit(torch.from_numpy(vectors))

    # Export to ONNX
    model_output_path = output_path / "model.onnx"
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.model.encoder.encoder,
        torch.zeros(1, vectors.shape[1]),
        model_output_path,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=12,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    typer.echo(f"Saved clustering model to {model_output_path}")

    # Save training vector IDs for consistent mapping and export
    training_ids_path = vectors_path.parent / "pumap_training_vector_ids.npy"
    np.save(training_ids_path, training_vector_ids)
    typer.echo(f"Saved training vector IDs to {training_ids_path}")


def save_mappings_png(output_path: Path, mappings: np.ndarray) -> None:
    """
    Save raw mappings as a PNG scatter plot for comparison.

    Args:
        output_path: Directory where to save the PNG file
        mappings: Raw float32 mappings array of shape (num_vectors, 2)
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
    model_path: Path = typer.Argument(..., help="Path to model.onnx"),
    vectors_path: Path = typer.Argument(..., help="Path to embeddings.npy file"),
    output_path: Path = typer.Argument(
        ..., help="Path to output directory for the mapped data"
    ),
    batch_size: int = typer.Option(
        32, help="Number of vectors to process in each batch"
    ),
    export_png: bool = typer.Option(
        False, "--export-png", help="Export a reference PNG file"
    ),
    num_vectors: Optional[int] = typer.Option(
        None, help="Limit the total number of inputs processed, process all if None"
    ),
) -> None:
    """
    Map vectors to a lower-dimensional space using a trained ONNX model.
    """
    typer.echo(f"Mapping vectors from {vectors_path} using model {model_path}")

    # Load the cluster model
    onnx_session = ort.InferenceSession(str(model_path))
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Load the vectors
    vectors = np.load(vectors_path)

    # Select first num_vectors if specified, otherwise use all vectors
    if num_vectors is not None:
        num_vectors = min(num_vectors, vectors.shape[0])
        vectors = vectors[:num_vectors]
        typer.echo(f"Using first {num_vectors} vectors from {vectors_path}")
    else:
        num_vectors = vectors.shape[0]

    typer.echo(f"Processing {num_vectors} vectors with batch size {batch_size}")

    # Initialize list to store mappings
    all_mappings = []

    # Process in batches
    with tqdm(total=num_vectors) as pbar:
        for batch_start in range(0, num_vectors, batch_size):
            batch_end = min(batch_start + batch_size, num_vectors)
            batch_size_actual = batch_end - batch_start

            # Get batch of vectors
            batch_vectors = vectors[batch_start:batch_end]

            # Run the model
            model_input = {input_name: batch_vectors.astype(np.float32)}
            batch_mappings = onnx_session.run([output_name], model_input)[0]

            # Store the mappings
            all_mappings.append(batch_mappings)

            pbar.update(batch_size_actual)

    # Combine all batches
    mappings = np.concatenate(all_mappings)  # Shape: (num_vectors, 2)

    # Save raw mappings as a numpy file
    mappings_path = output_path / "mappings.npy"
    mappings_path.parent.mkdir(parents=True, exist_ok=True)
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
        ..., help="Path to output directory for binary files and metadata"
    ),
) -> None:
    """
    Export mappings and labels to binary format files.

    NOTE: Only the first len(mappings.npy) x and y are exported, but
    categories are exported for all rows in labels.parquet.

    Reads mappings.npy and labels.parquet from input_path and outputs:
    - x.bin and y.bin: Coordinate arrays as float32 binary files
    - {column_name}.bin: Category indices for each label column as int16 binary files
    - metadata.json: Contains categories mapping for label columns and data shapes
    """
    typer.echo(f"Exporting data from {input_path} to {output_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load mappings.npy
    mappings_file = input_path / "mappings.npy"
    if not mappings_file.exists():
        typer.echo(f"Error: {mappings_file} not found", err=True)
        raise typer.Exit(1)

    typer.echo("Loading mappings...")
    mappings = np.load(mappings_file)  # Shape: (num_vectors, 2)

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

    # Export x and y coordinates as separate binary files
    typer.echo("Exporting coordinate columns...")

    # X coordinates
    x_data = normalized_mappings[:, 0].astype(np.float32)
    x_path = output_path / "x.bin"
    x_data.tofile(x_path)
    typer.echo(f"Saved X coordinates to {x_path}")

    # Y coordinates
    y_data = normalized_mappings[:, 1].astype(np.float32)
    y_path = output_path / "y.bin"
    y_data.tofile(y_path)
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
            indices = labels_df[col].cat.codes.values.astype(np.int16)
        else:
            # For non-categorical columns, create categories from unique values
            unique_values = labels_df[col].unique()
            # Sort to ensure consistent ordering
            categories = sorted([str(val) for val in unique_values if pd.notna(val)])

            # Create mapping from values to indices
            value_to_idx = {val: idx for idx, val in enumerate(categories)}
            indices = np.array(
                [
                    value_to_idx.get(str(x), -1) if pd.notna(x) else -1
                    for x in labels_df[col]
                ],
                dtype=np.int16,
            )

        # Validate that the column doesn't have too many categories
        assert (
            len(categories) <= 500
        ), f"Column '{col}' has {len(categories)} categories, which exceeds the maximum of 500"

        # Store categories for metadata
        categories_dict[col] = categories

        # Export indices as binary file
        col_path = output_path / f"{col}.bin"
        indices.tofile(col_path)
        typer.echo(f"    Saved {len(categories)} categories to {col_path}")

    # Create metadata.json
    typer.echo("Creating metadata...")
    metadata = {
        "categories": categories_dict,
        "xCenter": float(x_center),
        "yCenter": float(y_center),
        "maxRange": float(max_range),
        "numPoints": len(mappings),
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
