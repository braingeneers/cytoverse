#!/usr/bin/env python

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import typer
import onnxruntime as ort
from umap_pytorch import PUMAP
import anndata as ad
import scanpy as sc
from tqdm import tqdm

# Create Typer app
app = typer.Typer(
    help="Generate embeddings and train parametric UMAP models for single cell data using ONNX models.",
    add_completion=True,
)


@app.command()
def embed(
    onnx_model_path: Path = typer.Argument(
        ..., help="Path to embedding model .onnx file"
    ),
    sample_path: Path = typer.Argument(..., help="Path to .h5ad file"),
    batch_size: int = typer.Option(
        32, help="Number of samples to process in each batch"
    ),
    num_samples: Optional[int] = typer.Option(
        None, help="Limit the total number of inputs processed, process all if None"
    ),
) -> None:
    typer.echo(f"Generating embeddings {sample_path} using model {onnx_model_path}")

    # Instantiate the model
    model_session = ort.InferenceSession(str(onnx_model_path))

    # Get model input shape to determine the expected genes
    input_shape = model_session.get_inputs()[0].shape
    model_input_size = input_shape[1] if len(input_shape) > 1 else input_shape[0]

    # Load the sample data and preprocess it
    typer.echo(f"Loading sample data and preprocessing from {sample_path}...")
    adata = ad.read_h5ad(sample_path)
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    ### Logarithmizing the data
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # Create an inflation mapping between model genes and sample genes
    inflation_map = create_inflation_map(adata, onnx_model_path)

    # Read the classes file for the model
    classes_path = Path(onnx_model_path).with_suffix(".classes")
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f]

    typer.echo(f"Loaded {len(classes)} classes from {classes_path}")

    # Create a mapping from class name to index
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    # Determine number of cells to process
    num_cells = num_samples if num_samples is not None else adata.n_obs
    if num_samples is not None:
        num_cells = min(adata.n_obs, num_samples)

    typer.echo(f"Processing {num_cells} cells with batch size {batch_size}")

    # Initialize numpy array to store embeddings
    predictions = []
    embeddings = []
    ground_truth_labels = []

    # Check if the cell_type_field exists in adata.obs
    if cell_type_field not in adata.obs:
        typer.echo(
            f"Warning: '{cell_type_field}' not found in adata.obs. Using -1 as ground truth labels."
        )
        has_ground_truth = False
    else:
        has_ground_truth = True

    # Process in batches
    with tqdm(total=num_cells) as pbar:
        # Create a zero inflated data batch. We assume that each sample has the same
        # genes so we don't need to re-allocate this everytime, just inflate into it.
        inflated_batch = np.zeros((batch_size, model_input_size), dtype=np.float32)

        for batch_start in range(0, num_cells, batch_size):
            batch_end = min(batch_start + batch_size, num_cells)
            batch_size_actual = batch_end - batch_start

            # Get batch of expression data
            batch_expression = adata.X[batch_start:batch_end]

            # Handle last batch if its smaller then batch_size
            if batch_size_actual < batch_size:
                inflated_batch = np.zeros(
                    (batch_size_actual, model_input_size), dtype=np.float32
                )

            # # Convert to dense if it's sparse
            # if isinstance(batch_expression, np.ndarray) == False:
            #     batch_expression = batch_expression.toarray()

            # Fill in the data using the inflation map
            for sample_idx, model_idx in inflation_map.items():
                inflated_batch[:, model_idx] = batch_expression[:, sample_idx]

            # Run the model
            model_input = {"input": inflated_batch.astype(np.float32)}
            batch_predictions, batch_embeddings = model_session.run(
                ["topk_indices", "embedding"], model_input
            )

            # Store the embeddings
            embeddings.append(batch_embeddings)

            # Store just the top prediction indice
            predictions.append(batch_predictions[:, 0])

            # Extract ground truth class labels
            if has_ground_truth:
                batch_labels = adata.obs[cell_type_field].values[batch_start:batch_end]
                # Convert class names to indices
                batch_label_indices = np.array(
                    [class_to_idx.get(label, -1) for label in batch_labels],
                    dtype=np.int32,
                )
            else:
                # If no ground truth available, use -1 as placeholder
                batch_label_indices = np.full(batch_size_actual, -1, dtype=np.int32)

            ground_truth_labels.append(batch_label_indices)

            pbar.update(batch_size_actual)

    # Combine all batches
    all_embeddings = np.vstack(embeddings)
    all_predictions = np.concatenate(predictions)
    all_ground_truth = np.concatenate(ground_truth_labels)

    # Save the embeddings
    embeddings_path = onnx_model_path.with_name(
        f"{onnx_model_path.stem}-embeddings.npy"
    )
    np.save(embeddings_path, all_embeddings)

    # Save the predictions
    predictions_path = onnx_model_path.with_name(
        f"{onnx_model_path.stem}-predictions.npy"
    )
    np.save(predictions_path, all_predictions)

    # Create array of (ground_truth, prediction) pairs
    label_pairs = np.column_stack((all_ground_truth, all_predictions))

    # Save as binary file
    labels_bin_path = onnx_model_path.with_name(f"{onnx_model_path.stem}-labels.bin")
    label_pairs.astype(np.int32).flatten().tofile(labels_bin_path)

    typer.echo(f"Saved embeddings to {embeddings_path}")
    typer.echo(f"Saved predictions to {predictions_path}")
    typer.echo(f"Saved label pairs (ground truth, prediction) to {labels_bin_path}")


def create_inflation_map(adata, encoder_model_path: str) -> Dict[int, int]:
    """
    Create a mapping between the model's expected gene indices and the sample's gene indices.

    Args:
        adata: AnnData sample
        encoder_model_path: Path to the encoder model

    Returns:
        Dictionary mapping from sample gene indices to model gene indices
    """
    # Load the model genes
    model_genes_path = Path(encoder_model_path).with_suffix(".genes")
    with open(model_genes_path, "r") as f:
        model_genes = [line.strip() for line in f]

    # Create a mapping from gene names to indices for the model
    model_gene_to_idx = {gene: idx for idx, gene in enumerate(model_genes)}

    # Get the sample genes
    sample_genes = adata.var_names.tolist()

    # Create the inflation map: sample index -> model index
    inflation_map = {}

    # For each gene in the sample, find its index in the model
    for sample_idx, gene_name in enumerate(sample_genes):
        if gene_name in model_gene_to_idx:
            # If the gene exists in the model, add it to the mapping
            model_idx = model_gene_to_idx[gene_name]
            inflation_map[sample_idx] = model_idx

    print(
        f"{adata.X.shape[1] - len(inflation_map)} genes in the sample and not in the model"
    )
    return inflation_map


@app.command()
def train(
    sample_embeddings_path: Path = typer.Argument(
        ..., help="Path to embeddings.npy file"
    ),
    num_embeddings: Optional[int] = typer.Option(
        None,
        help="Limit the total number of embeddings used for training, use all if None",
    ),
) -> None:
    """
    Train a parametric umap dimensionality reduction model on the embeddings.

    Args:
        sample_embeddings_path: Path to sample-embeddings.npy file
        num_embeddings: Limit the total number of embeddings used for training, use all if None
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
    output_path = Path(sample_embeddings_path).with_name(
        f"{Path(sample_embeddings_path).stem.replace('-embeddings', '')}-pumap.onnx"
    )
    torch.onnx.export(
        model.model.encoder.encoder,
        torch.zeros(1, embeddings.shape[1]),
        output_path,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=12,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    typer.echo(f"Saved clustering model to {output_path}")

    typer.echo(f"Computing mappings for {sample_embeddings_path}...")
    mappings = model.transform(torch.from_numpy(embeddings))

    # Save the mappings
    output_path = Path(sample_embeddings_path).with_name(
        f"{Path(sample_embeddings_path).stem.replace('-embeddings', '')}-mappings.npy"
    )
    np.save(output_path, mappings)
    typer.echo(f"Saved mappings to {output_path}")


@app.command()
def map(
    model_path: Path = typer.Argument(..., help="Path to mapper.onnx"),
    sample_embeddings_path: Path = typer.Argument(
        ..., help="Path to sample-embeddings.npy file"
    ),
    labels_path: Path = typer.Argument(
        ..., help="Path to the labels.bin file (ground_truth, prediction pairs)"
    ),
    batch_size: int = typer.Option(
        32, help="Number of samples to process in each batch"
    ),
    num_samples: Optional[int] = typer.Option(
        None, help="Limit the total number of inputs processed, process all if None"
    ),
) -> None:
    """
    Map the embeddings to 2d coordinates using the mapper onnx model, combine with
    predicted class labels, and save in an optimized Apache ECharts format:
    Float32Array of [x1, y1, classIndex1, x2, y2, classIndex2, ...].

    Args:
        model_path: Path to pumap.onnx
        sample_embeddings_path: Path to sample-embeddings.npy file
        labels_path: Path to the labels.bin file containing (ground_truth, prediction) pairs.
        batch_size: Number of samples to process at a time
        num_samples: Limit the total number of inputs processed, process all if None
    """
    typer.echo(
        f"Mapping samples from {sample_embeddings_path} using model {model_path} and labels {labels_path}"
    )

    # Load the cluster model
    onnx_session = ort.InferenceSession(str(model_path))
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Load the embeddings
    embeddings = np.load(sample_embeddings_path)

    # Load the labels (ground_truth, prediction pairs)
    # The file contains flattened int32 pairs: [gt1, pred1, gt2, pred2, ...]
    label_pairs = np.fromfile(labels_path, dtype=np.int32)
    # Reshape to (n_samples, 2) and extract the predicted labels (second column)
    predicted_labels = label_pairs.reshape(-1, 2)[:, 1]

    # Determine number of samples to process
    num_embeddings = num_samples if num_samples is not None else embeddings.shape[0]
    if num_samples is not None:
        num_embeddings = min(embeddings.shape[0], num_samples)

    # Ensure the number of labels matches the number of embeddings to process
    if predicted_labels.shape[0] < num_embeddings:
        raise ValueError(
            f"Number of labels ({predicted_labels.shape[0]}) is less than the number of embeddings to process ({num_embeddings})."
        )
    # Slice labels if needed
    if predicted_labels.shape[0] > num_embeddings:
        predicted_labels = predicted_labels[:num_embeddings]

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

    # Combine mappings (x, y) with predicted labels
    # Reshape labels to (num_embeddings, 1) and cast to float32
    predicted_labels_float = predicted_labels.reshape(-1, 1).astype(np.float32)
    # Concatenate along the second axis to get [x, y, classIndex]
    combined_data = np.concatenate(
        (mappings, predicted_labels_float), axis=1
    )  # Shape: (num_embeddings, 3)

    # Flatten the array to [x1, y1, classIndex1, x2, y2, classIndex2, ...]
    flat_combined_data = combined_data.flatten()

    # Define the output path based on the embeddings file name
    output_filename = (
        f"{Path(sample_embeddings_path).stem.replace('-embeddings', '')}-mappings.bin"
    )
    output_path = Path(sample_embeddings_path).with_name(output_filename)

    # Save the flattened data as a binary file
    flat_combined_data.tofile(output_path)

    typer.echo(f"Saved optimized ECharts data [x, y, classIndex] to {output_path}")


if __name__ == "__main__":
    app()
