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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Create Typer app
app = typer.Typer(
    help="Export an ONNX parametric UMAP model from a sample embedding file and map embeddings to 2D coordinates.",
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
    output_path: Path = typer.Argument(
        ..., help="Path to output directory for the trained model"
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
    num_embeddings: Optional[int] = typer.Option(
        None, help="Limit the total number of inputs processed, process all if None"
    ),
) -> None:
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

    # Read the labels file
    labels_df = pd.read_parquet(output_path / "labels.parquet")
    
    # Ensure we have the same number of rows
    if len(mappings) != len(labels_df):
        raise ValueError(f"Mismatch between mappings ({len(mappings)}) and labels ({len(labels_df)}) count")

    # Scale coordinates to INT16 for delta compression
    scaled_coords = (mappings * 1000).astype(np.int16)
    
    # Create the base mappings dataframe with coordinates
    mappings_df = pd.DataFrame({
        'x': scaled_coords[:, 0],
        'y': scaled_coords[:, 1]
    })
    
    # Process each label column to create category mappings
    categories_dict = {}
    
    for col in labels_df.columns:
        if labels_df[col].dtype.name == 'category':
            # Get unique categories and create mapping
            categories = labels_df[col].cat.categories.tolist()
            categories_dict[col] = categories
            
            # Add category indices to mappings dataframe as int16
            mappings_df[f'{col}_idx'] = labels_df[col].cat.codes.astype(np.int16)
        else:
            # Handle non-categorical columns by creating categories
            unique_values = labels_df[col].unique()
            categories_dict[col] = unique_values.tolist()
            
            # Create mapping from values to indices
            value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
            mappings_df[f'{col}_idx'] = labels_df[col].map(value_to_idx).astype(np.int16)
    
    # Sort by x coordinate for optimal delta compression
    mappings_df = mappings_df.sort_values('x').reset_index(drop=True)
    
    # Save optimized mappings with delta-friendly format
    mappings_path = output_path / "mappings.parquet"
    mappings_df.to_parquet(
        mappings_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    typer.echo(f"Saved optimized mappings to {mappings_path}")
    
    # Save categories mapping
    categories_df = pd.DataFrame.from_dict(categories_dict, orient='index')
    categories_df = categories_df.reset_index()
    categories_df.columns = ['column_name'] + [f'category_{i}' for i in range(categories_df.shape[1] - 1)]
    
    categories_path = output_path / "categories.parquet"
    categories_df.to_parquet(
        categories_path,
        engine="pyarrow", 
        compression="snappy",
        index=False,
    )
    typer.echo(f"Saved categories mapping to {categories_path}")
    
    # Print summary statistics
    typer.echo(f"Processed {len(mappings_df)} mappings with {len(categories_dict)} label columns:")
    for col, cats in categories_dict.items():
        typer.echo(f"  {col}: {len(cats)} categories")


if __name__ == "__main__":
    app()
