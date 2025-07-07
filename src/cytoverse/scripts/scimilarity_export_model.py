#!/usr/bin/env python3

import typer
from pathlib import Path
import numpy as np

from typing import List, Dict

import anndata
import scanpy as sc

import scimilarity
from scimilarity import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts

import torch
import onnx
import onnxruntime as ort


# Create Typer app
app = typer.Typer(
    help="Export SCimilarity model to ONNX format",
    add_completion=False,
)


def create_inflation_map(adata, model_genes: List[str]) -> Dict[int, int]:
    """
    Create a mapping between the model's expected gene indices and the sample's gene indices.

    Args:
        adata: AnnData sample
        encoder_model_path: Path to the encoder model

    Returns:
        Dictionary mapping from sample gene indices to model gene indices
    """
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


class PreprocessingWrapper(torch.nn.Module):
    """
    Standalone preprocessing module that applies SCimilarity normalization steps:
    - sc.pp.normalize_total(data, target_sum=1e4) - all cells total to 10k
    - sc.pp.log1p(data) - X = log (X + 1)
    """

    def __init__(self, target_sum=1e4):
        """
        Parameters:
        -----------
        target_sum : float, optional
            Target sum for normalization (default: 1e4).
        """
        super(PreprocessingWrapper, self).__init__()
        self.target_sum = torch.tensor(float(target_sum), dtype=torch.float32)

    def forward(self, x):
        """
        Forward pass with preprocessing only.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns:
        --------
        torch.Tensor
            Preprocessed tensor ready for embedding model.
        """
        # Ensure input is float32
        x = x.to(torch.float32)

        # Handle NaN values by zeroing them (matches scimilarity behavior)
        x = torch.nan_to_num(x, nan=0.0)

        # Step 1: Normalize total counts per sample to target_sum
        # Compute row sums
        row_sums = torch.sum(x, dim=1, keepdim=True)

        # Replace zero sums with 1 to avoid division by zero (matches scimilarity behavior)
        row_sums = torch.where(
            row_sums == 0, torch.tensor(1.0, dtype=torch.float32), row_sums
        )

        # Compute scaling factors
        scaling_factors = self.target_sum / row_sums

        # Normalize by multiplying each row by its scaling factor
        normalized = x * scaling_factors

        # Step 2: Apply log1p transformation
        transformed = torch.log1p(normalized)

        return transformed


class NormalizationWrapper(torch.nn.Module):
    """
    Combined wrapper that applies preprocessing and embedding in sequence.
    """

    def __init__(self, original_module, target_sum=1e4):
        """
        Parameters:
        -----------
        original_module : torch.nn.Module
            The original PyTorch model to wrap.
        target_sum : float, optional
            Target sum for normalization (default: 1e4).
        """
        super(NormalizationWrapper, self).__init__()
        self.preprocessing = PreprocessingWrapper(target_sum)
        self.original_module = original_module

    def forward(self, x):
        """
        Forward pass with preprocessing followed by embedding.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns:
        --------
        torch.Tensor
            Output of the original module after preprocessing.
        """
        # Apply preprocessing
        preprocessed = self.preprocessing(x)

        # Pass through the original module
        output = self.original_module(preprocessed)

        return output


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
    Exports preprocessing and embedding steps separately for validation,
    then combines them into a single ONNX model.
    """
    print(f"Loading model from {model_path}...")
    ce = CellEmbedding(model_path)

    print("Converting to ONNX format...")
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Export preprocessing module
    print("\n1. Exporting preprocessing module...")
    preprocessing_model = PreprocessingWrapper(target_sum=1e4)
    preprocessing_path = output_path / "preprocessing.onnx"

    torch.onnx.export(
        preprocessing_model,
        torch.zeros(1, ce.n_genes),
        preprocessing_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["preprocessed"],
        dynamic_axes={"input": {0: "batch_size"}, "preprocessed": {0: "batch_size"}},
        verbose=False,
    )
    print(f"  Preprocessing ONNX saved to {preprocessing_path}")

    # Step 2: Export embedding module (standalone)
    print("\n2. Exporting embedding module...")
    embedding_path = output_path / "embedding.onnx"

    torch.onnx.export(
        ce.model,
        torch.zeros(1, ce.n_genes),
        embedding_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["preprocessed"],
        output_names=["embedding"],
        dynamic_axes={
            "preprocessed": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        verbose=False,
    )
    print(f"  Embedding ONNX saved to {embedding_path}")

    # Step 3: Export combined model
    print("\n3. Exporting combined model...")
    combined_model = NormalizationWrapper(ce.model, target_sum=1e4)
    combined_path = output_path / "model.onnx"

    # Set combined model to evaluation mode before export
    combined_model.eval()
    torch.onnx.export(
        combined_model,
        torch.zeros(1, ce.n_genes),
        combined_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
    )
    print(f"  Combined ONNX saved to {combined_path}")

    # Validate all models
    print("\n4. Validating ONNX models...")
    for model_file, name in [
        (preprocessing_path, "Preprocessing"),
        (embedding_path, "Embedding"),
        (combined_path, "Combined"),
    ]:
        onnx_model = onnx.load(str(model_file))
        onnx.checker.check_model(onnx_model)
        print(f"  ✅ {name} ONNX model validation passed")

    # Step 5: Individual component validation
    print("\n5. Validating individual components...")

    # Load test data and use align_dataset for consistent input
    adata = sc.read_h5ad("data/GSE136831_subsample.h5ad")
    adata = adata[0:10, :].copy()
    adata = align_dataset(adata, ce.gene_order)

    # Check for and handle NaN values (matches scimilarity behavior)
    if hasattr(adata.X, "toarray"):
        X_array = adata.X.toarray()
    else:
        X_array = adata.X

    if np.isnan(X_array).any():
        print("    ⚠️ NaN values detected in test data, zeroing them...")
        X_array = np.nan_to_num(X_array, nan=0.0)
        adata.X = X_array

    # Get raw aligned data (before normalization) for preprocessing test
    raw_aligned_batch = X_array.astype(np.float32)

    # Test preprocessing step
    print("  Testing preprocessing step...")
    preprocessing_pytorch = PreprocessingWrapper(target_sum=1e4)
    with torch.no_grad():
        preprocessed_pytorch = preprocessing_pytorch(
            torch.from_numpy(raw_aligned_batch)
        ).numpy()

    ort_session_prep = ort.InferenceSession(preprocessing_path)
    preprocessed_onnx = ort_session_prep.run(None, {"input": raw_aligned_batch})[0]

    max_diff_prep = np.max(np.abs(preprocessed_pytorch - preprocessed_onnx))
    print(f"    Preprocessing max diff: {max_diff_prep:.2e}")
    if max_diff_prep < 1e-5:
        print("    ✅ Preprocessing concordant")
    else:
        print(f"    ❌ Preprocessing differs: {max_diff_prep:.2e}")

    # Test embedding step (using preprocessed output)
    print("  Testing embedding step...")

    # Use the same preprocessing that SCimilarity uses for fair comparison
    adata_for_embedding = sc.read_h5ad("data/GSE136831_subsample.h5ad")
    adata_for_embedding = adata_for_embedding[0:10, :].copy()
    adata_for_embedding = align_dataset(adata_for_embedding, ce.gene_order)
    adata_for_embedding = lognorm_counts(adata_for_embedding)
    scimilarity_preprocessed = (
        adata_for_embedding.X.toarray().astype(np.float32)
        if hasattr(adata_for_embedding.X, "toarray")
        else adata_for_embedding.X.astype(np.float32)
    )

    # Set model to evaluation mode (like working version)
    ce.model.eval()
    with torch.no_grad():
        embedding_pytorch = ce.model(torch.from_numpy(scimilarity_preprocessed)).numpy()

    ort_session_emb = ort.InferenceSession(embedding_path)
    embedding_onnx = ort_session_emb.run(
        None, {"preprocessed": scimilarity_preprocessed}
    )[0]

    max_diff_emb = np.max(np.abs(embedding_pytorch - embedding_onnx))
    print(f"    Embedding max diff: {max_diff_emb:.2e}")
    if max_diff_emb < 1e-5:
        print("    ✅ Embedding concordant")
    else:
        print(f"    ❌ Embedding differs: {max_diff_emb:.2e}")

    # Step 6: Final combined validation
    print("\n6. Validating combined model...")

    # Run the ONNX model
    ort_session_combined = ort.InferenceSession(combined_path)
    combined_onnx_output = ort_session_combined.run(None, {"input": raw_aligned_batch})[
        0
    ]

    # Run the original SCimilarity get_embeddings function
    # The get_embeddings function expects log-normalized data.
    # Our ONNX model does this internally, but for a fair comparison,
    # we must manually preprocess for get_embeddings.
    adata_for_embedding = sc.read_h5ad("data/GSE136831_subsample.h5ad")
    adata_for_embedding = adata_for_embedding[0:10, :].copy()
    adata_for_embedding = align_dataset(adata_for_embedding, ce.gene_order)
    adata_for_embedding = lognorm_counts(adata_for_embedding)
    scimilarity_preprocessed = (
        adata_for_embedding.X.toarray().astype(np.float32)
        if hasattr(adata_for_embedding.X, "toarray")
        else adata_for_embedding.X.astype(np.float32)
    )
    scimilarity_output = ce.get_embeddings(scimilarity_preprocessed)

    # Compare the two outputs
    max_diff = np.max(np.abs(combined_onnx_output - scimilarity_output))
    mean_diff = np.mean(np.abs(combined_onnx_output - scimilarity_output))

    print(f"  ONNX vs SCimilarity get_embeddings:")
    print(f"    Max absolute difference: {max_diff:.2e}")
    print(f"    Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("    ✅ Combined ONNX concordant with SCimilarity")
    elif max_diff < 1e-3:
        print("    ⚠️ Combined ONNX has small differences with SCimilarity (< 1e-3)")
    else:
        print(
            f"    ❌ Combined ONNX differs significantly from SCimilarity: {max_diff:.2e}"
        )

    # Export genes file
    print("\n7. Exporting genes...")
    genes_path = output_path / "genes.txt"
    with open(genes_path, "w") as f:
        f.write("\n".join(map(str, ce.gene_order)))
    print(f"  Genes saved to {genes_path}")

    print("\n🎉 Model export complete!")
    print(f"  - Preprocessing: {preprocessing_path}")
    print(f"  - Embedding: {embedding_path}")
    print(f"  - Combined: {combined_path}")
    print(f"  - Genes: {genes_path}")


if __name__ == "__main__":
    app()
