"""
Test ONNX-based embedding pipeline concordance with PyTorch baseline.

This test verifies that the ONNX model produces identical embeddings
(within tolerance 1e-5) to the original PyTorch SCimilarity model.
"""

import pytest
import numpy as np
from pathlib import Path
import scanpy as sc
import onnxruntime as ort

# Import from scimilarity for baseline
import scimilarity
from scimilarity.utils import align_dataset, lognorm_counts

# Import ONNX functions
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from h5ad_to_embeddings import (
    _load_model_genes,
    _create_inflation_indices,
    _inflate_batch_to_model_space,
    _compute_embeddings_onnx,
)


@pytest.fixture
def test_h5ad():
    """Small test h5ad file."""
    return Path("fixtures/GSE136831_subsample_100.h5ad")


@pytest.fixture
def pytorch_model_path():
    """Path to PyTorch SCimilarity model."""
    return Path("/Users/rcurrie/data/cytoverse/models/scimilarity/model_v1.1")


@pytest.fixture
def onnx_model_path(tmp_path):
    """
    Path to ONNX model. For now, we'll export it in the test setup.
    Later this can be a pre-exported model.
    """
    # This will be set after exporting the model
    onnx_dir = tmp_path / "onnx_model"
    onnx_dir.mkdir()
    return onnx_dir / "model.onnx"


@pytest.fixture
def genes_path(tmp_path):
    """Path to genes.txt file."""
    return tmp_path / "genes.txt"


def test_onnx_vs_pytorch_baseline(
    test_h5ad, pytorch_model_path, tmp_path, pytestconfig
):
    """
    Test that ONNX embeddings match PyTorch embeddings within tolerance.

    This is the critical concordance test that validates the ONNX
    pipeline produces identical results to the PyTorch pipeline.
    """
    # Skip if models don't exist
    if not pytorch_model_path.exists():
        pytest.skip(f"PyTorch model not found at {pytorch_model_path}")

    if not test_h5ad.exists():
        pytest.skip(f"Test fixture not found at {test_h5ad}")

    # Export ONNX model first
    print("\n=== Exporting ONNX Model ===")
    ce = scimilarity.CellEmbedding(str(pytorch_model_path))

    onnx_model_path = tmp_path / "model.onnx"
    genes_path = tmp_path / "genes.txt"

    # Export genes
    with open(genes_path, "w") as f:
        f.write("\n".join(map(str, ce.gene_order)))

    # Export ONNX model
    import torch

    class NormalizationWrapper(torch.nn.Module):
        """Combined wrapper that applies preprocessing and embedding."""

        def __init__(self, original_module, target_sum=1e4):
            super(NormalizationWrapper, self).__init__()
            self.target_sum = torch.tensor(float(target_sum), dtype=torch.float32)
            self.original_module = original_module

        def forward(self, x):
            # Preprocessing
            x = x.to(torch.float32)
            x = torch.nan_to_num(x, nan=0.0)
            row_sums = torch.sum(x, dim=1, keepdim=True)
            row_sums = torch.where(
                row_sums == 0, torch.tensor(1.0, dtype=torch.float32), row_sums
            )
            scaling_factors = self.target_sum / row_sums
            normalized = x * scaling_factors
            transformed = torch.log1p(normalized)

            # Embedding
            return self.original_module(transformed)

    combined_model = NormalizationWrapper(ce.model, target_sum=1e4)
    combined_model.eval()

    torch.onnx.export(
        combined_model,
        torch.zeros(1, ce.n_genes),
        onnx_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
    )

    print(f"ONNX model exported to {onnx_model_path}")

    # Load test data
    print("\n=== Loading Test Data ===")
    adata = sc.read_h5ad(test_h5ad)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    # Take first 10 cells for quick testing
    adata = adata[:10, :].copy()

    # PYTORCH PATH (following scimilarity_export_model.py validation pattern)
    print("\n=== PyTorch Pipeline ===")
    # PyTorch: align + lognorm + get_embeddings
    aligned_adata = align_dataset(adata, ce.gene_order)

    # Ensure we have raw counts in layers['counts']
    if "counts" in adata.layers:
        aligned_adata.layers["counts"] = aligned_adata.layers["counts"]
    elif "raw_counts" in adata.layers:
        aligned_adata.layers["counts"] = aligned_adata.layers["raw_counts"]
    elif "raw" in adata.layers:
        aligned_adata.layers["counts"] = aligned_adata.layers["raw"]
    else:
        # Assume .X contains raw counts
        aligned_adata.layers["counts"] = aligned_adata.X

    lognorm_adata = lognorm_counts(aligned_adata)

    # Extract preprocessed data for PyTorch
    scimilarity_preprocessed = (
        lognorm_adata.X.toarray().astype(np.float32)
        if hasattr(lognorm_adata.X, "toarray")
        else lognorm_adata.X.astype(np.float32)
    )
    pytorch_embeddings = ce.get_embeddings(scimilarity_preprocessed)

    print(f"PyTorch embeddings shape: {pytorch_embeddings.shape}")

    # ONNX PATH (following scimilarity_export_model.py validation pattern)
    print("\n=== ONNX Pipeline ===")
    # ONNX: raw aligned counts (model does preprocessing internally)
    onnx_session = ort.InferenceSession(str(onnx_model_path))

    # Get raw aligned counts (from layers['counts'], NOT .X which might be normalized)
    raw_aligned_counts = (
        aligned_adata.layers["counts"].toarray().astype(np.float32)
        if hasattr(aligned_adata.layers["counts"], "toarray")
        else aligned_adata.layers["counts"].astype(np.float32)
    )

    # Run through ONNX (which does preprocessing internally)
    onnx_embeddings = onnx_session.run(
        ["output"], {"input": raw_aligned_counts}
    )[0]

    print(f"ONNX embeddings shape: {onnx_embeddings.shape}")

    # COMPARE
    print("\n=== Comparing Embeddings ===")
    assert (
        pytorch_embeddings.shape == onnx_embeddings.shape
    ), f"Shape mismatch: {pytorch_embeddings.shape} vs {onnx_embeddings.shape}"

    max_diff = np.max(np.abs(pytorch_embeddings - onnx_embeddings))
    mean_diff = np.mean(np.abs(pytorch_embeddings - onnx_embeddings))

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    tolerance = 1e-5
    assert (
        max_diff < tolerance
    ), f"ONNX embeddings differ from PyTorch by {max_diff:.2e} (tolerance: {tolerance})"

    print(f"✅ All embeddings match within tolerance ({tolerance})!")


def test_gene_inflation_all_present():
    """Test gene inflation when all sample genes are in model."""
    sample_genes = ["GENE1", "GENE2", "GENE3"]
    model_genes = ["GENE1", "GENE2", "GENE3", "GENE4"]

    inflation_indices = _create_inflation_indices(sample_genes, model_genes)

    assert len(inflation_indices) == 3
    assert inflation_indices[0] == 0  # GENE1 -> index 0
    assert inflation_indices[1] == 1  # GENE2 -> index 1
    assert inflation_indices[2] == 2  # GENE3 -> index 2


def test_gene_inflation_partial_overlap():
    """Test gene inflation when some genes are missing."""
    sample_genes = ["GENE1", "GENE_MISSING", "GENE3"]
    model_genes = ["GENE1", "GENE2", "GENE3"]

    inflation_indices = _create_inflation_indices(sample_genes, model_genes)

    assert len(inflation_indices) == 3
    assert inflation_indices[0] == 0  # GENE1 -> index 0
    assert inflation_indices[1] == -1  # GENE_MISSING -> not found
    assert inflation_indices[2] == 2  # GENE3 -> index 2


def test_batch_inflation_preserves_values():
    """Test that batch inflation correctly maps values."""
    # Sample has 3 genes, model has 5 genes
    sample_genes = ["GENE_A", "GENE_B", "GENE_C"]
    model_genes = ["GENE_B", "GENE_X", "GENE_A", "GENE_Y", "GENE_C"]

    inflation_indices = _create_inflation_indices(sample_genes, model_genes)

    # Create batch data: 2 cells x 3 genes
    batch_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    inflated = _inflate_batch_to_model_space(batch_data, inflation_indices, 5)

    # Expected shape: 2 cells x 5 genes
    assert inflated.shape == (2, 5)

    # Check mappings:
    # GENE_A (sample idx 0) -> model idx 2
    # GENE_B (sample idx 1) -> model idx 0
    # GENE_C (sample idx 2) -> model idx 4
    # GENE_X and GENE_Y should be zero

    # Cell 0
    assert inflated[0, 0] == 2.0  # GENE_B value
    assert inflated[0, 1] == 0.0  # GENE_X (not in sample)
    assert inflated[0, 2] == 1.0  # GENE_A value
    assert inflated[0, 3] == 0.0  # GENE_Y (not in sample)
    assert inflated[0, 4] == 3.0  # GENE_C value

    # Cell 1
    assert inflated[1, 0] == 5.0  # GENE_B value
    assert inflated[1, 2] == 4.0  # GENE_A value
    assert inflated[1, 4] == 6.0  # GENE_C value
