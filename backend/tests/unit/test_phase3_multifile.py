"""
Unit tests for Phase 3 features: Multi-file processing with label intersection.
"""

import numpy as np
import pytest
import scanpy as sc
import tempfile
from pathlib import Path
import pandas as pd

from h5ad_to_embeddings import (
    _compute_label_intersection,
    _process_single_h5ad_file,
    _process_multiple_h5ad_files,
    _load_model_genes,
)
import onnxruntime as ort


@pytest.fixture
def test_adata_file1(tmp_path):
    """Create first test h5ad file with specific labels."""
    np.random.seed(42)
    n_cells = 50
    n_genes = 100

    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    adata = sc.AnnData(X=X)
    adata.obs["cell_type"] = [f"type_{i % 3}" for i in range(n_cells)]
    adata.obs["tissue"] = [f"tissue_{i % 2}" for i in range(n_cells)]
    adata.obs["donor"] = [f"donor_{i % 4}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    file_path = tmp_path / "test_file1.h5ad"
    adata.write_h5ad(file_path)
    return str(file_path)


@pytest.fixture
def test_adata_file2(tmp_path):
    """Create second test h5ad file with overlapping labels."""
    np.random.seed(123)
    n_cells = 30
    n_genes = 100

    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    adata = sc.AnnData(X=X)
    adata.obs["cell_type"] = [f"type_{i % 3}" for i in range(n_cells)]
    adata.obs["tissue"] = [f"tissue_{i % 2}" for i in range(n_cells)]
    adata.obs["batch"] = [f"batch_{i % 2}" for i in range(n_cells)]  # Different label
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    file_path = tmp_path / "test_file2.h5ad"
    adata.write_h5ad(file_path)
    return str(file_path)


@pytest.fixture
def test_adata_file3(tmp_path):
    """Create third test h5ad file with minimal overlap."""
    np.random.seed(456)
    n_cells = 40
    n_genes = 100

    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    adata = sc.AnnData(X=X)
    adata.obs["cell_type"] = [f"type_{i % 3}" for i in range(n_cells)]
    adata.obs["sample"] = [f"sample_{i % 3}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    file_path = tmp_path / "test_file3.h5ad"
    adata.write_h5ad(file_path)
    return str(file_path)


def test_label_intersection_two_files(test_adata_file1, test_adata_file2):
    """Test label intersection with two files."""
    h5ad_paths = [test_adata_file1, test_adata_file2]
    requested_labels = ["cell_type", "tissue", "donor", "batch"]

    valid_labels = _compute_label_intersection(h5ad_paths, requested_labels)

    # Only cell_type and tissue are in both files
    assert set(valid_labels) == {"cell_type", "tissue"}
    assert "donor" not in valid_labels  # Only in file1
    assert "batch" not in valid_labels  # Only in file2


def test_label_intersection_three_files(
    test_adata_file1, test_adata_file2, test_adata_file3
):
    """Test label intersection with three files."""
    h5ad_paths = [test_adata_file1, test_adata_file2, test_adata_file3]
    requested_labels = ["cell_type", "tissue", "donor", "batch", "sample"]

    valid_labels = _compute_label_intersection(h5ad_paths, requested_labels)

    # Only cell_type is in all three files
    assert valid_labels == ["cell_type"]


def test_label_intersection_empty_request():
    """Test label intersection with no requested labels."""
    result = _compute_label_intersection([], [])
    assert result == []


def test_process_single_h5ad_file(tmp_path):
    """Test single file processing returns correct shapes."""
    # Create a simple test file
    np.random.seed(42)
    n_cells = 20
    n_genes = 50

    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    adata = sc.AnnData(X=X)
    adata.obs["cell_type"] = [f"type_{i % 3}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    h5ad_path = tmp_path / "test.h5ad"
    adata.write_h5ad(h5ad_path)

    # Load model (using a dummy ONNX session for testing)
    # In real tests, this would use the actual ONNX model
    # For now, we'll skip this test and rely on E2E tests
    pytest.skip("Requires ONNX model fixture")


def test_process_multiple_files_concatenation(tmp_path):
    """Test that multi-file processing concatenates correctly."""
    # Create two simple files
    np.random.seed(42)
    files = []

    for i in range(2):
        n_cells = 20 + i * 10
        n_genes = 50

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        adata = sc.AnnData(X=X)
        adata.obs["cell_type"] = [f"type_{j % 3}" for j in range(n_cells)]
        adata.var_names = [f"gene_{j}" for j in range(n_genes)]

        file_path = tmp_path / f"test_file_{i}.h5ad"
        adata.write_h5ad(file_path)
        files.append(str(file_path))

    # This test would require ONNX model
    pytest.skip("Requires ONNX model fixture")


def test_label_intersection_preserves_order(test_adata_file1, test_adata_file2):
    """Test that label intersection preserves the order of requested labels."""
    h5ad_paths = [test_adata_file1, test_adata_file2]

    # Request in specific order
    requested_labels = ["tissue", "cell_type"]
    valid_labels = _compute_label_intersection(h5ad_paths, requested_labels)

    # Should preserve order
    assert valid_labels == ["tissue", "cell_type"]


def test_label_intersection_no_common_labels(test_adata_file1, test_adata_file3):
    """Test label intersection when files have no common labels except one."""
    h5ad_paths = [test_adata_file1, test_adata_file3]
    requested_labels = ["tissue", "donor", "sample"]

    valid_labels = _compute_label_intersection(h5ad_paths, requested_labels)

    # None of these labels are in both files
    assert valid_labels == []
