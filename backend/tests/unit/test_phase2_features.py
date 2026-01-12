"""
Unit tests for Phase 2 features: S3 support, CSR validation, random sampling.
"""

import numpy as np
import pytest
import scanpy as sc
from pathlib import Path
from scipy.sparse import csr_matrix, csc_matrix

from h5ad_to_embeddings import (
    _validate_and_convert_to_csr,
    _random_sample_cells,
)


@pytest.fixture
def test_adata():
    """Create a test AnnData object with dense matrix."""
    np.random.seed(42)
    n_cells = 100
    n_genes = 50

    # Create dense matrix
    X = np.random.rand(n_cells, n_genes).astype(np.float32)

    # Create AnnData
    adata = sc.AnnData(X=X)
    adata.obs["cell_type"] = [f"type_{i % 5}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return adata


def test_csr_validation_with_dense(test_adata):
    """Test CSR validation converts dense arrays to CSR."""
    from scipy.sparse import issparse, isspmatrix_csr

    # Start with dense
    assert not issparse(test_adata.X), "Should start as dense"

    # Save original dense data for comparison
    original_X_dense = test_adata.X.copy()

    # Convert to CSR
    result = _validate_and_convert_to_csr(test_adata)

    # Verify conversion
    assert issparse(result.X), "Should be sparse after conversion"
    assert isspmatrix_csr(result.X), "Should be CSR format"

    # Verify data integrity
    np.testing.assert_allclose(result.X.toarray(), original_X_dense, rtol=1e-6)


def test_csr_validation_with_csc():
    """Test CSR validation converts CSC to CSR."""
    from scipy.sparse import isspmatrix_csr, isspmatrix_csc

    np.random.seed(42)
    n_cells = 100
    n_genes = 50
    X_dense = np.random.rand(n_cells, n_genes).astype(np.float32)

    # Create AnnData with CSC matrix
    adata = sc.AnnData(X=csc_matrix(X_dense))
    assert isspmatrix_csc(adata.X), "Should start as CSC"

    # Convert to CSR
    result = _validate_and_convert_to_csr(adata)

    # Verify conversion
    assert isspmatrix_csr(result.X), "Should be CSR format"
    np.testing.assert_allclose(result.X.toarray(), X_dense, rtol=1e-6)


def test_csr_validation_already_csr():
    """Test CSR validation leaves CSR unchanged."""
    from scipy.sparse import isspmatrix_csr

    np.random.seed(42)
    n_cells = 100
    n_genes = 50
    X_dense = np.random.rand(n_cells, n_genes).astype(np.float32)

    # Create AnnData with CSR matrix
    adata = sc.AnnData(X=csr_matrix(X_dense))
    original_X = adata.X

    # Validate (should be no-op)
    result = _validate_and_convert_to_csr(adata)

    # Verify no change
    assert result.X is original_X or np.array_equal(
        result.X.toarray(), original_X.toarray()
    )


def test_random_sample_cells(test_adata):
    """Test random cell sampling with seed."""
    max_cells = 30
    seed = 42

    # Sample cells
    sampled = _random_sample_cells(test_adata, max_cells, seed)

    # Verify shape
    assert sampled.n_obs == max_cells, f"Expected {max_cells} cells"
    assert sampled.n_vars == test_adata.n_vars, "Genes should be unchanged"

    # Verify reproducibility
    sampled2 = _random_sample_cells(test_adata, max_cells, seed)
    assert np.array_equal(sampled.obs.index, sampled2.obs.index), "Should be reproducible"


def test_random_sample_preserves_data():
    """Test that random sampling preserves cell data integrity."""
    np.random.seed(42)
    n_cells = 100
    n_genes = 50

    # Create AnnData with identifiable data
    X = np.arange(n_cells * n_genes, dtype=np.float32).reshape(n_cells, n_genes)
    adata = sc.AnnData(X=X)
    adata.obs["cell_id"] = [f"cell_{i}" for i in range(n_cells)]

    # Sample
    max_cells = 20
    sampled = _random_sample_cells(adata, max_cells, seed=123)

    # Verify each sampled cell matches original
    for idx, obs_idx in enumerate(sampled.obs.index):
        original_idx = int(obs_idx)
        np.testing.assert_array_equal(
            sampled.X[idx], adata.X[original_idx], err_msg=f"Cell {obs_idx} data mismatch"
        )


def test_random_sample_assertions():
    """Test random sampling validation."""
    np.random.seed(42)
    adata = sc.AnnData(X=np.random.rand(100, 50))

    # Test max_cells > n_cells
    with pytest.raises(AssertionError, match="exceeds dataset size"):
        _random_sample_cells(adata, 200, seed=42)

    # Test max_cells <= 0
    with pytest.raises(AssertionError, match="must be positive"):
        _random_sample_cells(adata, 0, seed=42)

    with pytest.raises(AssertionError, match="must be positive"):
        _random_sample_cells(adata, -10, seed=42)
