#!/usr/bin/env python3

import logging
import os
import typer
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from typing import Union, Any

import onnxruntime as ort
from tqdm import tqdm
from s3fs import S3FileSystem

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def _validate_s3_incremental_access(h5_file: Any) -> None:  # type: ignore[misc]
    """
    Validate that h5ad file supports efficient incremental access over S3.

    Checks that the X matrix and obs metadata are stored in formats that support
    efficient random access without requiring full file downloads.

    Args:
        h5_file: Open h5py File object

    Raises:
        ValueError: If file format doesn't support efficient incremental access
    """
    import h5py

    logger.info("  Validating file format for S3 incremental access...")

    # Check X matrix format
    x_group = h5_file["X"]

    # Check if it's CSR sparse matrix (efficient for row access)
    is_csr = (
        isinstance(x_group, h5py.Group)
        and "data" in x_group
        and "indices" in x_group
        and "indptr" in x_group
    )

    # Check if it's dense array
    is_dense = isinstance(x_group, h5py.Dataset)
    is_chunked_dense = is_dense and x_group.chunks is not None

    if is_csr:
        logger.info("  ✅ X matrix is stored as CSR sparse matrix (efficient)")
    elif is_dense:
        # Dense arrays can be contiguous (row-major) or chunked
        # Both support incremental access over S3
        if is_chunked_dense:
            chunks = x_group.chunks
            logger.info(
                "  ✅ X matrix is chunked dense array (chunk size: %s)", chunks
            )
            # Warn if chunks are column-oriented (bad for row access)
            chunk_rows, chunk_cols = chunks
            if chunk_rows == x_group.shape[0] and chunk_cols < x_group.shape[1]:
                logger.warning(
                    "  ⚠️  Column-oriented chunking detected - inefficient for cell access"
                )
            elif chunk_rows > 100:
                logger.warning(
                    "  ⚠️  Large row chunk size (%d) may cause inefficient S3 access",
                    chunk_rows,
                )
        else:
            # Unchunked dense - stored contiguously in row-major order
            # This DOES support efficient incremental access!
            plist = x_group.id.get_create_plist()
            layout = plist.get_layout()
            if layout == 1:  # H5D_CONTIGUOUS
                logger.info("  ✅ X matrix is contiguous dense array (row-major order)")
                logger.info("     Supports efficient incremental cell access over S3")
            else:
                logger.warning(
                    "  ⚠️  Unexpected HDF5 layout (code: %d)", layout
                )
    else:
        raise ValueError(
            f"Unknown X matrix storage format: {type(x_group)}\n"
            f"Expected CSR sparse matrix or dense array."
        )

    # Check obs metadata (should be reasonably small, but warn if huge)
    obs_group = h5_file["obs"]
    obs_index_key = "_index" if "_index" in obs_group else "index"
    n_cells = obs_group[obs_index_key].shape[0]

    # Estimate obs memory footprint
    obs_size_bytes = 0
    for key in obs_group.keys():
        item = obs_group[key]
        if isinstance(item, h5py.Dataset):
            obs_size_bytes += item.size * item.dtype.itemsize
        elif isinstance(item, h5py.Group) and "codes" in item:
            # Categorical: codes + categories
            obs_size_bytes += item["codes"].size * item["codes"].dtype.itemsize
            obs_size_bytes += (
                item["categories"].size * item["categories"].dtype.itemsize
            )

    obs_size_mb = obs_size_bytes / (1024**2)
    logger.info("  obs metadata: %d cells, ~%.2f MB", n_cells, obs_size_mb)

    if obs_size_mb > 100:
        logger.warning(
            "  ⚠️  Large obs metadata (%.2f MB) may cause slow incremental access",
            obs_size_mb,
        )

    logger.info("  ✅ File format validated for S3 incremental access")


def _load_h5ad_from_path_or_s3(
    path: Union[str, Path],
    max_cells: int | None = None,
    seed: int = 42,
    s3_profile: str | None = None,
    s3_anon: bool = False,
) -> sc.AnnData:
    """
    Load h5ad file from local path or S3 URI.

    For local files, uses backed mode for memory efficiency.
    For S3 files, can sample rows BEFORE loading to avoid reading entire matrix.

    Args:
        path: Local file path or S3 URI (s3://bucket/key)
        max_cells: If specified, randomly sample this many cells before loading
        seed: Random seed for sampling
        s3_profile: AWS profile name for S3 access (optional)
        s3_anon: Use anonymous access for public S3 buckets

    Returns:
        Loaded AnnData object
    """
    import anndata as ad
    import h5py

    path_str = str(path)

    if path_str.startswith("s3://"):
        logger.info("Loading from S3: %s", path_str)

        # Build S3FileSystem configuration
        s3_config: dict[str, Any] = {
            "anon": s3_anon,
            "skip_instance_cache": True,
            "use_listings_cache": False,
        }

        # Add profile if specified
        if s3_profile is not None:
            logger.info("  Using AWS profile: %s", s3_profile)
            s3_config["profile"] = s3_profile

        # Add custom endpoint if specified
        endpoint_url = os.getenv("AWS_ENDPOINT_URL", None)
        if endpoint_url:
            logger.info("  Using custom S3 endpoint: %s", endpoint_url)
            s3_config["client_kwargs"] = {"endpoint_url": endpoint_url}

        if s3_anon:
            logger.info("  Using anonymous S3 access")

        # Create S3 filesystem
        logger.info("  Opening S3 file via S3FileSystem...")
        fs = S3FileSystem(**s3_config)

        # Extract bucket and key from s3://bucket/key
        s3_path = path_str.replace("s3://", "")

        try:
            # Open file with S3FileSystem - this creates a file-like object
            # that h5py can read from with random access
            s3_file = fs.open(s3_path, "rb")

            # Open the S3 file handle with h5py
            h5_file = h5py.File(s3_file, "r")
            logger.info("  Successfully opened S3 file")
        except Exception as e:
            logger.error("  Failed to open S3 file: %s", e)
            raise

        # Validate file format for efficient S3 access BEFORE reading data
        _validate_s3_incremental_access(h5_file)

        # Check total number of cells
        # h5_file['obs'] is a Group, not a Dataset, so we need to access index
        obs_index_key = "_index" if "_index" in h5_file["obs"] else "index"
        n_cells = h5_file["obs"][obs_index_key].shape[0]
        logger.info("  Total cells in file: %d", n_cells)

        # If sampling, determine which rows to read
        if max_cells is not None and n_cells > max_cells:
            logger.info("  Sampling %d cells before loading", max_cells)
            np.random.seed(seed)
            indices = np.random.choice(n_cells, size=max_cells, replace=False)
            indices.sort()
            logger.info(
                "  Sample indices: min=%d, max=%d", indices.min(), indices.max()
            )

            # Read only sampled rows from h5ad
            # This is memory-efficient - only reads what we need!
            adata = _read_h5ad_subset(h5_file, indices)
            logger.info("  Loaded %d sampled cells from S3", len(adata))
        else:
            # Load entire file using same parsing logic as subset
            logger.info("  Loading entire file from S3")
            indices = np.arange(n_cells)
            adata = _read_h5ad_subset(h5_file, indices)
            logger.info("  Loaded %d cells from S3", len(adata))

        # Store file references so they don't get closed prematurely
        adata._h5_file = h5_file
        adata._s3_file = s3_file

    else:
        logger.info("Loading from local path: %s", path_str)
        # For local files, use backed mode for memory efficiency
        adata = ad.read_h5ad(path_str, backed="r")

    return adata


def _read_h5ad_subset(h5_file, indices: np.ndarray) -> sc.AnnData:
    """
    Read a subset of cells from an h5ad file.

    Only reads the specified row indices, avoiding loading the entire matrix.

    Args:
        h5_file: Open h5py File object
        indices: Row indices to read

    Returns:
        AnnData object with only the specified cells
    """
    import anndata as ad
    import pandas as pd
    import h5py

    # Read obs (cell metadata) for selected indices
    obs_group = h5_file["obs"]

    # h5py doesn't support fancy indexing with numpy arrays directly
    # We need to read elements individually or use a list
    indices_list = indices.tolist()

    # Determine index key name (could be 'index' or '_index')
    index_key = "_index" if "_index" in obs_group else "index"

    # Read index column to get cell names
    obs_index = obs_group[index_key][indices_list]
    if hasattr(obs_index[0], "decode"):
        obs_index = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in obs_index
        ]
    else:
        obs_index = [str(x) for x in obs_index]

    obs_df = pd.DataFrame(index=obs_index)

    # Read other obs columns
    for key in obs_group.keys():
        if key not in (index_key, "_index", "index"):
            item = obs_group[key]

            # Check if it's a Dataset or a Group (categorical)
            if isinstance(item, h5py.Dataset):
                # Regular column - read directly
                values = item[indices_list]
                # Decode bytes if necessary
                if len(values) > 0 and hasattr(values[0], "decode"):
                    values = [
                        x.decode("utf-8") if isinstance(x, bytes) else x for x in values
                    ]
                obs_df[key] = values
            else:
                # Categorical column stored as Group with codes/categories
                if "codes" in item and "categories" in item:
                    # Read codes for selected indices
                    codes = item["codes"][indices_list]
                    # Read categories
                    categories = item["categories"][:]
                    # Decode categories if bytes
                    if len(categories) > 0 and hasattr(categories[0], "decode"):
                        categories = [
                            x.decode("utf-8") if isinstance(x, bytes) else x
                            for x in categories
                        ]
                    # Map codes to category values
                    values = [categories[c] if c >= 0 else None for c in codes]
                    obs_df[key] = values

    # Read var (gene metadata) - all genes
    var_group = h5_file["var"]

    # Determine index key name - try common alternatives
    var_index_key = None
    for possible_key in ["_index", "index", "row_names", "gene_names", "var_names"]:
        if possible_key in var_group:
            var_index_key = possible_key
            break

    assert var_index_key is not None, (
        f"Could not find var index in h5ad file. "
        f"Available var keys: {list(var_group.keys())}"
    )

    var_index_raw = var_group[var_index_key][:]
    logger.info(
        f"  var index dtype: {var_index_raw.dtype}, first 5: {var_index_raw[:5]}"
    )

    # Convert to strings, handling bytes
    if len(var_index_raw) > 0 and hasattr(var_index_raw[0], "decode"):
        var_index = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in var_index_raw
        ]
    else:
        var_index = [str(x) for x in var_index_raw]

    logger.info(f"  Converted var index first 5: {var_index[:5]}")

    # Check if index contains only numeric strings (invalid gene names)
    is_all_numeric = all(
        name.isdigit() for name in var_index[: min(100, len(var_index))]
    )

    if is_all_numeric:
        logger.warning(
            "  ⚠️  var index contains only numeric strings, looking for gene symbols..."
        )
        # Check common gene name columns
        gene_col_found = False
        for potential_gene_col in [
            "gene_symbols",
            "gene_names",
            "genes",
            "var_names",
            "feature_name",
        ]:
            if potential_gene_col in var_group:
                logger.info(f"  Found {potential_gene_col} column, using as index")
                gene_names_raw = var_group[potential_gene_col][:]
                if hasattr(gene_names_raw[0], "decode"):
                    var_index = [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in gene_names_raw
                    ]
                else:
                    var_index = [str(x) for x in gene_names_raw]
                logger.info(f"  Updated var index first 5: {var_index[:5]}")
                gene_col_found = True
                break

        if not gene_col_found:
            raise ValueError(
                f"h5ad file has numeric gene indices instead of gene symbols. "
                f"var_names contains: {var_index[:10]} "
                f"Available var columns: {list(var_group.keys())} "
                f"This file appears to be improperly created - gene symbols are missing. "
                f"Please regenerate the h5ad file with proper gene names in var_names."
            )

    var_df = pd.DataFrame(index=var_index)
    for key in var_group.keys():
        if key not in (var_index_key, "_index", "index"):
            item = var_group[key]

            # Check if it's a Dataset or a Group (categorical)
            if isinstance(item, h5py.Dataset):
                values = item[:]
                if len(values) > 0 and hasattr(values[0], "decode"):
                    values = [
                        x.decode("utf-8") if isinstance(x, bytes) else x for x in values
                    ]
                var_df[key] = values
            else:
                # Categorical column
                if "codes" in item and "categories" in item:
                    codes = item["codes"][:]
                    categories = item["categories"][:]
                    if len(categories) > 0 and hasattr(categories[0], "decode"):
                        categories = [
                            x.decode("utf-8") if isinstance(x, bytes) else x
                            for x in categories
                        ]
                    values = [categories[c] if c >= 0 else None for c in codes]
                    var_df[key] = values

    # Read .X matrix for selected rows only
    x_group = h5_file["X"]

    # Check if it's CSR sparse matrix (stored as Group with data/indices/indptr)
    if (
        isinstance(x_group, h5py.Group)
        and "data" in x_group
        and "indices" in x_group
        and "indptr" in x_group
    ):
        # CSR format - read selected rows efficiently
        logger.info("  Reading CSR sparse matrix (selected rows only)")
        X = _read_csr_subset(x_group, indices)
    else:
        # Dense or other format - read selected rows
        logger.info("  Reading dense matrix (selected rows only)")
        # For dense matrices, we need to read row by row or use list indexing
        X = x_group[indices_list, :]

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

    return adata


def _read_csr_subset(csr_group, row_indices: np.ndarray):
    """
    Read a subset of rows from a CSR sparse matrix stored in HDF5.

    Args:
        csr_group: HDF5 group containing 'data', 'indices', 'indptr'
        row_indices: Rows to extract

    Returns:
        CSR matrix with only the specified rows
    """
    from scipy.sparse import csr_matrix

    # Debug: log CSR group structure
    logger.info(f"  CSR group keys: {list(csr_group.keys())}")
    logger.info(f"  CSR group attrs: {dict(csr_group.attrs)}")
    for key in ["data", "indices", "indptr"]:
        if key in csr_group:
            ds = csr_group[key]
            logger.info(
                f"  {key}: shape={ds.shape}, dtype={ds.dtype}, attrs={dict(ds.attrs)}"
            )

    indptr_full = csr_group["indptr"][:]

    # Build new CSR matrix with only selected rows
    data_list = []
    indices_list = []
    indptr_new = [0]

    for row_idx in row_indices:
        start = indptr_full[row_idx]
        end = indptr_full[row_idx + 1]

        # Read data and indices for this row
        row_data = csr_group["data"][start:end]
        row_indices_csr = csr_group["indices"][start:end]

        data_list.append(row_data)
        indices_list.append(row_indices_csr)
        indptr_new.append(indptr_new[-1] + len(row_data))

    # Concatenate all rows
    data = np.concatenate(data_list) if data_list else np.array([], dtype=np.float32)
    indices_concat = (
        np.concatenate(indices_list) if indices_list else np.array([], dtype=np.int32)
    )
    indptr = np.array(indptr_new, dtype=np.int32)

    # Determine number of columns
    # Try multiple approaches to get n_cols from HDF5 storage
    n_cols = None

    # Approach 1: Check 'shape' attribute on various datasets
    for key in ["indices", "data", "indptr"]:
        if "shape" in csr_group[key].attrs:
            shape = csr_group[key].attrs["shape"]
            if len(shape) >= 2:
                n_cols = shape[1]
                break

    # Approach 2: Check parent group attributes
    if n_cols is None and "shape" in csr_group.attrs:
        shape = csr_group.attrs["shape"]
        if len(shape) >= 2:
            n_cols = shape[1]

    # Approach 3: Infer from max column index
    if n_cols is None:
        if len(indices_concat) > 0:
            n_cols = int(indices_concat.max()) + 1
        else:
            # No data - check full indices dataset
            full_indices = csr_group["indices"][:]
            n_cols = int(full_indices.max()) + 1 if len(full_indices) > 0 else 0

    if n_cols is None or n_cols == 0:
        raise ValueError(
            f"Could not determine number of columns for CSR matrix. "
            f"Checked attributes: {dict(csr_group.attrs)}, "
            f"indices range: {indices_concat.min() if len(indices_concat) > 0 else 'empty'} to "
            f"{indices_concat.max() if len(indices_concat) > 0 else 'empty'}"
        )

    logger.info(f"  CSR matrix shape: ({len(row_indices)}, {n_cols})")

    return csr_matrix((data, indices_concat, indptr), shape=(len(row_indices), n_cols))


def _validate_and_convert_to_csr(adata: sc.AnnData) -> sc.AnnData:
    """
    Validate that .X is in CSR format for efficient cell-by-cell access.

    For backed mode (S3 files), verifies the storage format supports efficient
    row-wise random access without loading the entire matrix into memory.
    CSR format is required for efficient cell-by-cell streaming.

    Args:
        adata: AnnData object to validate (may be in backed mode)

    Returns:
        AnnData with .X in CSR format

    Raises:
        AssertionError: If backed mode file is not in CSR or chunked dense format
    """
    from scipy.sparse import issparse, isspmatrix_csr, csr_matrix
    import h5py

    # Check if in backed mode (file-backed, not loaded into memory)
    is_backed = hasattr(adata, "isbacked") and adata.isbacked

    if is_backed:
        logger.info("  File is in backed mode (streaming from disk/S3)")

        # For backed mode, check the underlying HDF5 storage format
        # We can't convert formats in backed mode without loading into memory
        h5_group = adata.file["X"]

        # Check if it's stored as CSR sparse matrix
        if "data" in h5_group and "indices" in h5_group and "indptr" in h5_group:
            logger.info("  ✅ .X is stored as CSR sparse matrix in HDF5")
        # Check if it's dense array with chunking (acceptable for streaming)
        elif isinstance(h5_group, h5py.Dataset):
            chunks = h5_group.chunks
            if chunks is not None:
                logger.info(
                    "  ✅ .X is stored as chunked dense array (chunk size: %s)", chunks
                )
            else:
                # Dense array without chunking - will load entire matrix on first access!
                assert False, (
                    "Backed mode requires .X to be stored as CSR sparse matrix or "
                    "chunked dense array for efficient random access. "
                    "Current file has unchunked dense storage which would load "
                    "the entire matrix into memory. "
                    "Please convert the h5ad file to CSR format before uploading to S3."
                )
        else:
            assert False, f"Unknown .X storage format in backed mode: {type(h5_group)}"

        # Cannot convert format in backed mode
        return adata
    else:
        # In-memory mode: can convert formats as needed
        if not issparse(adata.X):
            logger.warning("  .X is dense array, converting to CSR for efficiency")
            adata.X = csr_matrix(adata.X)
        elif not isspmatrix_csr(adata.X):
            current_format = type(adata.X).__name__
            logger.warning(
                "  .X is in %s format, converting to CSR for efficient row access",
                current_format,
            )
            adata.X = adata.X.tocsr()
        else:
            logger.info("  ✅ .X is already in CSR format")

        return adata


def _random_sample_cells(adata: sc.AnnData, max_cells: int, seed: int) -> sc.AnnData:
    """
    Randomly sample cells from AnnData object.

    Uses stratified sampling when possible to preserve label distributions.

    Args:
        adata: AnnData object to sample from
        max_cells: Maximum number of cells to keep
        seed: Random seed for reproducibility

    Returns:
        Sampled AnnData object
    """
    n_cells = adata.n_obs

    assert max_cells > 0, f"max_cells must be positive, got {max_cells}"
    assert (
        max_cells <= n_cells
    ), f"max_cells ({max_cells}) exceeds dataset size ({n_cells})"

    logger.info("Randomly sampling %d cells from %d total", max_cells, n_cells)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Random sample without replacement
    indices = np.random.choice(n_cells, size=max_cells, replace=False)
    indices.sort()  # Sort for consistent ordering

    sampled = adata[indices, :].copy()
    logger.info("  Sample indices: min=%d, max=%d", indices.min(), indices.max())

    return sampled


def _load_model_genes(genes_path: Path) -> list[str]:
    """
    Load model gene vocabulary from genes.txt.

    Args:
        genes_path: Path to genes.txt file (one gene per line)

    Returns:
        List of gene names in model order
    """
    with open(genes_path, "r") as f:
        genes = [line.strip() for line in f if line.strip()]

    logger.info("Loaded %d genes from %s", len(genes), genes_path)
    return genes


def _create_inflation_indices(
    sample_genes: list[str], model_genes: list[str]
) -> np.ndarray:
    """
    Create inflation index mapping from sample genes to model genes.

    Python port of worker.ts precomputeInflationIndices().

    Args:
        sample_genes: Gene names from h5ad file
        model_genes: Gene names from model genes.txt

    Returns:
        Array of shape [n_sample_genes] where:
        - inflation_indices[i] = j means sample gene i maps to model gene j
        - inflation_indices[i] = -1 means sample gene i not in model
    """
    # Create lookup dict for O(1) access
    model_gene_to_idx = {gene: idx for idx, gene in enumerate(model_genes)}

    inflation_indices = np.array(
        [model_gene_to_idx.get(gene, -1) for gene in sample_genes], dtype=np.int32
    )

    n_missing = np.sum(inflation_indices == -1)
    n_present = len(inflation_indices) - n_missing

    logger.info(
        "Gene mapping: %d/%d genes present in model", n_present, len(sample_genes)
    )
    logger.info("  Missing genes: %d", n_missing)

    assert n_present > 0, "No genes from sample found in model vocabulary!"

    return inflation_indices


def _inflate_batch_to_model_space(
    batch_data: np.ndarray, inflation_indices: np.ndarray, n_model_genes: int
) -> np.ndarray:
    """
    Inflate batch from sample gene space to model gene space.

    Python port of worker.ts fillBatchData().

    Args:
        batch_data: Expression data in sample gene space [batch_size, n_sample_genes]
        inflation_indices: Mapping from sample to model genes
        n_model_genes: Number of genes in model

    Returns:
        Expression data in model gene space [batch_size, n_model_genes]
    """
    batch_size = batch_data.shape[0]

    # Initialize with zeros
    inflated = np.zeros((batch_size, n_model_genes), dtype=np.float32)

    # Map non-zero values to correct positions
    for sample_idx in range(len(inflation_indices)):
        model_idx = inflation_indices[sample_idx]
        if model_idx != -1:
            inflated[:, model_idx] = batch_data[:, sample_idx]

    return inflated


def _extract_raw_counts_batch(
    adata: sc.AnnData, start_idx: int, batch_size: int
) -> np.ndarray:
    """
    Extract raw counts for a batch of cells, handling sparse/dense formats.

    Args:
        adata: AnnData object with raw counts
        start_idx: Starting cell index
        batch_size: Number of cells to extract

    Returns:
        Dense array [batch_size, n_genes] with raw counts
    """
    end_idx = min(start_idx + batch_size, adata.n_obs)

    # Extract batch (handles both sparse and dense)
    batch = adata.X[start_idx:end_idx]

    # Convert to dense if sparse
    if hasattr(batch, "toarray"):
        batch = batch.toarray()

    return batch.astype(np.float32)


def _compute_embeddings_onnx(
    adata: sc.AnnData,
    onnx_session: ort.InferenceSession,
    inflation_indices: np.ndarray,
    n_model_genes: int,
    batch_size: int = 1000,
) -> np.ndarray:
    """
    Compute embeddings using ONNX model with gene inflation.

    Args:
        adata: AnnData object with raw counts in .X
        onnx_session: ONNX Runtime session for model.onnx (combined model)
        inflation_indices: Gene mapping from sample to model space
        n_model_genes: Number of genes in model
        batch_size: Batch size for inference

    Returns:
        Embeddings array [n_cells, embedding_dim]
    """
    n_cells = adata.n_obs
    embeddings_list = []

    logger.info(
        "Computing embeddings for %d cells in batches of %d", n_cells, batch_size
    )

    # Get input/output names from ONNX model
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    for start_idx in tqdm(range(0, n_cells, batch_size), desc="Embedding batches"):
        # Extract raw counts batch
        batch_counts = _extract_raw_counts_batch(adata, start_idx, batch_size)

        # Inflate to model gene space
        batch_inflated = _inflate_batch_to_model_space(
            batch_counts, inflation_indices, n_model_genes
        )

        # Run ONNX inference
        # Note: model.onnx does preprocessing internally (normalize + log1p)
        batch_embeddings = onnx_session.run(
            [output_name], {input_name: batch_inflated}
        )[0]

        embeddings_list.append(batch_embeddings)

    # Concatenate all batches
    embeddings: np.ndarray = np.concatenate(
        [np.asarray(e) for e in embeddings_list], axis=0
    )

    logger.info("Generated embeddings: %s", embeddings.shape)
    return embeddings.astype(np.float32)


# Create Typer app
app = typer.Typer(
    help="Ingest h5ad file as reference by processing through SCimilarity",
    add_completion=False,
)


def _validate_outputs(
    h5ad_path: Union[str, Path],
    output_dir: Path,
    onnx_model_path: Path,
    genes_path: Path,
    n_samples: int,
    batch_size: int,
    max_cells: int | None = None,
    seed: int = 42,
    s3_profile: str | None = None,
    s3_anon: bool = False,
) -> None:
    """
    Validate that output files maintain correct row ordering and embeddings match.

    This function:
    1. Loads h5ad file and applies same sampling as during export (if any)
    2. Verifies labels match in labels.parquet
    3. Verifies embeddings match when regenerated through ONNX model

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Directory containing embeddings.npy and labels.parquet
        onnx_model_path: Path to ONNX model
        genes_path: Path to genes.txt
        n_samples: Number of random samples to validate
        batch_size: Batch size for ONNX inference
        max_cells: If specified, must match the sampling used during export
        seed: Random seed (must match the seed used during export)
    """

    logger.info("Validating output files against h5ad: %s", h5ad_path)

    # Load the h5ad file with SAME sampling as export (supports S3)
    # This uses incremental access - samples BEFORE loading if max_cells specified
    logger.info("Loading h5ad file with same sampling as export...")
    adata = _load_h5ad_from_path_or_s3(
        h5ad_path,
        max_cells=max_cells,
        seed=seed,
        s3_profile=s3_profile,
        s3_anon=s3_anon,
    )
    logger.info("  Loaded shape: %s", adata.shape)

    # Validate and convert to CSR (same as during export)
    adata = _validate_and_convert_to_csr(adata)

    # Load output files
    embeddings_path = output_dir / "embeddings.npy"
    labels_path = output_dir / "labels.parquet"

    assert embeddings_path.exists(), f"Embeddings file not found: {embeddings_path}"

    logger.info("Loading embeddings from: %s", embeddings_path)
    embeddings_array = np.load(embeddings_path)
    logger.info("  Embeddings shape: %s", embeddings_array.shape)

    # Check if labels file exists
    has_labels = labels_path.exists()
    if has_labels:
        logger.info("Loading labels from: %s", labels_path)
        labels_df = pd.read_parquet(labels_path)
        logger.info("  Labels shape: %s", labels_df.shape)
        logger.info("  Label columns: %s", list(labels_df.columns))

    # Select random indices to validate
    n_cells = min(adata.n_obs, embeddings_array.shape[0])
    n_samples = min(n_samples, n_cells)
    random_indices = np.random.choice(n_cells, size=n_samples, replace=False)
    random_indices.sort()

    logger.info("Validating %d random samples: %s", n_samples, random_indices)

    # Validate labels (if present)
    if has_labels:
        logger.info("=== Validating Labels ===")
        all_labels_match = True

        for idx in random_indices:
            logger.info("Row %d:", idx)

            # Check each label column
            for col in labels_df.columns:
                if col in adata.obs.columns:
                    h5ad_value = adata.obs.iloc[idx][col]
                    parquet_value = labels_df.iloc[idx][col]

                    match = str(h5ad_value) == str(parquet_value)
                    status = "✅" if match else "❌"
                    logger.info(
                        "  %s: %s h5ad='%s' parquet='%s'",
                        col,
                        status,
                        h5ad_value,
                        parquet_value,
                    )

                    if not match:
                        all_labels_match = False

        if all_labels_match:
            logger.info("✅ All labels match!")
        else:
            logger.info("❌ Some labels do not match!")

    # Validate embeddings
    logger.info("=== Validating Embeddings ===")

    # Load ONNX model
    logger.info("Loading ONNX model from: %s", onnx_model_path)
    onnx_session = ort.InferenceSession(str(onnx_model_path))

    logger.info("Loading model genes from: %s", genes_path)
    model_genes = _load_model_genes(genes_path)

    # Process subset through ONNX
    logger.info("Processing subset through ONNX model...")
    subset_adata = adata[random_indices, :].copy()

    # Create gene inflation mapping
    inflation_indices = _create_inflation_indices(
        subset_adata.var_names.tolist(), model_genes
    )

    # Generate embeddings via ONNX
    subset_embeddings = _compute_embeddings_onnx(
        subset_adata, onnx_session, inflation_indices, len(model_genes), batch_size
    )

    # Compare embeddings
    all_embeddings_match = True
    tolerance = 1e-5

    for i, idx in enumerate(random_indices):
        regenerated = subset_embeddings[i]
        stored = embeddings_array[idx]

        # Calculate max absolute difference
        max_diff = np.max(np.abs(regenerated - stored))
        match = max_diff < tolerance

        status = "✅" if match else "❌"
        logger.info("Row %d: %s max_diff=%.2e", idx, status, max_diff)

        if not match:
            all_embeddings_match = False
            # Show first few values for debugging
            logger.info("  Regenerated[:5]: %s", regenerated[:5])
            logger.info("  Stored[:5]: %s", stored[:5])

    if all_embeddings_match:
        logger.info("✅ All embeddings match within tolerance (%s)!", tolerance)
    else:
        logger.info("❌ Some embeddings do not match within tolerance (%s)!", tolerance)

    # Summary
    logger.info("=== Validation Summary ===")
    if has_labels:
        logger.info("Labels: %s", "✅ PASS" if all_labels_match else "❌ FAIL")
    logger.info("Embeddings: %s", "✅ PASS" if all_embeddings_match else "❌ FAIL")

    if (not has_labels or all_labels_match) and all_embeddings_match:
        logger.info("✅ Validation PASSED!")
    else:
        logger.info("❌ Validation FAILED!")
        raise typer.Exit(1)


def _compute_label_intersection(
    h5ad_paths: list[Union[str, Path]],
    requested_labels: list[str],
    s3_profile: str | None = None,
    s3_anon: bool = False,
) -> list[str]:
    """
    Compute intersection of label columns across multiple h5ad files.

    Args:
        h5ad_paths: List of paths to h5ad files
        requested_labels: User-requested label columns
        s3_profile: AWS profile name for S3 access (optional)
        s3_anon: Use anonymous access for public S3 buckets

    Returns:
        List of label columns present in ALL files
    """
    if not h5ad_paths:
        return []

    logger.info("Computing label intersection across %d files...", len(h5ad_paths))

    # Load obs columns from each file
    all_obs_columns = []
    for h5ad_path in h5ad_paths:
        adata = _load_h5ad_from_path_or_s3(
            h5ad_path, s3_profile=s3_profile, s3_anon=s3_anon
        )
        obs_columns = set(adata.obs.columns)
        all_obs_columns.append(obs_columns)
        logger.info("  %s: %d obs columns", Path(str(h5ad_path)).name, len(obs_columns))

    # Find intersection
    common_columns = set.intersection(*all_obs_columns)
    logger.info("  Common columns across all files: %d", len(common_columns))

    # Intersect with requested labels
    valid_labels = [label for label in requested_labels if label in common_columns]

    # Warn about missing labels
    for label in requested_labels:
        if label not in common_columns:
            logger.warning("Label '%s' not found in all files, will be excluded", label)

    logger.info("  ✅ Valid labels across all files: %s", valid_labels)
    return valid_labels


def _process_single_h5ad_file(
    h5ad_path: Union[str, Path],
    onnx_session: ort.InferenceSession,
    model_genes: list[str],
    labels: list[str],
    max_cells: int | None,
    seed: int,
    batch_size: int,
    s3_profile: str | None = None,
    s3_anon: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """
    Process a single h5ad file to generate embeddings and extract labels.

    Args:
        h5ad_path: Path or S3 URI to h5ad file
        onnx_session: ONNX Runtime inference session
        model_genes: List of model gene names
        labels: Label columns to extract
        max_cells: Maximum cells to sample (None for all)
        seed: Random seed for sampling
        batch_size: Batch size for embedding inference

    Returns:
        Tuple of (embeddings array, labels dataframe or None)
    """
    # Load h5ad file
    # For S3 files, sampling happens during load for efficiency
    # For local files, sampling happens after load
    adata = _load_h5ad_from_path_or_s3(
        h5ad_path,
        max_cells=max_cells,
        seed=seed,
        s3_profile=s3_profile,
        s3_anon=s3_anon,
    )
    logger.info("  Loaded data shape: %s", adata.shape)

    # Validate and convert to CSR
    adata = _validate_and_convert_to_csr(adata)

    # For local files in backed mode, we still need to sample after loading
    # For S3 files, sampling already happened during load
    if max_cells is not None and adata.n_obs > max_cells:
        logger.info("  Sampling after load (backed mode)")
        adata = _random_sample_cells(adata, max_cells, seed)
        logger.info("  Sampled data shape: %s", adata.shape)

    # Create gene inflation mapping
    inflation_indices = _create_inflation_indices(adata.var_names.tolist(), model_genes)

    # Compute embeddings
    embeddings = _compute_embeddings_onnx(
        adata, onnx_session, inflation_indices, len(model_genes), batch_size
    )

    # Extract labels if specified
    labels_df = None
    if labels:
        labels_df = pd.DataFrame()
        for label in labels:
            if label in adata.obs.columns:
                labels_df[label] = adata.obs[label].values
                labels_df[label] = labels_df[label].astype("category")

    return embeddings, labels_df


def _process_multiple_h5ad_files(
    h5ad_paths: list[Union[str, Path]],
    onnx_session: ort.InferenceSession,
    model_genes: list[str],
    requested_labels: list[str],
    max_cells: int | None,
    seed: int,
    batch_size: int,
    s3_profile: str | None = None,
    s3_anon: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """
    Process multiple h5ad files with label intersection.

    Args:
        h5ad_paths: List of paths or S3 URIs to h5ad files
        onnx_session: ONNX Runtime inference session
        model_genes: List of model gene names
        requested_labels: User-requested label columns
        max_cells: Maximum cells per file (None for all)
        seed: Base random seed
        batch_size: Batch size for embedding inference

    Returns:
        Tuple of (concatenated embeddings, concatenated labels or None)
    """
    # Compute label intersection
    valid_labels = _compute_label_intersection(
        h5ad_paths, requested_labels, s3_profile=s3_profile, s3_anon=s3_anon
    )

    # Process each file
    all_embeddings: list[np.ndarray] = []
    all_labels: list[pd.DataFrame] | None = [] if valid_labels else None

    for i, h5ad_path in enumerate(h5ad_paths):
        logger.info("Processing file %d/%d: %s", i + 1, len(h5ad_paths), h5ad_path)

        # Use per-file seed for reproducibility
        file_seed = seed + i

        embeddings, labels_df = _process_single_h5ad_file(
            h5ad_path=h5ad_path,
            onnx_session=onnx_session,
            model_genes=model_genes,
            labels=valid_labels,
            max_cells=max_cells,
            seed=file_seed,
            batch_size=batch_size,
            s3_profile=s3_profile,
            s3_anon=s3_anon,
        )

        all_embeddings.append(embeddings)
        if all_labels is not None and labels_df is not None:
            all_labels.append(labels_df)

        logger.info("  ✅ File %d complete: %d cells processed", i + 1, len(embeddings))

    # Concatenate results
    logger.info("Concatenating results from %d files...", len(h5ad_paths))
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    final_labels = None
    if all_labels:
        final_labels = pd.concat(all_labels, axis=0, ignore_index=True)

    logger.info("  Total cells: %d", len(final_embeddings))

    return final_embeddings, final_labels


@app.command()
def inspect(
    h5ad_path: str = typer.Argument(
        help="Path or S3 URI to h5ad file to inspect",
    ),
    s3_profile: str = typer.Option(
        None,
        help="AWS profile name to use for S3 access",
    ),
    s3_anon: bool = typer.Option(
        False,
        help="Access S3 bucket anonymously without credentials",
    ),
) -> None:
    """
    Inspect h5ad file structure and metadata without processing.

    Shows file format details including:
    - Number of cells and genes
    - X matrix storage format (dense/sparse, CSR/CSC, chunked)
    - Available obs columns (cell metadata)
    - File size estimates
    - S3 incremental access compatibility
    """
    import h5py

    logger.info("Inspecting h5ad file: %s", h5ad_path)

    # Open file (local or S3)
    if h5ad_path.startswith("s3://"):
        logger.info("Opening S3 file...")

        # Build S3FileSystem configuration
        s3_config: dict[str, Any] = {
            "anon": s3_anon,
            "skip_instance_cache": True,
            "use_listings_cache": False,
        }

        if s3_profile is not None:
            logger.info("  Using AWS profile: %s", s3_profile)
            s3_config["profile"] = s3_profile

        endpoint_url = os.getenv("AWS_ENDPOINT_URL", None)
        if endpoint_url:
            logger.info("  Using custom S3 endpoint: %s", endpoint_url)
            s3_config["client_kwargs"] = {"endpoint_url": endpoint_url}

        if s3_anon:
            logger.info("  Using anonymous S3 access")

        fs = S3FileSystem(**s3_config)
        s3_path = h5ad_path.replace("s3://", "")

        try:
            s3_file = fs.open(s3_path, "rb")
            h5_file = h5py.File(s3_file, "r")
            logger.info("  Successfully opened S3 file")
        except Exception as e:
            logger.error("  Failed to open S3 file: %s", e)
            raise typer.Exit(1)
    else:
        logger.info("Opening local file...")
        try:
            h5_file = h5py.File(h5ad_path, "r")
            logger.info("  Successfully opened local file")
        except Exception as e:
            logger.error("  Failed to open file: %s", e)
            raise typer.Exit(1)

    # Print file structure
    logger.info("=" * 60)
    logger.info("FILE STRUCTURE")
    logger.info("=" * 60)

    # Get number of cells and genes
    obs_group = h5_file["obs"]
    # Determine obs index key
    if "_index" in obs_group:
        obs_index_key = "_index"
    elif "index" in obs_group:
        obs_index_key = "index"
    else:
        logger.error("Could not find obs index (expected '_index' or 'index')")
        logger.error("Available obs keys: %s", list(obs_group.keys()))
        h5_file.close()
        raise typer.Exit(1)

    n_cells = obs_group[obs_index_key].shape[0]

    var_group = h5_file["var"]
    # Determine var index key - try common alternatives
    var_index_key = None
    for possible_key in ["_index", "index", "row_names", "gene_names", "var_names"]:
        if possible_key in var_group:
            var_index_key = possible_key
            break

    if var_index_key is None:
        logger.error("Could not find var index")
        logger.error("Available var keys: %s", list(var_group.keys()))
        h5_file.close()
        raise typer.Exit(1)

    n_genes = var_group[var_index_key].shape[0]

    logger.info("Dimensions:")
    logger.info("  Cells: %d", n_cells)
    logger.info("  Genes: %d", n_genes)

    # Analyze X matrix storage
    logger.info("\nX Matrix Storage:")
    x_group = h5_file["X"]

    # Check if CSR sparse matrix
    is_csr = (
        isinstance(x_group, h5py.Group)
        and "data" in x_group
        and "indices" in x_group
        and "indptr" in x_group
    )

    if is_csr:
        logger.info("  Format: CSR Sparse Matrix")
        data_size = x_group["data"].size * x_group["data"].dtype.itemsize
        indices_size = x_group["indices"].size * x_group["indices"].dtype.itemsize
        indptr_size = x_group["indptr"].size * x_group["indptr"].dtype.itemsize
        total_size_mb = (data_size + indices_size + indptr_size) / (1024**2)
        logger.info("  Data elements: %d", x_group["data"].size)
        logger.info("  Sparsity: %.2f%%", 100 * (1 - x_group["data"].size / (n_cells * n_genes)))
        logger.info("  Storage size: ~%.2f MB", total_size_mb)
        logger.info("  S3 Compatibility: ✅ Efficient (row-wise access)")
    elif isinstance(x_group, h5py.Dataset):
        logger.info("  Format: Dense Array")
        logger.info("  Shape: %s", x_group.shape)
        logger.info("  Dtype: %s", x_group.dtype)

        chunks = x_group.chunks
        if chunks is not None:
            logger.info("  Chunking: ✅ Enabled")
            logger.info("    Chunk size: %s", chunks)

            chunk_rows, chunk_cols = chunks
            chunk_bytes = chunk_rows * chunk_cols * x_group.dtype.itemsize
            chunk_kb = chunk_bytes / 1024

            # Analyze chunking strategy
            logger.info("    Chunk dimensions: %d cells × %d genes", chunk_rows, chunk_cols)
            logger.info("    Bytes per chunk: %.1f KB", chunk_kb)

            # Determine if optimized for cell (row) or gene (column) access
            if chunk_cols == x_group.shape[1]:
                # Full rows - optimized for cell access
                logger.info("    Strategy: Row-oriented (cell-optimized)")
                logger.info("    Chunks per cell read: 1")
                logger.info("    S3 Compatibility: ✅ Excellent (cell access)")
            elif chunk_rows == x_group.shape[0]:
                # Full columns - optimized for gene access (bad for cells!)
                logger.info("    Strategy: Column-oriented (gene-optimized)")
                chunks_per_cell = int(np.ceil(x_group.shape[1] / chunk_cols))
                data_per_cell_kb = chunks_per_cell * chunk_kb
                logger.info("    Chunks per cell read: %d", chunks_per_cell)
                logger.info("    Data transfer per cell: %.1f KB", data_per_cell_kb)
                logger.info("    S3 Compatibility: ❌ Very inefficient (cell access)")
            else:
                # Balanced chunking
                logger.info("    Strategy: Balanced")
                chunks_per_cell = int(np.ceil(x_group.shape[1] / chunk_cols))
                data_per_cell_kb = chunks_per_cell * chunk_kb
                logger.info("    Chunks per cell read: ~%d", chunks_per_cell)
                logger.info("    Data transfer per cell: ~%.1f KB", data_per_cell_kb)

                if chunk_rows <= 100 and data_per_cell_kb < 1024:
                    logger.info("    S3 Compatibility: ✅ Good (reasonable cell access)")
                else:
                    logger.info("    S3 Compatibility: ⚠️  Suboptimal (high overhead per cell)")
        else:
            logger.info("  Chunking: ❌ Not chunked")
            size_gb = (x_group.shape[0] * x_group.shape[1] * x_group.dtype.itemsize) / (1024**3)
            logger.info("  Storage size: ~%.2f GB", size_gb)

            # Check HDF5 storage layout
            plist = x_group.id.get_create_plist()
            layout = plist.get_layout()
            layout_names = {0: "COMPACT", 1: "CONTIGUOUS", 2: "CHUNKED"}
            logger.info("  HDF5 Layout: %s", layout_names.get(layout, "UNKNOWN"))

            if layout == 1:  # H5D_CONTIGUOUS
                logger.info("  Storage: Row-major (C-contiguous)")
                logger.info("  S3 Compatibility: ✅ Efficient (contiguous row access)")
                logger.info("")
                logger.info("  📊 Access characteristics:")
                bytes_per_row = x_group.shape[1] * x_group.dtype.itemsize
                kb_per_row = bytes_per_row / 1024
                logger.info("    Data per cell: ~%.1f KB", kb_per_row)
                logger.info("    Sequential cell reads are efficient over S3")
                logger.info("    Random cell access requires seeking in ~%.2f GB file", size_gb)
            else:
                logger.info("  S3 Compatibility: ⚠️  Unknown (unexpected layout)")

            # Optionally suggest chunking for even better S3 performance
            if size_gb > 1.0 and layout == 1:
                logger.info("")
                logger.info("  💡 Optional: Add chunking for better S3 random access:")
                target_chunk_kb = 100
                target_chunk_bytes = target_chunk_kb * 1024
                bytes_per_row = x_group.shape[1] * x_group.dtype.itemsize
                optimal_chunk_rows = max(1, int(target_chunk_bytes / bytes_per_row))
                logger.info("    chunks=(%d, %d)  # %d cells per chunk, ~%.1f KB per chunk",
                           optimal_chunk_rows, x_group.shape[1],
                           optimal_chunk_rows,
                           (optimal_chunk_rows * x_group.shape[1] * x_group.dtype.itemsize) / 1024)
                logger.info("    (Chunking enables efficient random cell access patterns)")
    else:
        logger.info("  Format: Unknown (%s)", type(x_group))
        logger.info("  S3 Compatibility: ❌ Unknown")

    # List obs columns
    logger.info("\nObs Columns (Cell Metadata):")
    obs_columns = []
    for key in sorted(obs_group.keys()):
        if key not in (obs_index_key, "_index", "index"):
            item = obs_group[key]
            if isinstance(item, h5py.Dataset):
                dtype_str = str(item.dtype)
                obs_columns.append((key, dtype_str, "regular"))
            elif isinstance(item, h5py.Group) and "codes" in item and "categories" in item:
                n_categories = item["categories"].shape[0]
                obs_columns.append((key, f"categorical ({n_categories} levels)", "categorical"))
            else:
                obs_columns.append((key, str(type(item)), "unknown"))

    if obs_columns:
        max_name_len = max(len(name) for name, _, _ in obs_columns)
        for name, dtype, _ in obs_columns:
            logger.info("  %-*s  %s", max_name_len, name, dtype)
    else:
        logger.info("  (no obs columns)")

    # Estimate obs metadata size
    obs_size_bytes = 0
    for key in obs_group.keys():
        item = obs_group[key]
        if isinstance(item, h5py.Dataset):
            obs_size_bytes += item.size * item.dtype.itemsize
        elif isinstance(item, h5py.Group) and "codes" in item:
            obs_size_bytes += item["codes"].size * item["codes"].dtype.itemsize
            obs_size_bytes += item["categories"].size * item["categories"].dtype.itemsize

    obs_size_mb = obs_size_bytes / (1024**2)
    logger.info("\nObs Metadata Size: ~%.2f MB", obs_size_mb)
    if obs_size_mb > 100:
        logger.info("  ⚠️  Large metadata may slow S3 access")

    # Run S3 validation if it's an S3 file
    if h5ad_path.startswith("s3://"):
        logger.info("\n" + "=" * 60)
        logger.info("S3 INCREMENTAL ACCESS VALIDATION")
        logger.info("=" * 60)
        try:
            _validate_s3_incremental_access(h5_file)
            logger.info("✅ File is suitable for S3 incremental access")
        except ValueError as e:
            logger.error("❌ File is NOT suitable for S3 incremental access")
            logger.error("\n%s", str(e))
            h5_file.close()
            raise typer.Exit(1)

    h5_file.close()
    logger.info("\n✅ Inspection complete")


@app.command()
def ingest(
    h5ad_paths: list[str] = typer.Argument(
        help="Path(s) or S3 URI(s) to h5ad file(s). Multiple files will be processed and concatenated.",
    ),
    onnx_model_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to ONNX embedding model file (model.onnx)",
    ),
    genes_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to model genes vocabulary file (genes.txt)",
    ),
    output_path: Path = typer.Argument(
        file_okay=False,
        dir_okay=True,
        help="Output directory path",
    ),
    labels: list[str] = typer.Option(
        [],
        help="Label columns from adata.obs to extract (e.g., --labels cell_type --labels tissue)",
    ),
    max_cells: int = typer.Option(
        None,
        help="Maximum number of cells per file to sample (default: use all cells)",
    ),
    validate: bool = typer.Option(
        True,
        help="Validate the output files after processing (default: True)",
    ),
    seed: int = typer.Option(
        42,
        help="Random seed for reproducibility",
    ),
    batch_size: int = typer.Option(
        128,
        help="Batch size for embedding inference",
    ),
    s3_profile: str = typer.Option(
        None,
        help="AWS profile name to use for S3 access (default: use default profile or environment credentials)",
    ),
    s3_anon: bool = typer.Option(
        False,
        help="Access S3 bucket anonymously without credentials (for public buckets)",
    ),
) -> None:
    """
    Process h5ad file(s) through ONNX embedding model to generate embeddings and extract labels.

    Supports both local files and S3 URIs (s3://bucket/key). Multiple files will be processed
    with label intersection (only labels present in ALL files are extracted).

    This script:
    1. Loads h5ad file(s) (from local path or S3)
    2. Computes label intersection for multi-file processing
    3. Validates/converts to CSR format for efficient processing
    4. Optionally samples random cells with seed
    5. Processes raw counts through ONNX embedding model
    6. Extracts specified label columns from adata.obs
    7. Saves embeddings to embeddings.npy and labels to labels.parquet
    """
    # Load ONNX model and genes (needed for both single and multi-file)
    logger.info("Loading ONNX model from: %s", onnx_model_path)
    onnx_session = ort.InferenceSession(str(onnx_model_path))

    logger.info("Loading model genes from: %s", genes_path)
    model_genes = _load_model_genes(genes_path)

    # Dispatch to single or multi-file processing
    if len(h5ad_paths) == 1:
        logger.info("Processing single h5ad file: %s", h5ad_paths[0])
        embeddings, labels_df = _process_single_h5ad_file(
            h5ad_path=h5ad_paths[0],
            onnx_session=onnx_session,
            model_genes=model_genes,
            labels=labels,
            max_cells=max_cells,
            seed=seed,
            batch_size=batch_size,
            s3_profile=s3_profile,
            s3_anon=s3_anon,
        )
    else:
        logger.info("Processing %d h5ad files...", len(h5ad_paths))
        embeddings, labels_df = _process_multiple_h5ad_files(
            h5ad_paths=h5ad_paths,
            onnx_session=onnx_session,
            model_genes=model_genes,
            requested_labels=labels,
            max_cells=max_cells,
            seed=seed,
            batch_size=batch_size,
            s3_profile=s3_profile,
            s3_anon=s3_anon,
        )

    # Create output directory
    logger.info("Creating output directory: %s", output_path)
    os.makedirs(output_path, exist_ok=True)

    # Save embeddings
    embeddings_path = output_path / "embeddings.npy"
    logger.info("Saving embeddings to: %s", embeddings_path)
    np.save(embeddings_path, embeddings.astype(np.float32))

    # Save labels if available
    labels_path = None
    if labels_df is not None:
        labels_path = output_path / "labels.parquet"
        logger.info("Saving labels to: %s", labels_path)
        labels_df.to_parquet(labels_path, compression="snappy", index=None)
        logger.info("Labels saved with shape: %s", labels_df.shape)

    logger.info("✅ Processing complete!")
    logger.info("   Embeddings: %s (shape: %s)", embeddings_path, embeddings.shape)
    if labels_path:
        logger.info("   Labels: %s (shape: %s)", labels_path, labels_df.shape)
    logger.info("   Output directory: %s", output_path)

    # Run validation if requested (only for single file)
    if validate and len(h5ad_paths) == 1:
        logger.info("=" * 60)
        logger.info("Running validation...")
        logger.info("=" * 60)
        try:
            _validate_outputs(
                h5ad_path=h5ad_paths[0],
                output_dir=output_path,
                onnx_model_path=onnx_model_path,
                genes_path=genes_path,
                n_samples=10,
                batch_size=batch_size,
                max_cells=max_cells,
                seed=seed,
                s3_profile=s3_profile,
                s3_anon=s3_anon,
            )
        except Exception as e:
            logger.error("Validation failed: %s", e)
            raise typer.Exit(1)
    elif validate and len(h5ad_paths) > 1:
        logger.info("Validation skipped for multi-file processing")


if __name__ == "__main__":
    app()
