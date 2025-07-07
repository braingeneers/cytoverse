"""
Unit tests for embedding data validation.
"""

import pytest
import anndata
import tiledb
import pandas as pd
from pathlib import Path


@pytest.fixture
def adata():
    """Fixture that loads the GSE136831 H5AD file."""
    data_path = Path("data/GSE136831_subsample.h5ad")
    return anndata.read_h5ad(data_path)


@pytest.fixture
def metadata_db():
    """Fixture that opens the SCimilarity TileDB metadata database."""
    db_path = "data/scimilarity/model_v1.1/cellsearch/cell_metadata"
    with tiledb.open(db_path, "r") as db:
        yield db


@pytest.fixture
def embeddings_db():
    """Fixture that opens the SCimilarity TileDB embeddings database."""
    db_path = "data/scimilarity/model_v1.1/cellsearch/cell_embedding"
    with tiledb.open(db_path, "r") as db:
        yield db


class TestEmbeddingData:
    """Test cases for embedding data files."""

    def test_study_id(self, adata):
        """Test that GSE136831_subsample.h5ad contains the expected study ID."""
        # Extract unique study IDs
        study_ids = adata.obs["study"].unique()

        # Verify there's exactly one study ID and it matches the expected value
        assert (
            len(study_ids) == 1
        ), f"Expected exactly one study ID, got {len(study_ids)}: {study_ids}"

        # The internal study ID for GSE136831 should be 'DS000011735'
        expected_study_id = "DS000011735"
        actual_study_id = study_ids[0]

        assert (
            actual_study_id == expected_study_id
        ), f"Expected study ID '{expected_study_id}', got '{actual_study_id}'"

    def test_study_in_scimilarity_db(self, adata, metadata_db):
        """Test whether the GSE136831 study ID exists in the SCimilarity database."""
        # Get the study ID from the H5AD file
        h5ad_study_id = adata.obs["study"].unique()[0]

        # Get all studies from the SCimilarity database
        all_studies_df = metadata_db.query(attrs=["study"]).df[:]
        scimilarity_studies = set(all_studies_df["study"].unique())

        # Check if the H5AD study is in the SCimilarity database
        is_present = h5ad_study_id in scimilarity_studies

        if is_present:
            print(f"✅ Study {h5ad_study_id} found in SCimilarity database")
        else:
            print(f"❌ Study {h5ad_study_id} NOT found in SCimilarity database")
            print(f"SCimilarity database contains {len(scimilarity_studies)} studies")

        # Assert that the study is present in the SCimilarity database
        assert (
            is_present
        ), f"GSE136831 study '{h5ad_study_id}' should be present in the SCimilarity database"

    def test_study_in_labels_parquet(self, adata):
        """Test that the study ID appears in the exported labels.parquet file."""
        # Get the study ID from the H5AD file
        h5ad_study_id = adata.obs["study"].unique()[0]

        # Load the labels.parquet file
        labels_path = Path("data/scimilarity/labels.parquet")
        labels_df = pd.read_parquet(labels_path)

        # Check if the study appears in the labels file
        study_present = h5ad_study_id in labels_df["study"].values

        if study_present:
            count = (labels_df["study"] == h5ad_study_id).sum()
            print(f"✅ Study {h5ad_study_id} found {count} times in labels.parquet")
        else:
            print(f"❌ Study {h5ad_study_id} NOT found in labels.parquet")
            available_studies = labels_df["study"].unique()[:10]
            print(f"Available studies sample: {available_studies}")

        # Assert that the study is present
        assert (
            study_present
        ), f"Study '{h5ad_study_id}' should appear at least once in labels.parquet"

    def test_sample_embedding_validation(self, adata, embeddings_db):
        """Test that we can retrieve and validate embeddings for a sample from the H5AD file."""
        # Load the labels.parquet file to find embedding indices
        labels_path = Path("data/scimilarity/labels.parquet")
        labels_df = pd.read_parquet(labels_path)

        # Get the study ID from the H5AD file
        h5ad_study_id = adata.obs["study"].unique()[0]

        # Find a sample that appears in both H5AD and parquet files
        h5ad_samples = adata.obs[adata.obs["study"] == h5ad_study_id]["sample"].unique()
        parquet_samples = labels_df[labels_df["study"] == h5ad_study_id][
            "sample"
        ].unique()
        common_samples = set(h5ad_samples) & set(parquet_samples)

        assert (
            len(common_samples) > 0
        ), "No common samples found between H5AD and parquet files"

        # Use the first common sample
        target_sample = list(common_samples)[0]
        print(f"Testing embeddings for sample: {target_sample}")

        # Get embedding indices for this sample from the parquet file
        sample_indices = labels_df[labels_df["sample"] == target_sample].index.tolist()
        assert (
            len(sample_indices) > 0
        ), f"No embedding indices found for sample {target_sample}"

        # Get the first embedding index for this sample
        first_embedding_index = sample_indices[0]
        print(f"First embedding index: {first_embedding_index}")

        # Retrieve the embedding from TileDB
        embedding = embeddings_db[first_embedding_index]["vals"]

        # Validate the embedding structure
        assert embedding.shape == (
            128,
        ), f"Expected embedding shape (128,), got {embedding.shape}"
        assert (
            embedding.dtype == "float32"
        ), f"Expected float32 dtype, got {embedding.dtype}"

        # Validate the first 5 dimensions
        first_5_dims = embedding[:5]
        print(f"First 5 embedding dimensions: {first_5_dims}")

        # Check that all values are finite (not NaN, inf, or -inf)
        for i, val in enumerate(first_5_dims):
            assert not (val != val), f"Dimension {i} is NaN: {val}"  # NaN check
            assert val != float("inf"), f"Dimension {i} is +inf: {val}"
            assert val != float("-inf"), f"Dimension {i} is -inf: {val}"

        # Check that values are reasonable (between -10 and 10 for typical embeddings)
        for i, val in enumerate(first_5_dims):
            assert (
                -10.0 <= val <= 10.0
            ), f"Dimension {i} value {val} is outside reasonable range [-10, 10]"

        print(f"✅ Successfully validated embedding for sample {target_sample}")
        print(f"   Embedding index: {first_embedding_index}")
        print(f"   Shape: {embedding.shape}")
        print(f"   First 5 dimensions: {first_5_dims}")
