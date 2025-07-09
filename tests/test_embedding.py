"""
Unit tests for embedding data validation.
"""

import pytest
import anndata
import tiledb
import pandas as pd
import numpy as np
from pathlib import Path

# Add SCimilarity imports for embedding computation
import scimilarity
from scimilarity import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts
import scanpy as sc


@pytest.fixture
def adata():
    """Fixture that loads an h5ad file."""
    data_path = Path("tests/GSE136831_subsample_10.h5ad")
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


@pytest.fixture
def labels_df():
    """Fixture that loads the SCimilarity labels.parquet file."""
    labels_path = Path("data/scimilarity/labels.parquet")
    return pd.read_parquet(labels_path)


@pytest.fixture
def scimilarity_model():
    """Fixture that loads the SCimilarity model for computing embeddings."""
    model_path = Path("data/scimilarity/model_v1.1")
    return CellEmbedding(model_path)


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

    def test_study_in_labels_parquet(self, adata, labels_df):
        """Test that the study ID appears in the exported labels.parquet file."""
        # Get the study ID from the H5AD file
        h5ad_study_id = adata.obs["study"].unique()[0]

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