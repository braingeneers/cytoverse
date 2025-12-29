"""
Unit tests for embedding data validation.
"""

import pytest
from pathlib import Path
import anndata
import tiledb
import pandas as pd
import numpy as np
import torch

# Add SCimilarity imports for embedding computation
from scimilarity import CellEmbedding
from scimilarity.utils import align_dataset, lognorm_counts

from ivfpq import IVFPQ


@pytest.fixture
def adata():
    """Fixture that loads an h5ad file."""
    data_path = Path("tests/fixtures/GSE136831_subsample_10.h5ad")
    return anndata.read_h5ad(data_path)


@pytest.fixture
def metadata_db():
    """Fixture that opens the SCimilarity TileDB metadata database."""
    db_path = "data/models/scimilarity/model_v1.1/cellsearch/cell_metadata"
    with tiledb.open(db_path, "r") as db:
        yield db


@pytest.fixture
def embeddings_db():
    """Fixture that opens the SCimilarity TileDB embeddings database."""
    db_path = "data/models/scimilarity/model_v1.1/cellsearch/cell_embedding"
    with tiledb.open(db_path, "r") as db:
        yield db


@pytest.fixture
def labels_df():
    """Fixture that loads the SCimilarity labels.parquet file."""
    labels_path = Path("data/references/scimilarity/labels.parquet")
    return pd.read_parquet(labels_path)


@pytest.fixture
def scimilarity_model():
    """Fixture that loads the SCimilarity model for computing embeddings."""
    model_path = Path("data/models/scimilarity/model_v1.1")
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

    def test_find_train_sample_via_scimilarity(
        self, scimilarity_model, embeddings_db, metadata_db, labels_df
    ):
        """Test computing embeddings and finding nearest neighbors using IVFPQ search."""
        # Load 10 cells from GSE154109.h5ad - was part of the training and reference?
        gse154109_path = Path("tests/fixtures/GSE154109_10.h5ad")
        adata_query = anndata.read_h5ad(gse154109_path)
        adata_query = adata_query[:10].copy()  # Take first 10 cells

        print(f"Loaded {adata_query.n_obs} cells from GSE154109_10.h5ad")

        # Add counts layer if it doesn't exist (required by lognorm_counts)
        if "counts" not in adata_query.layers:
            adata_query.layers["counts"] = adata_query.X.copy()

        # Align dataset to model genes and compute embeddings
        adata_aligned = align_dataset(adata_query, scimilarity_model.gene_order)
        adata_normalized = lognorm_counts(adata_aligned)
        expression_data = adata_normalized.X.toarray().astype(np.float32)
        computed_embeddings = scimilarity_model.get_embeddings(expression_data)
        print(f"Computed embeddings shape: {computed_embeddings.shape}")

        # Load the IVF model for nearest neighbor search
        ivf_model_path = Path("public/models/scimilarity/ivfpq")
        assert ivf_model_path.exists()

        ivf = IVFPQ.load(ivf_model_path)

        # Search for nearest neighbors for the first embedding
        query_embedding = torch.from_numpy(computed_embeddings[0]).float()

        # Perform nearest neighbor search using disk-based partitions
        vector_ids, distances, _ = ivf.search(
            query_embedding,
            model_path=ivf_model_path,
            n_probe=4,  # Search multiple partitions for better accuracy
        )

        print(
            f"Found nearest neighbor: vector_id={vector_ids[0]}, distance={distances[0]:.6f}"
        )

        # Get the embedding of the nearest neighbor from the database
        nearest_vector_id = vector_ids[0]

        # Query the embeddings database for the nearest neighbor
        nearest_embedding_result = embeddings_db.query(attrs=["vals"]).df[
            nearest_vector_id:nearest_vector_id
        ]
        nearest_embedding = nearest_embedding_result["vals"].iloc[0]

        # Verify that the distance computation is reasonable
        # The IVFPQ search should return vectors that are actually close
        computed_distance = np.linalg.norm(computed_embeddings[0] - nearest_embedding)

        print(f"Computed distance to nearest neighbor: {computed_distance:.6f}")
        print(f"IVFPQ reported distance: {distances[0]:.6f}")

        # Validate that we found a meaningful nearest neighbor
        # The computed embedding should be reasonably close to something in the database
        assert distances[0] >= 0, "Distance should be non-negative"
        assert (
            computed_distance < 100.0
        ), "Distance should be reasonable for normalized embeddings"

        # Ensure we got valid results
        assert isinstance(
            vector_ids[0], (int, np.integer)
        ), "Vector ID should be an integer"
        assert isinstance(
            distances[0], (float, np.floating)
        ), "Distance should be a float"

        # See if the metadata on this cell matches that in the h5ad
        neighbor_metadata = metadata_db.query().df[
            nearest_vector_id:nearest_vector_id
        ]
        print("Neighbor metadata")
        print(neighbor_metadata.iloc[0])

        assert neighbor_metadata.iloc[0]["study"] == "GSE154109"
        assert neighbor_metadata.iloc[0]["sample"] == "GSM4664010"
        assert (
            neighbor_metadata.iloc[0]["prediction"]
            == labels_df.iloc[nearest_vector_id].prediction
        )

        print(
            "✅ Successfully computed embeddings and found nearest neighbor via IVFPQ"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
