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

ADATA_STUDY_ID = "DS000011735"  # Expected study ID for GSE136831


@pytest.fixture
def adata():
    """Fixture that loads an h5ad file."""
    # data_path = Path("data/GSE136831_subsample.h5ad")
    data_path = Path("data/Pancreas_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad")
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

    def test_some_adata_in_labels(self, adata, embeddings_db, labels_df):
        """Test that we can retrieve and validate embeddings for a sample from the H5AD file."""
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

    def test_embedding_vs_stored(
        self, adata, metadata_db, labels_df, scimilarity_model
    ):
        """Test that computed embeddings from H5AD match stored embeddings in TileDB database."""
        print("\n🔬 Testing embedding computation vs stored embeddings...")

        # Get the study ID from the H5AD file
        h5ad_study_id = adata.obs["study"].unique()[0]
        print(f"Working with study: {h5ad_study_id}")

        # Find samples that appear in both H5AD and labels (i.e. in IVFPQ)
        h5ad_samples = adata.obs[adata.obs["study"] == h5ad_study_id]["sample"].unique()
        parquet_samples = labels_df[labels_df["study"] == h5ad_study_id][
            "sample"
        ].unique()
        common_samples = set(h5ad_samples) & set(parquet_samples)
        print(f"Common samples found: {len(common_samples)}")
        assert (
            len(common_samples) > 0
        ), "No common samples found between H5AD and parquet files"

        # Pick the first common sample for testing
        test_sample = list(common_samples)[0]
        print(f"Testing sample: {test_sample}")

        # Get the cell(s) from H5AD file for this sample
        sample_mask = adata.obs["sample"] == test_sample
        sample_cells = adata[sample_mask].copy()
        print(f"Found {sample_cells.n_obs} cells for sample {test_sample} in H5AD")

        # Take only the first cell to keep test fast and focused
        if sample_cells.n_obs > 1:
            sample_cells = sample_cells[0:1].copy()
            print(f"Using first cell for embedding comparison")

        # Step 1: Compute embeddings using SCimilarity get_embeddings
        print("Computing embeddings using SCimilarity...")

        # Align dataset to model gene order (critical for correct embeddings)
        aligned_data = align_dataset(sample_cells, scimilarity_model.gene_order)

        # Apply log normalization (required preprocessing for SCimilarity)
        aligned_data = lognorm_counts(aligned_data)

        # Convert to numpy array format expected by get_embeddings
        if hasattr(aligned_data.X, "toarray"):
            expression_data = aligned_data.X.toarray().astype(np.float32)
        else:
            expression_data = aligned_data.X.astype(np.float32)

        # Compute embeddings using SCimilarity
        computed_embeddings = scimilarity_model.get_embeddings(expression_data)
        print(f"Computed embeddings shape: {computed_embeddings.shape}")

        # Step 2: Get stored embeddings from TileDB database
        print("Retrieving stored embeddings from TileDB...")

        # Find the TileDB indices for our sample
        sample_metadata = labels_df[labels_df["sample"] == test_sample]
        if len(sample_metadata) == 0:
            pytest.skip(f"Sample {test_sample} not found in labels.parquet")
        print("Found matching sample in labels.parquet")

        # Get the TileDB indices (stored in the DataFrame index)
        tiledb_indices = sample_metadata.index.values
        print(
            f"Found {len(tiledb_indices)} matching cells in TileDB for sample {test_sample}"
        )

        # Retrieve stored embeddings for all TileDB indices for this sample
        stored_embeddings = scimilarity.utils.embedding_from_tiledb(
            tiledb_indices, "data/scimilarity/model_v1.1/cellsearch/cell_embedding"
        )
        print(f"Retrieved stored embeddings shape: {stored_embeddings.shape}")

        # Step 3: Compare the computed embedding against all stored embeddings
        print(
            f"Comparing computed embedding vs {len(tiledb_indices)} stored embeddings..."
        )

        # Get the single computed vector
        computed_vector = (
            computed_embeddings[0]
            if computed_embeddings.shape[0] > 1
            else computed_embeddings.flatten()
        )
        computed_norm = np.linalg.norm(computed_vector)

        # Compute cosine distances to all stored embeddings
        cosine_distances = []
        for i, stored_vector in enumerate(stored_embeddings):
            stored_norm = np.linalg.norm(stored_vector)
            cosine_similarity = np.dot(computed_vector, stored_vector) / (
                computed_norm * stored_norm
            )
            cosine_distance = 1.0 - cosine_similarity
            cosine_distances.append(cosine_distance)

        cosine_distances = np.array(cosine_distances)

        # Find the best match (minimum cosine distance)
        best_match_idx = np.argmin(cosine_distances)
        min_cosine_distance = cosine_distances[best_match_idx]
        best_tiledb_idx = tiledb_indices[best_match_idx]

        # Print detailed comparison results
        print(f"📊 Embedding Distance Analysis:")
        print(f"   Total stored embeddings compared: {len(cosine_distances)}")
        print(f"   Minimum cosine distance: {min_cosine_distance:.2e}")
        print(f"   Maximum cosine distance: {np.max(cosine_distances):.6f}")
        print(f"   Mean cosine distance: {np.mean(cosine_distances):.6f}")
        print(f"   Best matching TileDB index: {best_tiledb_idx}")
        print(f"   Best match position: {best_match_idx}/{len(cosine_distances)}")

        # For the best match, compute additional metrics
        best_stored_vector = stored_embeddings[best_match_idx]
        mse = np.mean((computed_vector - best_stored_vector) ** 2)
        max_abs_diff = np.max(np.abs(computed_vector - best_stored_vector))
        l2_distance = np.linalg.norm(computed_vector - best_stored_vector)

        print(f"📊 Best Match Detailed Metrics:")
        print(f"   Cosine similarity: {1.0 - min_cosine_distance:.6f}")
        print(f"   Mean squared error: {mse:.2e}")
        print(f"   Max absolute difference: {max_abs_diff:.2e}")
        print(f"   L2 distance: {l2_distance:.6f}")
        print(f"   Computed vector norm: {computed_norm:.6f}")
        print(f"   Best match vector norm: {np.linalg.norm(best_stored_vector):.6f}")

        # Assert that we found an extremely close match
        # Cosine distance should be extremely small for identical vectors
        assert min_cosine_distance < 1e-6, (
            f"Minimum cosine distance too high: {min_cosine_distance:.2e}. "
            f"Expected to find an identical embedding with cosine distance < 1e-6."
        )

        # MSE should be very small for nearly identical vectors
        assert mse < 1e-6, (
            f"Mean squared error too high: {mse:.2e}. "
            f"Best matching embedding should be nearly identical."
        )

        print(
            "✅ Embedding computation test passed! Found identical match among stored embeddings."
        )

    def test_verify_adata_in_tiledb(
        self, adata, metadata_db, labels_df, scimilarity_model
    ):
        """Test that computed embeddings for all H5AD cells can be matched against stored TileDB embeddings."""
        print("\n🔬 Testing all H5AD cells vs stored TileDB embeddings...")

        # Get the study ID from the H5AD file
        h5ad_study_id = adata.obs["study"].unique()[0]
        print(f"Working with study: {h5ad_study_id}")
        print(f"Total cells in H5AD: {adata.n_obs}")

        # Step 1: Compute embeddings for all cells in H5AD
        print("Computing embeddings for all H5AD cells...")

        # Align dataset to model gene order (critical for correct embeddings)
        aligned_data = align_dataset(adata, scimilarity_model.gene_order)

        # Apply log normalization (required preprocessing for SCimilarity)
        aligned_data = lognorm_counts(aligned_data)

        # Convert to numpy array format expected by get_embeddings
        if hasattr(aligned_data.X, "toarray"):
            expression_data = aligned_data.X.toarray().astype(np.float32)
        else:
            expression_data = aligned_data.X.astype(np.float32)

        # Compute embeddings using SCimilarity
        computed_embeddings = scimilarity_model.get_embeddings(expression_data)
        print(f"Computed embeddings shape: {computed_embeddings.shape}")

        # Step 2: Get all stored embeddings for this study from TileDB
        print("Retrieving all stored embeddings for this study from TileDB...")

        # Find all TileDB entries for this study
        study_metadata = labels_df[labels_df["study"] == h5ad_study_id]
        if len(study_metadata) == 0:
            pytest.skip(f"Study {h5ad_study_id} not found in labels.parquet")

        print(f"Found {len(study_metadata)} stored cells for study {h5ad_study_id}")

        # Get the TileDB indices for all cells in this study
        tiledb_indices = study_metadata.index.values

        # Retrieve stored embeddings for all TileDB indices for this study
        stored_embeddings = scimilarity.utils.embedding_from_tiledb(
            tiledb_indices, "data/scimilarity/model_v1.1/cellsearch/cell_embedding"
        )
        print(f"Retrieved stored embeddings shape: {stored_embeddings.shape}")

        # Step 3: Find closest matches between computed and stored embeddings
        print(
            f"Computing pairwise distances between {computed_embeddings.shape[0]} computed and {stored_embeddings.shape[0]} stored embeddings..."
        )

        # For efficiency with large datasets, we'll use batch cosine similarity computation
        # Normalize both embedding sets for cosine similarity
        computed_norm = computed_embeddings / np.linalg.norm(
            computed_embeddings, axis=1, keepdims=True
        )
        stored_norm = stored_embeddings / np.linalg.norm(
            stored_embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarities (computed x stored)
        cosine_similarities = np.dot(computed_norm, stored_norm.T)
        cosine_distances = 1.0 - cosine_similarities

        # Find best matches: for each computed embedding, find closest stored embedding
        closest_stored_indices = np.argmin(cosine_distances, axis=1)
        min_distances = np.min(cosine_distances, axis=1)

        # Find best matches: for each stored embedding, find closest computed embedding
        closest_computed_indices = np.argmin(cosine_distances, axis=0)
        min_distances_reverse = np.min(cosine_distances, axis=0)

        # Analyze the results
        print(f"📊 Distance Analysis Results:")
        print(f"   Computed -> Stored matches:")
        print(f"     Minimum distance: {np.min(min_distances):.2e}")
        print(f"     Maximum distance: {np.max(min_distances):.6f}")
        print(f"     Mean distance: {np.mean(min_distances):.6f}")
        print(f"     Std distance: {np.std(min_distances):.6f}")

        print(f"   Stored -> Computed matches:")
        print(f"     Minimum distance: {np.min(min_distances_reverse):.2e}")
        print(f"     Maximum distance: {np.max(min_distances_reverse):.6f}")
        print(f"     Mean distance: {np.mean(min_distances_reverse):.6f}")
        print(f"     Std distance: {np.std(min_distances_reverse):.6f}")

        # Count very close matches (cosine distance < 1e-6)
        very_close_matches = np.sum(min_distances < 1e-6)
        print(
            f"   Very close matches (distance < 1e-6): {very_close_matches}/{len(min_distances)}"
        )

        # Count reasonably close matches (cosine distance < 1e-3)
        close_matches = np.sum(min_distances < 1e-3)
        print(
            f"   Close matches (distance < 1e-3): {close_matches}/{len(min_distances)}"
        )

        # Find mutual closest pairs (where A's closest is B and B's closest is A)
        mutual_pairs = []
        for i, stored_idx in enumerate(closest_stored_indices):
            if closest_computed_indices[stored_idx] == i:
                mutual_pairs.append((i, stored_idx, min_distances[i]))

        print(f"   Mutual closest pairs: {len(mutual_pairs)}")
        if len(mutual_pairs) > 0:
            mutual_distances = [dist for _, _, dist in mutual_pairs]
            print(
                f"     Mutual pair distances - min: {np.min(mutual_distances):.2e}, max: {np.max(mutual_distances):.6f}, mean: {np.mean(mutual_distances):.6f}"
            )

        # Detailed analysis of the best overall match
        best_overall_match = np.unravel_index(
            np.argmin(cosine_distances), cosine_distances.shape
        )
        best_computed_idx, best_stored_idx = best_overall_match
        best_distance = cosine_distances[best_computed_idx, best_stored_idx]

        print(f"📊 Best Overall Match:")
        print(f"   Computed cell index: {best_computed_idx}")
        print(
            f"   Stored cell index: {best_stored_idx} (TileDB index: {tiledb_indices[best_stored_idx]})"
        )
        print(f"   Cosine distance: {best_distance:.2e}")
        print(f"   Cosine similarity: {1.0 - best_distance:.6f}")

        # Additional metrics for the best match
        best_computed_vector = computed_embeddings[best_computed_idx]
        best_stored_vector = stored_embeddings[best_stored_idx]
        mse = np.mean((best_computed_vector - best_stored_vector) ** 2)
        l2_distance = np.linalg.norm(best_computed_vector - best_stored_vector)
        max_abs_diff = np.max(np.abs(best_computed_vector - best_stored_vector))

        print(f"   MSE: {mse:.2e}")
        print(f"   L2 distance: {l2_distance:.6f}")
        print(f"   Max absolute difference: {max_abs_diff:.2e}")

        # Assertions to validate the embedding matching
        # At least one very close match should exist (indicating correct preprocessing)
        assert very_close_matches > 0, (
            f"Expected at least one very close match (distance < 1e-6), but found {very_close_matches}. "
            f"This suggests preprocessing differences between computed and stored embeddings."
        )

        # The best overall match should be extremely close
        assert best_distance < 1e-6, (
            f"Best overall match distance too high: {best_distance:.2e}. "
            f"Expected at least one near-identical match with distance < 1e-6."
        )

        # Most matches should be reasonably close (allowing for some numerical differences)
        close_ratio = close_matches / len(min_distances)
        assert close_ratio > 0.5, (
            f"Only {close_ratio:.2%} of computed embeddings have close matches (distance < 1e-3). "
            f"Expected at least 50% to be close, suggesting systematic differences."
        )

        print("✅ All H5AD cells vs TileDB embeddings test passed!")
        print(
            f"   Found {very_close_matches} very close matches and {close_matches} close matches"
        )
        print(f"   Best match has cosine distance of {best_distance:.2e}")

    def test_find_pancreas_match_in_stored_embeddings(
        self, adata, metadata_db, labels_df, scimilarity_model
    ):
        """Test to find ids of adata expected to be in the SCimilarity training set."""
        print(
            f"Testing SCimilarity training set membership for study: {ADATA_STUDY_ID}"
        )

        # Step 2: Get all stored embeddings for study DS000011735
        print(f"Looking up stored embeddings for study: {ADATA_STUDY_ID}")

        study_metadata = labels_df[labels_df["study"] == ADATA_STUDY_ID]
        if len(study_metadata) == 0:
            pytest.skip(f"Study {ADATA_STUDY_ID} not found in labels.parquet")

        print(f"Found {len(study_metadata)} stored cells for study {ADATA_STUDY_ID}")

        # Get TileDB indices for the target study
        tiledb_indices = study_metadata.index.values

        # Retrieve stored embeddings
        stored_embeddings = scimilarity.utils.embedding_from_tiledb(
            tiledb_indices, "data/scimilarity/model_v1.1/cellsearch/cell_embedding"
        )
        print(f"Retrieved stored embeddings shape: {stored_embeddings.shape}")

        # Step 3: Compute embeddings for Pancreas H5AD cells
        print("Computing embeddings...")

        # Align dataset to model gene order
        aligned_adata = align_dataset(adata, scimilarity_model.gene_order)

        # # Apply log normalization
        # aligned_pancreas = lognorm_counts(aligned_pancreas)
        aligned_adata.X = aligned_adata.layers["raw_counts"].copy()
        sc.pp.normalize_total(aligned_adata, target_sum=1e4)
        sc.pp.log1p(aligned_adata)

        # Convert to numpy array format
        if hasattr(aligned_adata.X, "toarray"):
            expression = aligned_adata.X.toarray().astype(np.float32)
        else:
            expression = aligned_adata.X.astype(np.float32)

        # Compute embeddings
        embeddings = scimilarity_model.get_embeddings(expression)
        print(f"Computed Pancreas embeddings shape: {embeddings.shape}")

        # Step 4: Find matching embeddings by computing pairwise distances
        print(
            f"Searching for matches between {embeddings.shape[0]} Pancreas cells and {stored_embeddings.shape[0]} stored cells..."
        )

        # Normalize embeddings for cosine similarity computation
        pancreas_norm = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        stored_norm = stored_embeddings / np.linalg.norm(
            stored_embeddings, axis=1, keepdims=True
        )

        # Search for matches with early stopping
        match_found = False
        best_distance = float("inf")
        best_pancreas_idx = -1
        best_stored_idx = -1

        # Process in batches to manage memory and allow early stopping
        batch_size = 100
        total_comparisons = 0

        for i in range(0, embeddings.shape[0], batch_size):
            end_i = min(i + batch_size, embeddings.shape[0])
            batch_pancreas = pancreas_norm[i:end_i]

            # Compute cosine similarities for this batch
            cosine_similarities = np.dot(batch_pancreas, stored_norm.T)
            cosine_distances = 1.0 - cosine_similarities

            # Find minimum distance in this batch
            batch_min_dist = np.min(cosine_distances)
            batch_min_pos = np.unravel_index(
                np.argmin(cosine_distances), cosine_distances.shape
            )
            batch_pancreas_idx = i + batch_min_pos[0]
            batch_stored_idx = batch_min_pos[1]

            total_comparisons += batch_pancreas.shape[0] * stored_norm.shape[0]

            # Update best match if this is better
            if batch_min_dist < best_distance:
                best_distance = batch_min_dist
                best_pancreas_idx = batch_pancreas_idx
                best_stored_idx = batch_stored_idx

            print(
                f"Batch {i//batch_size + 1}: Processed cells {i}-{end_i-1}, min distance: {batch_min_dist:.2e}"
            )

            # Early stopping if we find an extremely close match
            if batch_min_dist < 1e-6:
                match_found = True
                print(f"🎯 Found extremely close match! Distance: {batch_min_dist:.2e}")
                break

        # Report results
        print(f"📊 Search Results:")
        print(f"   Total pairwise comparisons: {total_comparisons:,}")
        print(f"   Best match distance: {best_distance:.2e}")
        print(f"   Best Pancreas cell index: {best_pancreas_idx}")
        print(
            f"   Best stored cell index: {best_stored_idx} (TileDB index: {tiledb_indices[best_stored_idx]})"
        )
        print(f"   Cosine similarity: {1.0 - best_distance:.6f}")

        if match_found:
            # Compute additional metrics for the best match
            best_pancreas_vector = embeddings[best_pancreas_idx]
            best_stored_vector = stored_embeddings[best_stored_idx]
            mse = np.mean((best_pancreas_vector - best_stored_vector) ** 2)
            l2_distance = np.linalg.norm(best_pancreas_vector - best_stored_vector)
            max_abs_diff = np.max(np.abs(best_pancreas_vector - best_stored_vector))

            print(f"📊 Best Match Detailed Metrics:")
            print(f"   MSE: {mse:.2e}")
            print(f"   L2 distance: {l2_distance:.6f}")
            print(f"   Max absolute difference: {max_abs_diff:.2e}")
            print(
                f"   Pancreas vector norm: {np.linalg.norm(best_pancreas_vector):.6f}"
            )
            print(f"   Stored vector norm: {np.linalg.norm(best_stored_vector):.6f}")

            print(
                "✅ Found matching embedding between Pancreas H5AD and stored DS000011735 data!"
            )
        else:
            print(
                f"⚠️ No extremely close match found. Best distance: {best_distance:.2e}"
            )

        # Assert that we found a very close match
        assert best_distance < 1e-6, (
            f"Expected to find a matching embedding with distance < 1e-6, "
            f"but best match has distance {best_distance:.2e}. "
            f"This suggests the Pancreas H5AD cells may not be present in the stored DS000011735 embeddings."
        )
