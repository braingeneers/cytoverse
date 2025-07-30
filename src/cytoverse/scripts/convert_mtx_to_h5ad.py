import scanpy as sc
import pandas as pd
import anndata
import os
import gzip
import shutil
from pathlib import Path

# Define the input folder containing .mtx.gz, barcodes.tsv.gz, and features.tsv.gz files
input_folder = "GSE154109"  # Replace with your folder path
output_file = "GSE154109.h5ad"

# Ensure the input folder exists
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"The folder {input_folder} does not exist.")

# Find all .mtx.gz files in the folder
mtx_files = [f for f in os.listdir(input_folder) if f.endswith(".mtx.gz")]

if not mtx_files:
    raise FileNotFoundError("No .mtx.gz files found in the specified folder.")

# List to store AnnData objects
adata_list = []

# Process each .mtx.gz file
for mtx_file in mtx_files:
    try:
        # Extract sample prefix (e.g., "GSM4663368" from "GSM4663368_matrix.mtx.gz")
        sample_id = mtx_file.split("_matrix.mtx.gz")[0]

        # Define corresponding barcodes and features files
        barcodes_file = os.path.join(input_folder, f"{sample_id}_barcodes.tsv.gz")
        features_file = os.path.join(input_folder, f"{sample_id}_features.tsv.gz")

        # Check if all required files exist
        mtx_path = os.path.join(input_folder, mtx_file)
        if not all([os.path.exists(mtx_path), os.path.exists(barcodes_file), os.path.exists(features_file)]):
            print(f"Skipping {sample_id}: Missing required files")
            continue

        # Create a temporary folder for organizing files
        temp_folder = os.path.join(input_folder, "temp")
        os.makedirs(temp_folder, exist_ok=True)

        # Read 10x data into AnnData directly from compressed files
        # Create a temporary folder structure that scanpy expects
        sample_temp_folder = os.path.join(temp_folder, sample_id)
        os.makedirs(sample_temp_folder, exist_ok=True)
        
        # Copy files to the expected locations
        shutil.copy2(os.path.join(input_folder, mtx_file), os.path.join(sample_temp_folder, "matrix.mtx.gz"))
        shutil.copy2(barcodes_file, os.path.join(sample_temp_folder, "barcodes.tsv.gz"))
        shutil.copy2(features_file, os.path.join(sample_temp_folder, "features.tsv.gz"))
        
        # Read 10x data into AnnData
        adata = sc.read_10x_mtx(
            sample_temp_folder,
            var_names="gene_symbols",  # Use gene symbols from features.tsv
            cache=False,
        )

        # Add sample identifier to barcodes to avoid duplicates across samples
        adata.obs_names = [f"{sample_id}_{bc}" for bc in adata.obs_names]

        # Add sample metadata
        adata.obs["sample_id"] = sample_id

        # Append to list
        adata_list.append(adata)

        # Clean up temporary files
        shutil.rmtree(sample_temp_folder, ignore_errors=True)

        print(f"Successfully processed {sample_id}")

    except Exception as e:
        print(f"Error processing {sample_id}: {str(e)}")
        continue

# Remove temporary folder if empty
if os.path.exists(temp_folder) and not os.listdir(temp_folder):
    os.rmdir(temp_folder)

# Check if any AnnData objects were created
if not adata_list:
    raise ValueError(
        "No valid AnnData objects were created. Check your files and folder structure."
    )

# Merge AnnData objects
combined_adata = anndata.concat(adata_list, merge="same", join="outer")

# Verify raw counts (should be integers or sparse matrix)
print("Combined AnnData object:")
print(combined_adata)
print("Sample count matrix (first 5x5):")
print(combined_adata.X[:5, :5].toarray())

# Save to H5AD
combined_adata.write_h5ad(output_file)
print(f"H5AD file saved as {output_file}")
