#!/usr/bin/env python3
"""
Label cells in an h5ad file using SCimilarity predictions.

This script loads an h5ad file, computes embeddings using SCimilarity,
and outputs predictions in CSV format matching the App.tsx output format.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import anndata
import scimilarity
from scimilarity import CellAnnotation
from scimilarity.utils import align_dataset, lognorm_counts


def main():
    parser = argparse.ArgumentParser(
        description="Predict cell labels for h5ad file using SCimilarity"
    )
    parser.add_argument(
        "input_h5ad",
        type=Path,
        help="Path to input h5ad file"
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/scimilarity/model_v1.1"),
        help="Path to SCimilarity model directory (default: data/scimilarity/model_v1.1)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for computation"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=50,
        help="Number of nearest neighbors for prediction (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_h5ad.exists():
        print(f"Error: Input file '{args.input_h5ad}' does not exist")
        sys.exit(1)
    
    if not args.model_path.exists():
        print(f"Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)
    
    # Ensure output directory exists
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading h5ad file: {args.input_h5ad}")
    adata = anndata.read_h5ad(args.input_h5ad)
    print(f"  Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    
    print(f"Loading SCimilarity model from: {args.model_path}")
    cell_annotation = CellAnnotation(
        model_path=str(args.model_path),
        use_gpu=args.use_gpu
    )
    
    # Add counts layer if it doesn't exist (required by lognorm_counts)
    if "counts" not in adata.layers:
        print("  Adding counts layer from X matrix")
        adata.layers["counts"] = adata.X.copy()
    
    # Align dataset to model genes
    print("Aligning dataset to model gene space...")
    adata_aligned = align_dataset(adata, cell_annotation.gene_order)
    
    # Log-normalize the data
    print("Log-normalizing counts...")
    adata_normalized = lognorm_counts(adata_aligned)
    
    # Get embeddings
    print("Computing cell embeddings...")
    if hasattr(adata_normalized.X, 'toarray'):
        # Handle sparse matrices
        expression_data = adata_normalized.X.toarray().astype(np.float32)
    else:
        expression_data = adata_normalized.X.astype(np.float32)
    
    embeddings = cell_annotation.get_embeddings(expression_data)
    print(f"  Computed embeddings shape: {embeddings.shape}")
    
    # Get predictions using k-NN search
    print(f"Predicting cell types using k={args.k_neighbors} nearest neighbors...")
    predictions, nn_idxs, nn_dists, stats = cell_annotation.get_predictions_knn(
        embeddings,
        k=args.k_neighbors,
        disable_progress=False
    )
    
    # Create output dataframe in the same format as App.tsx
    # App.tsx format: cell_id,category_label
    results_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'category_label': predictions.values
    })
    
    # Save to CSV
    print(f"Writing predictions to: {args.output_csv}")
    results_df.to_csv(args.output_csv, index=False, quoting=1)  # quoting=1 for QUOTE_ALL
    
    print(f"Successfully labeled {len(results_df)} cells")
    
    # Print summary statistics
    label_counts = results_df['category_label'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.head(10).items():
        print(f"  {label}: {count} cells")
    if len(label_counts) > 10:
        print(f"  ... and {len(label_counts) - 10} more cell types")


if __name__ == "__main__":
    main()