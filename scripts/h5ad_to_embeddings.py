#!/usr/bin/env python3
"""Generate SCimilarity reference embeddings from an h5ad file.

Outputs: embeddings.npy + labels.parquet in <output_path>/.

For UCE-brain reference embeddings use uce-brain/scripts/embed_h5ad.py
with the uce-brain/.venv (which carries torch/safetensors).

Usage:
  python scripts/h5ad_to_embeddings.py \\
      <h5ad> <model_dir> <output_dir> \\
      [--labels cell_type] [--max-cells 50000]
"""

import os
import typer
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import scanpy as sc


app = typer.Typer(
    help="Ingest h5ad file through SCimilarity → embeddings.npy + labels.parquet",
    add_completion=False,
)


@app.command()
def ingest(
    h5ad_path: Path = typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to h5ad file to process",
    ),
    model_path: Path = typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to SCimilarity model directory (e.g. data/models/scimilarity/model_v1.1)",
    ),
    output_path: Path = typer.Argument(
        file_okay=False,
        dir_okay=True,
        help="Output directory path",
    ),
    labels: list[str] = typer.Option(
        [],
        help="Label columns from adata.obs to extract (e.g. --labels cell_type --labels tissue)",
    ),
    max_cells: Optional[int] = typer.Option(
        None,
        help="Maximum number of cells to export (default: all)",
    ),
    validate: bool = typer.Option(
        True,
        help="Re-run 10 random cells and compare to stored embeddings",
    ),
) -> None:
    """Process h5ad through SCimilarity → embeddings.npy + labels.parquet."""
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")

    if max_cells is not None and adata.n_obs > max_cells:
        print(f"  subsetting to {max_cells} cells")
        adata = adata[:max_cells].copy()

    labels = _validate_label_columns(adata, labels)
    embeddings, ce = _run_scimilarity(adata, model_path)
    print(f"  embeddings shape: {embeddings.shape}")

    os.makedirs(output_path, exist_ok=True)

    emb_path = output_path / "embeddings.npy"
    np.save(emb_path, embeddings.astype(np.float32))
    print(f"Saved {emb_path}")

    lbl_path = None
    lbl_df = None
    if labels:
        lbl_df = pd.DataFrame(
            {l: adata.obs[l].astype("category").values for l in labels}
        )
        lbl_path = output_path / "labels.parquet"
        lbl_df.to_parquet(lbl_path, compression="snappy", index=None)
        print(f"Saved {lbl_path} {lbl_df.shape}")

    print(f"\n✅ Done: {output_path}")

    if validate:
        _validate(adata, embeddings, model_path, ce)


def _validate_label_columns(adata, labels):
    if not labels:
        print("  ⚠ No label columns specified.")
        return []
    valid = []
    for l in labels:
        if l not in adata.obs.columns:
            print(f"  ⚠ Column '{l}' not in adata.obs — skipping")
            continue
        n = adata.obs[l].nunique()
        if n > 1000:
            print(f"  ⚠ Column '{l}' has {n} unique values (>1000) — skipping")
            continue
        valid.append(l)
    return valid


def _run_scimilarity(adata, model_path: Path):
    import scimilarity
    from scimilarity.utils import align_dataset, lognorm_counts

    print(f"\nLoading SCimilarity from: {model_path}")
    ce = scimilarity.CellEmbedding(str(model_path))

    aligned = align_dataset(adata, ce.gene_order)

    if "counts" in adata.layers:
        aligned.layers["counts"] = aligned.layers["counts"]
    elif "raw_counts" in adata.layers:
        aligned.layers["counts"] = aligned.layers["raw_counts"]
    elif "raw" in adata.layers:
        aligned.layers["counts"] = aligned.layers["raw"]
    else:
        aligned.layers["counts"] = aligned.X

    lognorm = lognorm_counts(aligned)
    embeddings = ce.get_embeddings(lognorm.X)
    return embeddings, ce


def _validate(adata, embeddings, model_path, ce):
    import scimilarity
    from scimilarity.utils import align_dataset, lognorm_counts

    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(adata.n_obs, size=min(10, adata.n_obs), replace=False))
    print(f"\nValidating {len(idx)} cells: {idx}")

    subset = adata[idx].copy()
    aligned = align_dataset(subset, ce.gene_order)
    if "counts" in subset.layers:
        aligned.layers["counts"] = aligned.layers["counts"]
    elif "raw_counts" in subset.layers:
        aligned.layers["counts"] = aligned.layers["raw_counts"]
    elif "raw" in subset.layers:
        aligned.layers["counts"] = aligned.layers["raw"]
    else:
        aligned.layers["counts"] = aligned.X

    lognorm = lognorm_counts(aligned)
    rerun = ce.get_embeddings(lognorm.X)

    all_ok = True
    tol = 1e-5
    for i, row in enumerate(idx):
        diff = float(np.abs(rerun[i] - embeddings[row]).max())
        ok = diff < tol
        all_ok = all_ok and ok
        print(f"  cell {row}: {'OK' if ok else 'FAIL'} max|diff|={diff:.2e}")

    if all_ok:
        print("✅ Validation passed")
    else:
        print("❌ Validation failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
