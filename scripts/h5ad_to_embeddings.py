#!/usr/bin/env python3
"""Generate reference embeddings from an h5ad file.

Supports two foundation models:
  - scimilarity (default): runs under the cytoverse venv
  - uce-brain: runs under uce-edge/.venv (pulls torch, uce_brain, safetensors)

For uce-brain, invoke with:
  uce-edge/.venv/bin/python scripts/h5ad_to_embeddings.py \\
      <h5ad> <model_ckpt_dir> <output_dir> --foundation-model uce-brain
"""

import os
import sys
import typer
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import scanpy as sc


# Create Typer app
app = typer.Typer(
    help="Ingest h5ad file as a reference through a foundation model",
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
        help="Path to foundation model directory (SCimilarity: model_v1.1; UCE-brain: checkpoint dir)",
    ),
    output_path: Path = typer.Argument(
        file_okay=False,
        dir_okay=True,
        help="Output directory path",
    ),
    foundation_model: str = typer.Option(
        "scimilarity",
        "--foundation-model",
        help="Foundation model: 'scimilarity' or 'uce-brain'",
    ),
    labels: list[str] = typer.Option(
        [],
        help="Label columns from adata.obs to extract (e.g., --labels cell_type --labels tissue)",
    ),
    max_cells: int = typer.Option(
        None,
        help="Maximum number of cells to export (default: export all cells)",
    ),
    validate: bool = typer.Option(
        True,
        help="Validate the output files after processing (default: True)",
    ),
    brain_src: Path = typer.Option(
        None,
        help="[uce-brain only] Path to UCE-brain/src. Defaults to uce-brain/src.",
    ),
    gene_dict: Path = typer.Option(
        None,
        help="[uce-brain only] Path to human_gene_dict.json. Defaults to uce-brain/gene_data/human_gene_dict.json.",
    ),
    device: str = typer.Option(
        None,
        help="[uce-brain only] Torch device (cpu, cuda, mps). Default: auto-detect.",
    ),
) -> None:
    """Process h5ad through the selected foundation model → embeddings.npy + labels.parquet."""

    if foundation_model not in ("scimilarity", "uce-brain"):
        print(f"❌ Unknown --foundation-model: {foundation_model}")
        raise typer.Exit(1)

    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded data shape: {adata.shape}")
    print(f"  Available obs columns: {list(adata.obs.columns)}")

    # Subset cells if max_cells is specified
    if max_cells is not None and adata.n_obs > max_cells:
        print(f"\nSubsetting to first {max_cells} cells (from {adata.n_obs} total)")
        adata = adata[:max_cells, :].copy()
        print(f"  Subset data shape: {adata.shape}")

    # Validate label columns
    labels = _validate_label_columns(adata, labels)

    # Run the selected foundation model
    print(f"\nFoundation model: {foundation_model}")
    if foundation_model == "scimilarity":
        embeddings, model_handle = _run_scimilarity(adata, model_path)
    else:
        embeddings, model_handle = _run_uce_brain(
            adata, model_path, brain_src=brain_src, gene_dict=gene_dict, device=device
        )

    print(f"  Embeddings shape: {embeddings.shape}")

    # Create output directory
    print(f"\nCreating output directory: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Save embeddings
    embeddings_path = output_path / "embeddings.npy"
    print(f"Saving embeddings to: {embeddings_path}")
    np.save(embeddings_path, embeddings.astype(np.float32))

    # Extract and save labels if specified
    labels_path = None
    labels_df = None
    if labels:
        print(f"Extracting labels: {labels}")
        labels_df = pd.DataFrame()
        for label in labels:
            labels_df[label] = adata.obs[label].values
            labels_df[label] = labels_df[label].astype("category")
            print(f"  Added '{label}': {labels_df[label].nunique()} unique values")

        labels_path = output_path / "labels.parquet"
        print(f"Saving labels to: {labels_path}")
        labels_df.to_parquet(labels_path, compression="snappy", index=None)
        print(f"Labels saved with shape: {labels_df.shape}")

    print(f"\n✅ Processing complete!")
    print(f"   Embeddings: {embeddings_path} (shape: {embeddings.shape})")
    if labels_path is not None:
        print(f"   Labels: {labels_path} (shape: {labels_df.shape})")
    print(f"   Output directory: {output_path}")

    # Validation
    if validate:
        print("\n" + "=" * 60)
        print("Running validation...")
        print("=" * 60)
        try:
            _validate_outputs(
                h5ad_path=h5ad_path,
                output_dir=output_path,
                model_path=model_path,
                foundation_model=foundation_model,
                model_handle=model_handle,
                n_samples=10,
            )
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            raise typer.Exit(1)


def _validate_label_columns(adata, labels):
    """Drop label columns that don't exist or have > 1000 unique values."""
    if not labels:
        print(
            "  ⚠️ No label columns specified. Only embeddings will be exported."
        )
        return []

    print(f"\nValidating label columns: {labels}")
    valid_labels = []
    for label in labels:
        if label not in adata.obs.columns:
            print(f"  ❌ Warning: Column '{label}' not found in adata.obs")
            continue
        unique_count = adata.obs[label].nunique()
        print(f"  Column '{label}': {unique_count} unique values")
        if unique_count > 1000:
            print(
                f"  ❌ Warning: Column '{label}' has {unique_count} unique values, exceeding limit of 1000. Skipping."
            )
            continue
        valid_labels.append(label)
    print(f"  ✅ Valid label columns: {valid_labels}")
    return valid_labels


# ----------------------------------------------------------------------
# SCimilarity path
# ----------------------------------------------------------------------

def _run_scimilarity(adata, model_path: Path):
    """Run SCimilarity on adata. Returns (embeddings, CellEmbedding handle)."""
    import scimilarity
    from scimilarity.utils import align_dataset, lognorm_counts

    print(f"\nLoading SCimilarity model from: {model_path}")
    ce = scimilarity.CellEmbedding(str(model_path))

    print("\nProcessing data through SCimilarity...")
    print("  1. Aligning dataset...")
    aligned_adata = align_dataset(adata, ce.gene_order)
    print(f"     Aligned data shape: {aligned_adata.shape}")

    # Locate raw counts
    if "counts" in adata.layers:
        print("  Counts found in adata.layers['counts']")
    elif "raw_counts" in adata.layers:
        print("  Raw counts found in adata.layers['raw_counts']")
        aligned_adata.layers["counts"] = aligned_adata.layers["raw_counts"]
    elif "raw" in adata.layers:
        print("  Raw counts found in adata.layers['raw']")
        aligned_adata.layers["counts"] = aligned_adata.layers["raw"]
    else:
        print("  No raw counts found in adata.layers. Using adata.X as counts.")
        aligned_adata.layers["counts"] = aligned_adata.X

    print("  2. Log-normalizing counts...")
    lognorm_adata = lognorm_counts(aligned_adata)
    print(f"     Log-normalized data shape: {lognorm_adata.shape}")

    print("  3. Generating embeddings...")
    embeddings = ce.get_embeddings(lognorm_adata.X)
    return embeddings, ce


# ----------------------------------------------------------------------
# UCE-brain path
# ----------------------------------------------------------------------

# Sampler constants (mirrored from uce-edge/scripts/brain_reference_pipeline.py)
_UCE_PAD_LENGTH = 2048
_UCE_SAMPLE_SIZE = 1024
_UCE_CLS_TOKEN_IDX = 1
_UCE_CHROM_TOKEN_OFFSET = 1000
_UCE_CHROM_TOKEN_RIGHT_IDX = 2000
_UCE_PAD_TOKEN_IDX = 0


def _default_brain_src() -> Path:
    return Path(__file__).resolve().parent.parent / "uce-brain" / "src"


def _default_gene_dict() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "uce-brain"
        / "gene_data"
        / "human_gene_dict.json"
    )


def _pick_torch_device(prefer):
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_uce_brain(
    adata,
    ckpt_dir: Path,
    *,
    brain_src: Optional[Path],
    gene_dict: Optional[Path],
    device: Optional[str],
):
    """Run UCE-brain on adata. Returns (embeddings, LoadedBrain handle)."""
    # Lazy imports — these only resolve under uce-edge/.venv
    import torch

    brain_src = brain_src or _default_brain_src()
    gene_dict = gene_dict or _default_gene_dict()

    if not brain_src.exists():
        raise FileNotFoundError(f"UCE-brain src not found: {brain_src}")
    if not gene_dict.exists():
        raise FileNotFoundError(f"UCE-brain gene_dict not found: {gene_dict}")

    # Import UCE-brain packages (requires brain_src on sys.path)
    if str(brain_src) not in sys.path:
        sys.path.insert(0, str(brain_src))

    # Reuse the LoadedBrain helper from the export script — same venv
    # (uce-edge/.venv) carries both, and the loader is identical.
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from uce_brain.data.dataset import H5ADDataset, load_gene_mapping  # noqa: E402
    from uce_brain_export_model import load_brain  # noqa: E402

    dev = _pick_torch_device(device)
    print(f"  Using device: {dev}")

    print(f"  Loading UCE-brain from {ckpt_dir}")
    core, embedding_layer = load_brain(ckpt_dir, brain_src)
    core = core.to(dev).eval()
    embedding_layer = embedding_layer.to(dev).eval()

    print(f"  Loading gene mapping from {gene_dict}")
    gene_mapping = load_gene_mapping(str(gene_dict), species="human")

    print(f"  Building H5ADDataset (pad_length={_UCE_PAD_LENGTH}, sample_size={_UCE_SAMPLE_SIZE})")
    ds = H5ADDataset(
        adata=adata,
        gene_mapping={"human": gene_mapping},
        pad_length=_UCE_PAD_LENGTH,
        sample_size=_UCE_SAMPLE_SIZE,
        cls_token_idx=_UCE_CLS_TOKEN_IDX,
        chrom_token_offset=_UCE_CHROM_TOKEN_OFFSET,
        chrom_token_right_idx=_UCE_CHROM_TOKEN_RIGHT_IDX,
        pad_token_idx=_UCE_PAD_TOKEN_IDX,
        mask_prop=0.0,
        max_cells=adata.n_obs,
    )
    print(f"  Aligned genes: {len(ds.aligned_gene_names)}")

    n = adata.n_obs
    out_dim = core.output_projector.out_features if hasattr(core.output_projector, "out_features") else 512
    embeddings = np.zeros((n, out_dim), dtype=np.float32)

    from tqdm import tqdm

    # Process cell-by-cell. Batching hurts at O(L²) attention (proven in uce-edge).
    with torch.no_grad():
        for i in tqdm(range(n), desc="  UCE-brain inference"):
            sample = ds[i]
            input_ids = sample["batch_sentences"].to(dev)  # (1, L)
            padding_mask = sample["mask"].to(dev)          # (1, L) True=pad
            attn_mask = (~padding_mask).float()
            src = embedding_layer(input_ids)
            _, cell_emb = core(src, attn_mask)
            embeddings[i] = cell_emb.squeeze(0).cpu().numpy().astype(np.float32)

    # Sanity: embeddings should be unit-norm (core L2-normalizes CLS).
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"  Embedding norms: min={norms.min():.6f} max={norms.max():.6f} "
        f"(expect ~1.0)"
    )

    # Return a handle so validation can re-run a few cells
    loaded = {
        "core": core,
        "embedding_layer": embedding_layer,
        "dataset": ds,
        "device": dev,
    }
    return embeddings, loaded


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def _validate_outputs(
    h5ad_path: Path,
    output_dir: Path,
    model_path: Path,
    foundation_model: str,
    model_handle,
    n_samples: int = 10,
) -> None:
    """Reload output, regenerate a handful of rows, compare."""

    print(f"Validating output files against h5ad: {h5ad_path}")
    print(f"\nLoading h5ad file...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    embeddings_path = output_dir / "embeddings.npy"
    labels_path = output_dir / "labels.parquet"

    if not embeddings_path.exists():
        print(f"❌ Error: {embeddings_path} not found")
        raise typer.Exit(1)

    print(f"\nLoading embeddings from: {embeddings_path}")
    embeddings_array = np.load(embeddings_path)
    print(f"  Embeddings shape: {embeddings_array.shape}")

    has_labels = labels_path.exists()
    if has_labels:
        print(f"\nLoading labels from: {labels_path}")
        labels_df = pd.read_parquet(labels_path)
        print(f"  Labels shape: {labels_df.shape}")
        print(f"  Label columns: {list(labels_df.columns)}")

    n_cells = min(adata.n_obs, embeddings_array.shape[0])
    n_samples = min(n_samples, n_cells)
    random_indices = np.random.choice(n_cells, size=n_samples, replace=False)
    random_indices.sort()

    print(f"\nValidating {n_samples} random samples: {random_indices}")

    # Labels (same for both foundation models)
    all_labels_match = True
    if has_labels:
        print("\n=== Validating Labels ===")
        for idx in random_indices:
            print(f"\nRow {idx}:")
            for col in labels_df.columns:
                if col in adata.obs.columns:
                    h5ad_value = adata.obs.iloc[idx][col]
                    parquet_value = labels_df.iloc[idx][col]
                    match = str(h5ad_value) == str(parquet_value)
                    status = "✅" if match else "❌"
                    print(f"  {col}: {status} h5ad='{h5ad_value}' parquet='{parquet_value}'")
                    if not match:
                        all_labels_match = False
        if all_labels_match:
            print("\n✅ All labels match!")
        else:
            print("\n❌ Some labels do not match!")

    # Embeddings
    print("\n=== Validating Embeddings ===")
    if foundation_model == "scimilarity":
        all_embeddings_match = _validate_embeddings_scimilarity(
            adata, embeddings_array, random_indices, model_path, model_handle
        )
    else:
        all_embeddings_match = _validate_embeddings_uce_brain(
            adata, embeddings_array, random_indices, model_handle  # type: ignore[arg-type]
        )

    print("\n=== Validation Summary ===")
    if has_labels:
        print(f"Labels: {'✅ PASS' if all_labels_match else '❌ FAIL'}")
    print(f"Embeddings: {'✅ PASS' if all_embeddings_match else '❌ FAIL'}")

    if (not has_labels or all_labels_match) and all_embeddings_match:
        print("\n✅ Validation PASSED!")
    else:
        print("\n❌ Validation FAILED!")
        raise typer.Exit(1)


def _validate_embeddings_scimilarity(
    adata, embeddings_array, random_indices, model_path, ce
) -> bool:
    """Exact match required — SCimilarity is deterministic."""
    import scimilarity
    from scimilarity.utils import align_dataset, lognorm_counts

    if ce is None:
        print(f"\nLoading SCimilarity model from: {model_path}")
        ce = scimilarity.CellEmbedding(str(model_path))
    else:
        print("\nUsing provided SCimilarity model")

    print("\nProcessing subset through SCimilarity...")
    subset_adata = adata[random_indices, :].copy()
    aligned_subset = align_dataset(subset_adata, ce.gene_order)

    if "counts" in subset_adata.layers:
        aligned_subset.layers["counts"] = aligned_subset.layers["counts"]
    elif "raw_counts" in subset_adata.layers:
        aligned_subset.layers["counts"] = aligned_subset.layers["raw_counts"]
    elif "raw" in subset_adata.layers:
        aligned_subset.layers["counts"] = aligned_subset.layers["raw"]
    else:
        aligned_subset.layers["counts"] = aligned_subset.X

    lognorm_subset = lognorm_counts(aligned_subset)
    subset_embeddings = ce.get_embeddings(lognorm_subset.X)

    tolerance = 1e-5
    all_match = True
    for i, idx in enumerate(random_indices):
        regenerated = subset_embeddings[i]
        stored = embeddings_array[idx]
        max_diff = np.max(np.abs(regenerated - stored))
        match = max_diff < tolerance
        status = "✅" if match else "❌"
        print(f"\nRow {idx}: {status} max_diff={max_diff:.2e}")
        if not match:
            all_match = False
            print(f"  Regenerated[:5]: {regenerated[:5]}")
            print(f"  Stored[:5]: {stored[:5]}")

    if all_match:
        print(f"\n✅ All embeddings match within tolerance ({tolerance})!")
    else:
        print(f"\n❌ Some embeddings do not match within tolerance ({tolerance})!")
    return all_match


def _validate_embeddings_uce_brain(
    _adata, embeddings_array, random_indices, model_handle
) -> bool:
    """Bit-identical match — the sampler is deterministically seeded by cell idx.

    H5ADDataset uses seed=idx internally, so re-running `ds[i]` produces the
    same tokens; the core transformer is pure; so embeddings should match
    exactly. We still allow a tiny tolerance for cross-run GPU non-determinism.
    """
    import torch

    core = model_handle["core"]
    embedding_layer = model_handle["embedding_layer"]
    ds = model_handle["dataset"]
    dev = model_handle["device"]

    print("\nRe-running UCE-brain on the sampled rows...")
    tolerance = 1e-4  # GPU non-determinism can exceed 1e-5
    all_match = True
    with torch.no_grad():
        for idx in random_indices:
            sample = ds[int(idx)]
            input_ids = sample["batch_sentences"].to(dev)
            padding_mask = sample["mask"].to(dev)
            attn_mask = (~padding_mask).float()
            src = embedding_layer(input_ids)
            _, cell_emb = core(src, attn_mask)
            regenerated = cell_emb.squeeze(0).cpu().numpy().astype(np.float32)
            stored = embeddings_array[idx]

            max_diff = float(np.max(np.abs(regenerated - stored)))
            cos = float(np.dot(regenerated, stored))  # L2-normalized
            match = max_diff < tolerance
            status = "✅" if match else "❌"
            print(f"\nRow {idx}: {status} max_diff={max_diff:.2e}  cos={cos:.6f}")
            if not match:
                all_match = False

    if all_match:
        print(f"\n✅ All embeddings match within tolerance ({tolerance})!")
    else:
        print(f"\n⚠️ Some embeddings differ beyond tolerance ({tolerance}). "
              f"Cosine should still be ≈1 if the sampler replayed correctly.")
    return all_match


if __name__ == "__main__":
    app()
