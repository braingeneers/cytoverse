#!/usr/bin/env python3
"""Generate UCE-brain reference embeddings from an h5ad file.

Runs the PyTorch ground-truth pipeline (not ONNX) so the output can be used
as the reference that IVFPQ and PUMAP training are validated against. Only
the ONNX transformer is deployed in the browser; this script generates the
embeddings that prove the ONNX concordance holds.

Outputs: embeddings.npy (float32, shape (n_cells, 512)) + labels.parquet

Usage:
  uce-brain/.venv/bin/python uce-brain/scripts/embed_h5ad.py \\
      ~/data/h5ad/adata_metaatlas_final_raw.h5ad \\
      data/references/sspsygene-ucebrain \\
      --labels Dataset --labels Class --labels Type.v1 \\
      --max-cells 5000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import typer

UCE_BRAIN_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = UCE_BRAIN_ROOT.parent

DEFAULT_CKPT = UCE_BRAIN_ROOT / "data" / "models" / "uce-brain-pilot-8l-512d"
DEFAULT_BRAIN_SRC = UCE_BRAIN_ROOT / "uce-brain" / "src"
DEFAULT_GENE_DICT = UCE_BRAIN_ROOT / "uce-brain" / "gene_data" / "human_gene_dict.json"

_UCE_PAD_LENGTH = 2048
_UCE_SAMPLE_SIZE = 1024
_UCE_CLS_TOKEN_IDX = 1
_UCE_CHROM_TOKEN_OFFSET = 1000
_UCE_CHROM_TOKEN_RIGHT_IDX = 2000
_UCE_PAD_TOKEN_IDX = 0

app = typer.Typer(add_completion=False)


def _pick_device(prefer: Optional[str]):
    import torch
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@app.command()
def embed(
    h5ad_path: Path = typer.Argument(exists=True, help="Input h5ad file"),
    output_path: Path = typer.Argument(help="Output directory (embeddings.npy + labels.parquet)"),
    ckpt: Path = typer.Option(DEFAULT_CKPT, help="UCE-brain checkpoint directory"),
    brain_src: Path = typer.Option(DEFAULT_BRAIN_SRC, help="Path to uce-brain/src"),
    gene_dict_path: Path = typer.Option(DEFAULT_GENE_DICT, help="Path to human_gene_dict.json"),
    labels: list[str] = typer.Option([], help="obs columns to export as labels"),
    max_cells: Optional[int] = typer.Option(None, help="Subset to first N cells"),
    device: Optional[str] = typer.Option(None, help="Torch device (cpu/cuda/mps)"),
    validate: bool = typer.Option(True, help="Re-run 10 random cells and compare"),
) -> None:
    """Embed h5ad cells with UCE-brain (PyTorch) → embeddings.npy + labels.parquet."""
    import torch
    from tqdm import tqdm

    # Put brain_src on path so uce_brain.* imports resolve
    if str(brain_src) not in sys.path:
        sys.path.insert(0, str(brain_src))
    # Put uce-brain/scripts on path for export_model.load_brain
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from uce_brain.data.dataset import H5ADDataset, load_gene_mapping
    from export_model import load_brain

    print(f"Loading h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  shape: {adata.shape}")

    if max_cells is not None and adata.n_obs > max_cells:
        print(f"  subsetting to {max_cells} cells")
        adata = adata[:max_cells].copy()

    dev = _pick_device(device)
    print(f"  device: {dev}")

    print(f"Loading UCE-brain from {ckpt}")
    core, embedding_layer = load_brain(ckpt, brain_src)
    core = core.to(dev).eval()
    embedding_layer = embedding_layer.to(dev).eval()

    print(f"Loading gene mapping from {gene_dict_path}")
    gene_mapping = load_gene_mapping(str(gene_dict_path), species="human")

    print(f"Building H5ADDataset (pad_length={_UCE_PAD_LENGTH}, sample_size={_UCE_SAMPLE_SIZE})")
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
    print(f"  aligned genes: {len(ds.aligned_gene_names)}")

    n = adata.n_obs
    out_dim = core.output_projector.out_features if hasattr(core.output_projector, "out_features") else 512
    embeddings = np.zeros((n, out_dim), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(n), desc="UCE-brain inference"):
            sample = ds[i]
            input_ids = sample["batch_sentences"].to(dev)
            padding_mask = sample["mask"].to(dev)
            attn_mask = (~padding_mask).float()
            src = embedding_layer(input_ids)
            _, cell_emb = core(src, attn_mask)
            embeddings[i] = cell_emb.squeeze(0).cpu().numpy()

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  embedding norms: min={norms.min():.6f} max={norms.max():.6f} (expect ~1.0)")

    os.makedirs(output_path, exist_ok=True)
    emb_path = output_path / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved {emb_path} {embeddings.shape}")

    if labels:
        valid = [l for l in labels if l in adata.obs.columns]
        missing = set(labels) - set(valid)
        if missing:
            print(f"  ⚠ label columns not found: {missing}")
        df = pd.DataFrame({l: adata.obs[l].astype("category").values for l in valid})
        lbl_path = output_path / "labels.parquet"
        df.to_parquet(lbl_path, compression="snappy", index=None)
        print(f"Saved {lbl_path} {df.shape}")

    if validate:
        rng = np.random.default_rng(0)
        sample_idx = sorted(rng.choice(n, size=min(10, n), replace=False))
        print(f"\nValidating {len(sample_idx)} cells: {sample_idx}")
        stored = np.load(emb_path)
        all_ok = True
        with torch.no_grad():
            for idx in sample_idx:
                s = ds[int(idx)]
                src2 = embedding_layer(s["batch_sentences"].to(dev))
                _, emb2 = core(src2, (~s["mask"].to(dev)).float())
                diff = float(np.abs(emb2.squeeze(0).cpu().numpy() - stored[idx]).max())
                ok = diff < 1e-4
                all_ok = all_ok and ok
                print(f"  cell {idx}: {'OK' if ok else 'FAIL'} max|diff|={diff:.2e}")
        if all_ok:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
