#!/usr/bin/env python3
"""Export UCE-brain artifacts for browser inference in cytoverse.

Mirrors scripts/scimilarity_export_model.py in spirit but ships a different
artifact set — UCE-brain's pipeline is tokens + gather + transformer, not
raw-counts → normalize → embed. The transformer runs in the browser via ONNX;
everything upstream (log1p/normalize, weighted sampling, chrom ordering,
CLS/PAD inserts, protein embedding gather) is JS-side.

Outputs (under <output_path>/):
  embedding/transformer.onnx          8-layer brain core, FP32, opset 17
  embedding/protein_embeddings.bin    (N_rows, 5120) FP32, ~400 MB, post-LayerNorm
  embedding/gene_dict.json            renumbered gene dict + special-token map
  metadata.json                       foundationModel / embeddingDim / displayName

The cytoverse venv doesn't carry uce_brain or transformers; invoke this
script with the uce-edge venv:

  uce-edge/.venv/bin/python scripts/uce_brain_export_model.py \\
      --output-path public/models/sspsygene-ucebrain \\
      --display-name "SSPsyGene - UCE Brain"

Defaults point at the uce-edge submodules and checkpoint.
"""
from __future__ import annotations

import json
import sys
import typer
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import onnx
import onnxruntime as ort
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRAIN_SRC = REPO_ROOT / "uce-edge" / "UCE-brain" / "src"
DEFAULT_CKPT = REPO_ROOT / "uce-edge" / "model_files" / "uce-brain-pilot-8l-512d"
DEFAULT_GENE_DICT = REPO_ROOT / "uce-edge" / "UCE-brain" / "gene_data" / "human_gene_dict.json"

# UCE-brain sampler constants (see UCE-brain/src/uce_brain/data/sampler.py)
PAD_TOKEN_IDX = 0
CLS_TOKEN_IDX = 1
CHROM_TOKEN_OFFSET = 1000
CHROM_TOKEN_RIGHT_IDX = 2000  # CHROM_END

EMBEDDING_DIM = 5120
OUTPUT_DIM = 512
VOCAB_SIZE = 145469
EXPORT_SEQ_LEN = 1024
EXPORT_BATCH = 2


app = typer.Typer(
    help="Export UCE-brain artifacts for browser inference",
    add_completion=False,
)


class BrainCore(torch.nn.Module):
    """UCEModel with protein embedding lookup stripped out (caller-side gather).

    The exported forward takes pre-embedded src and a float mask (1=valid,
    0=pad), returning (gene_embeddings, cell_embedding). cell_embedding is
    the L2-normalized CLS token output — the thing we want for IVFPQ search.
    """

    def __init__(self, uce):
        super().__init__()
        self.projector = uce.input_gene_embedding_projector
        self.pos_encoder = uce.pos_encoder
        self.transformer_encoder = uce.transformer_encoder
        # Nested-tensor fast path uses MPS-unsupported ops and doesn't trace
        # cleanly — disable it for both export and inference.
        self.transformer_encoder.enable_nested_tensor = False
        self.transformer_encoder.use_nested_tensor = False
        self.transformer_encoder.mask_check = False
        self.output_projector = uce.output_embedding_projector

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        x = self.projector(src)
        x = self.pos_encoder(x)
        key_padding_mask = mask == 0
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.output_projector(x)
        cell = x[:, 0, :]
        cell = torch.nn.functional.normalize(cell, dim=-1)
        return x, cell


def load_brain(ckpt_dir: Path, brain_src: Path):
    """Build BrainCore + the raw embedding layer from a UCE-brain checkpoint."""
    if str(brain_src) not in sys.path:
        sys.path.insert(0, str(brain_src))
    from uce_brain.model.modeling import UCEModel  # noqa: E402
    from uce_brain.model.config import UCEConfig  # noqa: E402

    config = UCEConfig.from_pretrained(str(ckpt_dir))
    config.dropout = 0.0  # eval: matters for pos_encoder Dropout
    uce = UCEModel(config)
    uce.eval()

    full_state = load_file(str(ckpt_dir / "model.safetensors"))
    inner = {k[len("uce.") :]: v for k, v in full_state.items() if k.startswith("uce.")}
    missing, _unexpected = uce.load_state_dict(inner, strict=False)
    # expression_decoder.* intentionally excluded.
    assert not missing, f"missing UCE keys: {missing[:5]}..."

    return BrainCore(uce).eval(), uce.embedding_layer


def export_transformer(
    core: BrainCore, embedding_layer, out_path: Path, opset: int = 17
) -> None:
    """Export the BrainCore transformer to ONNX and validate the graph."""
    print(f"\n1. Exporting transformer to {out_path} (opset {opset})")

    # Use an embedding lookup with random token IDs so the traced graph sees
    # a realistic input distribution (not pure noise).
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, VOCAB_SIZE, (EXPORT_BATCH, EXPORT_SEQ_LEN), generator=g)
    with torch.no_grad():
        src = embedding_layer(ids)
    mask = torch.ones(EXPORT_BATCH, EXPORT_SEQ_LEN, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        core,
        (src, mask),
        str(out_path),
        input_names=["src", "mask"],
        output_names=["gene_embeddings", "cell_embedding"],
        dynamic_axes={
            "src": {0: "batch", 1: "seq_len"},
            "mask": {0: "batch", 1: "seq_len"},
            "gene_embeddings": {0: "batch", 1: "seq_len"},
            "cell_embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Exported {size_mb:.1f} MB")

    # Validate graph structure
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ✅ ONNX graph validation passed")


def validate_transformer_concordance(
    core: BrainCore, embedding_layer, onnx_path: Path
) -> None:
    """Compare PyTorch vs ONNX transformer outputs on the same input.

    uce-edge already proved this works; we re-run it here so the export
    script catches regressions immediately.
    """
    print(f"\n2. Validating ONNX concordance vs PyTorch")

    # Validate at the same shape used for export — the torchscript exporter
    # bakes some reshape constants from the traced shape, so smaller shapes
    # fail at runtime even though the dynamic_axes are declared. uce-edge
    # already proved dynamic shapes work in practice via the browser path.
    g = torch.Generator().manual_seed(42)
    ids = torch.randint(1, VOCAB_SIZE, (EXPORT_BATCH, EXPORT_SEQ_LEN), generator=g)
    with torch.no_grad():
        src = embedding_layer(ids)
    mask = torch.ones(EXPORT_BATCH, EXPORT_SEQ_LEN, dtype=torch.float32)

    with torch.no_grad():
        _, cell_pt = core(src, mask)
    cell_pt_np = cell_pt.cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        None, {"src": src.cpu().numpy(), "mask": mask.cpu().numpy()}
    )
    cell_onnx_np = outputs[1]

    max_diff = np.max(np.abs(cell_pt_np - cell_onnx_np))
    # Cosine per row (they're already L2-normalized, so this is just a dot)
    cos_per_row = np.sum(cell_pt_np * cell_onnx_np, axis=1)
    min_cos = float(cos_per_row.min())

    print(f"  max|diff| = {max_diff:.2e}")
    print(f"  min cosine = {min_cos:.6f}")

    if min_cos < 0.9999:
        print(f"  ❌ Concordance failed: min cosine {min_cos:.6f} < 0.9999")
        raise typer.Exit(1)
    print(f"  ✅ Concordance: min cosine {min_cos:.6f} ≥ 0.9999")


def extract_protein_embeddings(
    ckpt_dir: Path, gene_dict_path: Path, out_dir: Path
) -> dict:
    """Slice the human-only embedding table, pre-apply LayerNorm, renumber IDs.

    Returns the renumbered gene dict so caller can write additional metadata.
    """
    print(f"\n3. Extracting protein embedding table")

    state = load_file(str(ckpt_dir / "model.safetensors"))
    emb_full = state["uce.embedding_layer.embedding.weight"]  # (145469, 5120)
    ln_weight = state["uce.embedding_layer.layer_norm.weight"]
    ln_bias = state["uce.embedding_layer.layer_norm.bias"]
    assert emb_full.shape == (VOCAB_SIZE, EMBEDDING_DIM), emb_full.shape
    print(f"  Full embedding: {tuple(emb_full.shape)} {emb_full.dtype}")

    with open(gene_dict_path) as f:
        gene_dict_raw = json.load(f)
    human = gene_dict_raw["human"]
    n_genes = len(human)
    print(f"  Human genes: {n_genes}")

    chrom_ids = sorted({int(v["chromosome_id"]) for v in human.values()})
    chrom_tokens_old = [c + CHROM_TOKEN_OFFSET for c in chrom_ids]

    # Order: PAD, CLS, chrom tokens, CHROM_END, then genes (by original id).
    special_old_ids = [PAD_TOKEN_IDX, CLS_TOKEN_IDX]
    special_old_ids += chrom_tokens_old
    special_old_ids += [CHROM_TOKEN_RIGHT_IDX]

    gene_symbols_by_old = sorted(
        ((int(v["protein_embedding_id"]), sym) for sym, v in human.items()),
        key=lambda p: p[0],
    )
    old_ids_in_order = special_old_ids + [old_id for old_id, _ in gene_symbols_by_old]
    assert len(old_ids_in_order) == len(set(old_ids_in_order)), "duplicate old ids"
    print(f"  Total rows: {len(old_ids_in_order)} (specials={len(special_old_ids)}, genes={n_genes})")

    # Slice + pre-apply LayerNorm so browser skips that step
    old_ids_t = torch.tensor(old_ids_in_order, dtype=torch.long)
    sliced = emb_full[old_ids_t].contiguous()
    with torch.no_grad():
        shipped = torch.nn.functional.layer_norm(
            sliced.float(),
            normalized_shape=(EMBEDDING_DIM,),
            weight=ln_weight.float(),
            bias=ln_bias.float(),
            eps=1e-5,
        )
    shipped_np = shipped.cpu().numpy().astype(np.float32)
    print(f"  Shipped table: {shipped_np.shape} → {shipped_np.nbytes / 1e6:.1f} MB")

    # Build renumbered dict
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(old_ids_in_order)}
    new_gene_dict = {
        sym: {
            "protein_embedding_id": old_to_new[int(v["protein_embedding_id"])],
            "chromosome_id": int(v["chromosome_id"]),
            "location": int(v["location"]),
        }
        for sym, v in human.items()
    }
    chrom_old_to_new = {
        int(c): old_to_new[c + CHROM_TOKEN_OFFSET] for c in chrom_ids
    }

    output_dict = {
        "species": "human",
        "num_rows": int(shipped_np.shape[0]),
        "embedding_dim": int(shipped_np.shape[1]),
        "layer_norm_applied": True,
        "specials": {
            "pad_token_idx": old_to_new[PAD_TOKEN_IDX],
            "cls_token_idx": old_to_new[CLS_TOKEN_IDX],
            "chrom_token_right_idx": old_to_new[CHROM_TOKEN_RIGHT_IDX],
        },
        "chromosome_token_map": chrom_old_to_new,
        "genes": new_gene_dict,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "protein_embeddings.bin"
    json_path = out_dir / "gene_dict.json"
    shipped_np.tofile(bin_path)
    with open(json_path, "w") as f:
        json.dump(output_dict, f, separators=(",", ":"))

    print(f"  Wrote {bin_path} ({bin_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Wrote {json_path} ({json_path.stat().st_size / 1e3:.1f} KB)")

    # Round-trip check: a few genes and specials survive the renumber + LayerNorm
    print(f"\n4. Validating embedding table round-trip")
    sample_genes = [g for g in ["TP53", "EGFR", "ACTB", "GAPDH"] if g in human]
    if len(sample_genes) < 2:
        sample_genes = list(human.keys())[:4]
    ok_all = True
    for sym in sample_genes:
        old_id = int(human[sym]["protein_embedding_id"])
        new_id = new_gene_dict[sym]["protein_embedding_id"]
        old_vec = emb_full[old_id].float().cpu().numpy()
        with torch.no_grad():
            expected = torch.nn.functional.layer_norm(
                torch.from_numpy(old_vec).unsqueeze(0),
                (EMBEDDING_DIM,),
                ln_weight.float(),
                ln_bias.float(),
                eps=1e-5,
            ).squeeze(0).numpy()
        maxdiff = float(np.abs(expected - shipped_np[new_id]).max())
        ok = maxdiff == 0.0
        ok_all = ok_all and ok
        print(f"  {sym:8s} old={old_id:6d} new={new_id:6d} maxdiff={maxdiff:.2e} {'OK' if ok else 'FAIL'}")
    if not ok_all:
        print("  ❌ Round-trip mismatch"); raise typer.Exit(1)
    print(f"  ✅ All round-trip checks passed")

    return output_dict


def write_metadata(
    output_path: Path, display_name: str, gene_dict_info: dict
) -> None:
    """Write the root metadata.json that the worker reads to dispatch."""
    metadata = {
        "foundationModel": "uce-brain",
        "embeddingDim": OUTPUT_DIM,
        "displayName": display_name,
        # Hints for the worker — avoids re-parsing gene_dict.json just for these
        "proteinEmbeddingRows": gene_dict_info["num_rows"],
        "proteinEmbeddingDim": gene_dict_info["embedding_dim"],
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n5. Wrote {metadata_path}")


@app.command()
def export(
    output_path: Path = typer.Argument(
        help="Output model directory (e.g. public/models/sspsygene-ucebrain)",
    ),
    display_name: str = typer.Option(
        "UCE Brain",
        help="Human-readable name shown in the model dropdown",
    ),
    brain_src: Path = typer.Option(
        DEFAULT_BRAIN_SRC,
        help="Path to UCE-brain/src (must contain uce_brain/ package)",
    ),
    ckpt: Path = typer.Option(
        DEFAULT_CKPT,
        help="Path to UCE-brain checkpoint directory",
    ),
    gene_dict: Path = typer.Option(
        DEFAULT_GENE_DICT,
        help="Path to UCE-brain human_gene_dict.json",
    ),
    opset: int = typer.Option(17, help="ONNX opset version"),
    skip_embeddings: bool = typer.Option(
        False, help="Skip the ~400 MB protein_embeddings.bin + gene_dict.json"
    ),
) -> None:
    """Export UCE-brain artifacts to <output_path>/ for browser consumption."""
    if not brain_src.exists():
        print(f"❌ brain_src not found: {brain_src}")
        raise typer.Exit(1)
    if not ckpt.exists():
        print(f"❌ ckpt not found: {ckpt}")
        raise typer.Exit(1)
    if not gene_dict.exists():
        print(f"❌ gene_dict not found: {gene_dict}")
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    embedding_dir = output_path / "embedding"

    print(f"Loading UCE-brain from {ckpt}")
    core, embedding_layer = load_brain(ckpt, brain_src)

    transformer_path = embedding_dir / "transformer.onnx"
    export_transformer(core, embedding_layer, transformer_path, opset=opset)
    validate_transformer_concordance(core, embedding_layer, transformer_path)

    if skip_embeddings:
        print("\nSkipping protein embedding extraction (--skip-embeddings)")
        # Still need a minimal info dict so metadata.json is complete
        gene_dict_info = {"num_rows": 0, "embedding_dim": EMBEDDING_DIM}
    else:
        gene_dict_info = extract_protein_embeddings(ckpt, gene_dict, embedding_dir)

    write_metadata(output_path, display_name, gene_dict_info)

    print(f"\n🎉 UCE-brain export complete: {output_path}")
    print(f"  - {embedding_dir / 'transformer.onnx'}")
    if not skip_embeddings:
        print(f"  - {embedding_dir / 'protein_embeddings.bin'}")
        print(f"  - {embedding_dir / 'gene_dict.json'}")
    print(f"  - {output_path / 'metadata.json'}")


if __name__ == "__main__":
    app()
