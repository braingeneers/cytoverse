"""
Vet the exported UCE-brain transformer.onnx outside the browser.

Runs the same forward pass the worker does — gather protein embeddings for a
sampled sentence, then feed (src, mask) to the ONNX session — using Python's
onnxruntime CPU provider. If this passes, the exported model is fine and the
browser failure is specific to onnxruntime-web's WebGPU backend.

Usage:
  uce-brain/.venv/bin/python uce-brain/scripts/vet_onnx.py \\
    uce-brain/artifacts
  # or against a promoted model dir:
  uce-brain/.venv/bin/python uce-brain/scripts/vet_onnx.py \\
    public/models/sspsygene-ucebrain/embedding
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PAD_LENGTH = 2048
SAMPLE_SIZE = 1024


def build_sentence(
    sample_indices: np.ndarray,
    chrom_order: np.ndarray,
    table_chroms: np.ndarray,
    table_starts: np.ndarray,
    table_protein_ids: np.ndarray,
    specials: dict,
    chrom_token_map: dict,
    pad_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.full(pad_length, specials["pad_token_idx"], dtype=np.int32)
    mask = np.zeros(pad_length, dtype=np.float32)
    token_ids[0] = specials["cls_token_idx"]
    mask[0] = 1.0
    cursor = 1

    by_chrom: dict[int, list[int]] = {}
    for idx in sample_indices:
        c = int(table_chroms[idx])
        by_chrom.setdefault(c, []).append(int(idx))

    for c in chrom_order:
        c = int(c)
        token_ids[cursor] = chrom_token_map[c]
        mask[cursor] = 1.0
        cursor += 1
        genes = sorted(by_chrom[c], key=lambda g: (table_starts[g], g))
        for g in genes:
            token_ids[cursor] = table_protein_ids[g]
            mask[cursor] = 1.0
            cursor += 1
        token_ids[cursor] = specials["chrom_token_right_idx"]
        mask[cursor] = 1.0
        cursor += 1

    return token_ids, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("embedding_dir", type=Path)
    ap.add_argument("--num-cells", type=int, default=3)
    ap.add_argument("--trim", action="store_true", help="Trim to valid seq_len (default: send full pad_length)")
    args = ap.parse_args()

    d = args.embedding_dir

    print(f"loading {d}/gene_dict.json")
    gd = json.loads((d / "gene_dict.json").read_text())
    num_rows = gd["num_rows"]
    emb_dim = gd["embedding_dim"]
    specials = gd["specials"]
    chrom_token_map = {int(k): int(v) for k, v in gd["chromosome_token_map"].items()}

    genes = gd["genes"]
    print(f"  num_rows={num_rows}  emb_dim={emb_dim}  genes={len(genes)}")

    sym_order = list(genes.keys())
    table_protein_ids = np.array([genes[s]["protein_embedding_id"] for s in sym_order], dtype=np.int32)
    table_chroms = np.array([genes[s]["chromosome_id"] for s in sym_order], dtype=np.int32)
    table_starts = np.array([genes[s]["location"] for s in sym_order], dtype=np.int64)

    print(f"loading {d}/protein_embeddings.bin")
    table = np.fromfile(d / "protein_embeddings.bin", dtype=np.float32)
    assert table.size == num_rows * emb_dim, f"{table.size} != {num_rows}*{emb_dim}"
    table = table.reshape(num_rows, emb_dim)
    print(f"  table shape: {table.shape}  (~{table.nbytes/1e6:.0f} MB)")

    print(f"creating CPU session for {d}/transformer.onnx")
    sess = ort.InferenceSession(str(d / "transformer.onnx"), providers=["CPUExecutionProvider"])
    print(f"  inputs:  {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"  outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

    rng = np.random.default_rng(0)
    uniform_weights = np.ones(len(sym_order), dtype=np.float32) / len(sym_order)

    for cell in range(args.num_cells):
        sample_indices = rng.choice(len(sym_order), size=SAMPLE_SIZE, replace=True, p=uniform_weights)
        uq = np.unique(table_chroms[sample_indices])
        uq_shuffled = uq.copy()
        rng.shuffle(uq_shuffled)

        token_ids, mask = build_sentence(
            sample_indices, uq_shuffled,
            table_chroms, table_starts, table_protein_ids,
            specials, chrom_token_map, PAD_LENGTH,
        )
        valid_len = int(mask.sum())
        print(f"\ncell {cell}: valid_len={valid_len}  (pad_length={PAD_LENGTH})")

        if args.trim:
            ids = token_ids[:valid_len]
            m = mask[:valid_len]
        else:
            ids = token_ids
            m = mask

        src = table[ids].reshape(1, len(ids), emb_dim).astype(np.float32)
        m_batch = m.reshape(1, -1).astype(np.float32)
        print(f"  src shape: {src.shape}  mask shape: {m_batch.shape}")

        t0 = time.perf_counter()
        out = sess.run(None, {"src": src, "mask": m_batch})
        ms = (time.perf_counter() - t0) * 1e3

        gene_emb, cell_emb = out
        print(f"  cell_embedding shape: {cell_emb.shape}  "
              f"norm={np.linalg.norm(cell_emb):.4f}  "
              f"{ms:.1f} ms")

    print("\nOK — exported transformer runs cleanly under Python CPU ORT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
