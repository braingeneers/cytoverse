"""Compute the exact-kNN vs IVFPQ comparison behind Figures 5 and 6, and cache it.

The analysis notebook holds these results only in kernel memory, so the figures
could not be regenerated without re-running the whole notebook by hand. This
script runs the same pipeline once and writes a cache that the figure scripts
read, which makes Figs 5 and 6 reproducible from a clean checkout.

    uv run scripts/figures/compute_ivfpq_metrics.py

Ported from notebooks/analysis.ipynb cells 2, 10 and 11.
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ivfpq import IVFPQ  # noqa: E402
from scimilarity import CellAnnotation  # noqa: E402
from scimilarity.utils import align_dataset, lognorm_counts  # noqa: E402

ADATA_PATH = Path("tests/fixtures/GSE136831_subsample_100.h5ad")
MODEL_PATH = Path("data/models/scimilarity/model_v1.1")
LABELS_PATH = Path("data/references/scimilarity/labels.parquet")
OUT_PATH = Path("paper/figures/data/ivfpq_metrics.pkl")

K_NEIGHBORS = 50
N_SUBS = [8, 16, 32]
N_PROBES = [1, 2, 3, 4, 6, 8, 16, 32]


def folder_size(path) -> int:
    return sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(path)
        for f in files
    )


def knn_baseline(adata, k):
    ann = CellAnnotation(str(MODEL_PATH), use_gpu=False)
    aligned = align_dataset(adata, ann.gene_order)
    normalized = lognorm_counts(aligned)
    embeddings = ann.get_embeddings(normalized.X)

    start = time.time()
    _, nn_idxs, nn_dists, _ = ann.get_predictions_knn(
        embeddings, k=k, disable_progress=True, weighting=True
    )
    elapsed = time.time() - start
    return embeddings, nn_idxs, nn_dists, elapsed


def ivfpq_search(embeddings, model_path, k, n_probe):
    ivf = IVFPQ.load(model_path)
    idxs, dists = [], []
    loaded_paths: set = set()

    start = time.time()
    for embedding in embeddings:
        assert np.isclose(np.linalg.norm(embedding), 1.0)
        # NB: the notebook calls this with n_partitions_search=, which no longer
        # exists -- scripts/ivfpq.py takes (query_vector, model_path, n_probe, k).
        # analysis.ipynb cell 11 would fail against the current API.
        vector_ids, distances, paths = ivf.search(
            torch.from_numpy(embedding).float(),
            model_path,
            n_probe=n_probe,
            k=k,
        )
        idxs.append(np.asarray(vector_ids))
        dists.append(np.asarray(distances))
        loaded_paths.update(paths)
    elapsed = time.time() - start

    return np.vstack(idxs), np.vstack(dists), elapsed, loaded_paths


def main() -> None:
    adata = anndata.read_h5ad(ADATA_PATH)
    reference_labels = pd.read_parquet(LABELS_PATH)
    print(f"Loaded {adata.shape[0]} query cells from {ADATA_PATH.name}")

    embeddings, knn_idxs, knn_dists, knn_time = knn_baseline(adata, K_NEIGHBORS)
    knn_storage = folder_size(MODEL_PATH / "annotation")
    print(f"Exact kNN: {knn_time:.1f}s, index {knn_storage / 2**20:.0f} MiB")

    ivfpq_data = {}
    for n_sub in N_SUBS:
        model_path = Path(f"data/ivfpq{n_sub}x256")
        storage = folder_size(model_path)
        for n_probe in N_PROBES:
            gc.collect()
            idxs, dists, elapsed, paths = ivfpq_search(
                embeddings, model_path, K_NEIGHBORS, n_probe
            )
            ivfpq_data[(n_sub, n_probe)] = {
                "ivfpq_idxs": idxs,
                "ivfpq_dists": dists,
                "ivfpq_time": elapsed,
                "ivfpq_bw": sum(os.path.getsize(p) for p in paths),
                "ivfpq_storage": storage,
            }
            print(f"  n_sub={n_sub:2d} n_probe={n_probe:2d}  {elapsed:6.1f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as fh:
        pickle.dump(
            {
                "knn_idxs": np.asarray(knn_idxs),
                "knn_dists": np.asarray(knn_dists),
                "knn_time": knn_time,
                "knn_storage": knn_storage,
                "ivfpq_data": ivfpq_data,
                "k_neighbors": K_NEIGHBORS,
                "n_subs": N_SUBS,
                "n_probes": N_PROBES,
                "n_queries": int(adata.shape[0]),
                "source": str(ADATA_PATH),
            },
            fh,
        )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
