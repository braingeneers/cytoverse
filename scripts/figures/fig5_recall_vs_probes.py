"""Figure 5 -- IVFPQ recall vs partitions probed, sized for 114 mm (1.5 column).

Authored at final print width so the type lands where it is specified. See
scripts/figures/style.py for why we avoid bbox_inches="tight".

    uv run scripts/figures/fig5_recall_vs_probes.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from style import WIDTH_1_5COL, save, use_style  # noqa: E402

CACHE = Path("paper/figures/data/ivfpq_metrics.pkl")
OUT = Path("paper/figures/fig5_recall_vs_probes.pdf")

# Distinct in both color and marker so the figure survives grayscale printing.
SERIES = [
    (8, "#0072B2", "o"),
    (16, "#009E73", "s"),
    (32, "#D55E00", "^"),
]


def mean_recall(knn_idxs, ivfpq_idxs, k) -> float:
    return float(
        np.mean(
            [
                len(set(knn_idxs[q]) & set(ivfpq_idxs[q])) / k
                for q in range(knn_idxs.shape[0])
            ]
        )
    )


def main() -> None:
    with CACHE.open("rb") as fh:
        d = pickle.load(fh)

    knn_idxs = d["knn_idxs"]
    k = d["k_neighbors"]
    n_probes = d["n_probes"]

    use_style()
    fig, ax = plt.subplots(
        figsize=(WIDTH_1_5COL, WIDTH_1_5COL * 0.62), layout="constrained"
    )

    for n_sub, color, marker in SERIES:
        recalls = [
            mean_recall(knn_idxs, d["ivfpq_data"][(n_sub, p)]["ivfpq_idxs"], k)
            if (n_sub, p) in d["ivfpq_data"]
            else np.nan
            for p in n_probes
        ]
        ax.plot(
            n_probes,
            recalls,
            marker=marker,
            color=color,
            label=f"n_sub = {n_sub}",
            linewidth=1.0,
            markersize=3,
        )

    ax.axhline(y=1.0, color="0.2", linestyle="--", linewidth=0.8, label="Exact kNN")

    ax.set_xlabel("Partitions probed (n_probe)")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_probes)
    ax.set_xticklabels([str(p) for p in n_probes])
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(loc="lower right", ncol=2)

    save(fig, OUT, WIDTH_1_5COL)
    plt.close(fig)


if __name__ == "__main__":
    main()
