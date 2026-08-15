"""Figure 6 -- latent-space distortion vs recall, sized for 174 mm.

The original could not be fixed by rescaling. At 174 mm its four columns leave
~38 mm per panel, and every panel carried its own y-axis plus five x-tick labels
reading "0.0000 / 0.0005 / 0.0010" -- twelve characters in a 38 mm panel is
unreadable at any dpi.

Restructured per paper/FIGURES.md:
  * one shared x label per column and y label per row (was: every panel)
  * 3 ticks per panel with a shared scientific-notation exponent (was: 5 long
    decimal labels per panel)
  * shared x limits per column so panels are visually comparable
  * a single colorbar

    uv run scripts/figures/fig6_distortion_grid.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.append(str(Path(__file__).resolve().parent))

from style import SMALL_PT, WIDTH_2COL, save, use_style  # noqa: E402

CACHE = Path("paper/figures/data/ivfpq_metrics.pkl")
OUT = Path("paper/figures/fig6_distortion_grid.pdf")

PROBES = [1, 3, 6, 16]  # subset shown; full sweep lives in the cache


def metrics_per_query(knn_idxs, knn_dists, ivfpq_idxs, ivfpq_dists, k):
    """Recall@k and mean absolute distance distortion for matched neighbours."""
    recalls, distortions = [], []
    for q in range(knn_idxs.shape[0]):
        true_idx, approx_idx = knn_idxs[q], ivfpq_idxs[q]
        matched = set(true_idx) & set(approx_idx)
        recalls.append(len(matched) / k)
        if matched:
            kpos = {v: i for i, v in enumerate(true_idx)}
            ipos = {v: i for i, v in enumerate(approx_idx)}
            diffs = [
                abs(knn_dists[q][kpos[m]] - ivfpq_dists[q][ipos[m]]) for m in matched
            ]
            distortions.append(float(np.mean(diffs)))
        else:
            distortions.append(np.nan)
    return np.asarray(recalls), np.asarray(distortions)


def main() -> None:
    with CACHE.open("rb") as fh:
        d = pickle.load(fh)

    knn_idxs, knn_dists = d["knn_idxs"], d["knn_dists"]
    k, n_subs = d["k_neighbors"], d["n_subs"]
    nn_dist = np.mean(knn_dists, axis=1)

    use_style()
    nrow, ncol = len(n_subs), len(PROBES)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(WIDTH_2COL, WIDTH_2COL * 0.52),
        sharex="col",
        sharey=True,
        layout="constrained",
    )

    vmin, vmax = float(nn_dist.min()), float(nn_dist.max())
    scatter = None

    for i, n_sub in enumerate(reversed(n_subs)):
        for j, n_probe in enumerate(PROBES):
            ax = axes[i, j]
            entry = d["ivfpq_data"].get((n_sub, n_probe))
            if entry is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            recalls, distortions = metrics_per_query(
                knn_idxs, knn_dists, entry["ivfpq_idxs"], entry["ivfpq_dists"], k
            )
            # Plot distortion in units of 1e-3 so the exponent lives in the axis
            # label at full size. matplotlib's offset text is mathtext, whose
            # superscript renders at 0.7x -- that put 4.2 pt type in the figure.
            # (Arial also lacks U+207B, so a literal "x10^-3" would box-glyph.)
            scatter = ax.scatter(
                distortions * 1e3, recalls, c=nn_dist, cmap="viridis",
                alpha=0.7, s=2.0, vmin=vmin, vmax=vmax, linewidths=0,
            )
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.3)

            # 3 ticks, and a shared exponent in the corner rather than five
            # long decimals repeated under every panel.
            # prune=None so every column keeps its leading 0 and the panels read
            # consistently; 3 bins of short labels do not collide.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

            if i == 0:
                ax.set_title(f"n_probe = {n_probe}", pad=3)
            # One y label per ROW, one x label per COLUMN -- not per panel.
            if j == 0:
                ax.set_ylabel(f"n_sub = {n_sub}\nRecall@{k}")
            if i == nrow - 1:
                ax.set_xlabel("Distance distortion \u00d7 10\u00b3")

    cbar = fig.colorbar(scatter, ax=axes, location="right", fraction=0.025, pad=0.01)
    cbar.set_label("Mean kNN distance")
    cbar.ax.tick_params(labelsize=SMALL_PT)
    cbar.outline.set_linewidth(0.6)

    save(fig, OUT, WIDTH_2COL)
    plt.close(fig)


if __name__ == "__main__":
    main()
