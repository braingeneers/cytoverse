"""Figure 7 -- runtime and peak-memory scaling on an M4 MacBook Air, at 174 mm.

Ported from benchmark/m4/analyze.py. The analysis is unchanged; only sizing and
typography differ.

article.tex:255 quotes this figure's fitted constants, so the script re-derives
them and checks them against the manuscript rather than trusting the plot.

    uv run scripts/figures/fig7_performance_scaling.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from style import SMALL_PT, WIDTH_2COL, save, use_style  # noqa: E402

RESULTS_DIR = Path("benchmark/m4/results")
OUT = Path("paper/figures/fig7_performance_scaling.pdf")

COL = {"cpu": "#0072B2", "gpu": "#D55E00"}
LAB = {"cpu": "CPU / WASM", "gpu": "GPU / WebGPU"}

# Values asserted in article.tex:255 and :257.
CLAIMS = {
    "cpu_intercept_s": 13,
    "cpu_slope_s_per_1k": 14.6,
    "gpu_intercept_s": 27,
    "gpu_slope_s_per_1k": 12.1,
    "mem_base_gb": 2.7,
    "mem_kb_per_cell": 4.0,
}


def load_records(results_dir: Path):
    recs = []
    for f in sorted(glob.glob(str(results_dir / "*.json"))):
        stem = os.path.basename(f).replace(".json", "")
        # results/ also holds non-scaling runs (reset_100k_cpu.json from the
        # memory-release experiment); those do not lead with a cell count.
        if not stem.split("_")[0].isdigit():
            continue
        o = json.load(open(f))
        if not o.get("finished"):
            continue
        recs.append(
            dict(
                cells=int(stem.split("_")[0]),
                device=o["device"],
                wall_s=o["wall_s"],
                peak=o.get("peak", o.get("peak_renderer_rss_mb")),
            )
        )
    if not recs:
        raise SystemExit(f"No finished benchmark results in {results_dir}")
    return recs


def aggregate(recs, device):
    by: dict[int, list] = {}
    for r in recs:
        if r["device"] == device:
            by.setdefault(r["cells"], []).append(r)
    return [
        dict(
            cells=c,
            wall_s=float(np.median([x["wall_s"] for x in v])),
            peak=float(np.median([x["peak"] for x in v])),
        )
        for c, v in sorted(by.items())
    ]


def linfit(pts, key):
    x = np.array([p["cells"] for p in pts], float)
    y = np.array([p[key] for p in pts], float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return dict(
        slope=slope, intercept=intercept, r2=1 - ss_res / ss_tot if ss_tot else 1.0
    )


def main() -> None:
    recs = load_records(RESULTS_DIR)
    devs = sorted({r["device"] for r in recs})
    table = {d: aggregate(recs, d) for d in devs}
    wf = {d: linfit(table[d], "wall_s") for d in devs}
    mf = {d: linfit(table[d], "peak") for d in devs}

    use_style()
    fig, (axr, axm) = plt.subplots(
        1, 2, figsize=(WIDTH_2COL, WIDTH_2COL * 0.42), layout="constrained"
    )
    xr = np.linspace(10_000, 500_000, 100)

    for d in devs:
        pts = table[d]
        xmax = max(p["cells"] for p in pts)
        fit = wf[d]
        if d == "cpu":
            rates = [p["wall_s"] / p["cells"] for p in pts]
            rmin, rmax = min(rates), max(rates)
            b = fit["intercept"]
            axr.fill_between(
                xr, (b + rmin * xr) / 60, (b + rmax * xr) / 60,
                color=COL[d], alpha=0.15, zorder=1,
                label=f"cooling band (+{rmax / rmin - 1:.0%})",
            )
        axr.scatter(
            [p["cells"] for p in pts], [p["wall_s"] / 60 for p in pts],
            color=COL[d], s=8, zorder=5, label=f"{LAB[d]} (measured)",
        )
        solid = xr[xr <= xmax]
        axr.plot(solid, (fit["intercept"] + fit["slope"] * solid) / 60,
                 "-", color=COL[d], lw=1.0)

    axr.set_xlabel("Query cells")
    axr.set_ylabel("Wall-clock runtime (min)")
    axr.grid(alpha=0.3, linewidth=0.4)
    axr.legend(loc="lower right")
    axr.ticklabel_format(style="plain", axis="x")
    axr.annotate(
        f"CPU: {wf['cpu']['intercept']:.0f}s + {wf['cpu']['slope'] * 1000:.1f} s/1k "
        f"(R\u00b2={wf['cpu']['r2']:.4f})\n"
        f"500k measured: {table['cpu'][-1]['wall_s'] / 60:.0f} min",
        # Anchored to the top-left corner with va="top" so the box hangs
        # downward a predictable amount regardless of how many lines it has.
        # Both curves rise left-to-right, leaving this corner and the lower
        # right free -- hence the legend moving to "lower right".
        xy=(0.04, 0.96), xycoords="axes fraction", fontsize=SMALL_PT,
        va="top", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9, lw=0.5),
    )

    for d in devs:
        pts = table[d]
        axm.plot(
            [p["cells"] for p in pts], [p["peak"] / 1024 for p in pts],
            "o-", color=COL[d], ms=2.5, lw=1.0, label=f"{LAB[d]} peak RSS",
        )
    ytop = np.ceil(max(p["peak"] for d in devs for p in table[d]) / 1024 * 1.25)
    axm.set_ylim(0, ytop)
    axm.set_xlabel("Query cells")
    axm.set_ylabel("Peak renderer RSS (GB)")
    axm.grid(alpha=0.3, linewidth=0.4)
    axm.legend(loc="lower right")
    axm.ticklabel_format(style="plain", axis="x")
    axm.annotate(
        f"~{mf['cpu']['intercept'] / 1024:.1f} GB base + "
        f"{mf['cpu']['slope'] * 1024:.1f} KB/cell\n"
        f"500k: {table['cpu'][-1]['peak'] / 1024:.1f} GB\n"
        f"(well under the ~16 GB per-tab budget)",
        xy=(0.04, 0.96), xycoords="axes fraction", fontsize=SMALL_PT,
        va="top", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9, lw=0.5),
    )

    for ax in (axr, axm):
        ax.xaxis.get_offset_text().set_size(SMALL_PT)

    save(fig, OUT, WIDTH_2COL)
    plt.close(fig)

    # Cross-check the manuscript's quoted constants.
    got = {
        "cpu_intercept_s": wf["cpu"]["intercept"],
        "cpu_slope_s_per_1k": wf["cpu"]["slope"] * 1000,
        "gpu_intercept_s": wf["gpu"]["intercept"],
        "gpu_slope_s_per_1k": wf["gpu"]["slope"] * 1000,
        "mem_base_gb": mf["cpu"]["intercept"] / 1024,
        "mem_kb_per_cell": mf["cpu"]["slope"] * 1024,
    }
    print("\n  fitted vs article.tex:255-257")
    worst = 0.0
    for key, claimed in CLAIMS.items():
        actual = got[key]
        delta = abs(actual - claimed) / max(abs(claimed), 1e-9)
        worst = max(worst, delta)
        flag = "ok" if delta < 0.05 else "MISMATCH"
        print(f"    {key:22s} paper {claimed:>7.1f}   fitted {actual:>7.1f}   {flag}")
    print(f"    runtime R2  cpu {wf['cpu']['r2']:.4f}  gpu {wf['gpu']['r2']:.4f}")
    print(f"    memory  R2  cpu {mf['cpu']['r2']:.4f}")
    if worst >= 0.05:
        raise SystemExit("Fitted constants disagree with the manuscript by >5%")


if __name__ == "__main__":
    main()
