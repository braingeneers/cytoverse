#!/usr/bin/env python
# M4 benchmark: summary table + the combined figure (runtime + memory scaling) from results/*.json.
# Corrected story: runtime is ~LINEAR (mild ~10% throttle, no 2x regime); memory grows ~linearly
# with query size; GPU/WebGPU ~= CPU/WASM (GPU ~1.15x faster). GPU measured to 100k, projected beyond.
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(DIR, "results")

recs = []
for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    o = json.load(open(f))
    b = os.path.basename(f).replace(".json", "")
    cells = int(b.split("_")[0])
    if not o.get("finished"):
        continue
    ep = "WebGPU" if "WebGPU" in o.get("embedding_ep_log", "") else "WASM"
    recs.append(dict(cells=cells, device=o["device"], ep=ep, wall_s=o["wall_s"],
                     peak=o["peak"] if "peak" in o else o["peak_renderer_rss_mb"],
                     at_rest=o["at_rest_renderer_rss_mb"]))
if not recs:
    print("no finished results yet"); sys.exit(0)

devs = sorted(set(r["device"] for r in recs))

def agg(device):
    by = {}
    for r in recs:
        if r["device"] != device: continue
        by.setdefault(r["cells"], []).append(r)
    return [dict(cells=c, n=len(v),
                 wall_s=float(np.median([x["wall_s"] for x in v])),
                 peak=float(np.median([x["peak"] for x in v])),
                 at_rest=float(np.median([x["at_rest"] for x in v]))) for c, v in sorted(by.items())]

table = {d: agg(d) for d in devs}

def linfit(pts, key):
    x = np.array([p["cells"] for p in pts], float)
    y = np.array([p[key] for p in pts], float)
    s, b = np.polyfit(x, y, 1)
    yhat = s*x + b
    ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot else 1.0
    return dict(slope=s, intercept=b, r2=r2)

# ---- table + CSV ----
print(f"\n{'cells':>7} {'dev':>4} {'n':>2} {'wall_s':>8} {'s/1k':>6} {'peak_MB':>8} {'atrest_MB':>9}")
csv = ["cells,device,n_repeats,wall_s,s_per_1k,peak_rss_mb,at_rest_rss_mb"]
for d in devs:
    for r in table[d]:
        print(f"{r['cells']:>7} {d:>4} {r['n']:>2} {r['wall_s']:>8.1f} "
              f"{r['wall_s']/(r['cells']/1000):>6.2f} {r['peak']:>8.0f} {r['at_rest']:>9.0f}")
        csv.append(f"{r['cells']},{d},{r['n']},{r['wall_s']:.1f},{r['wall_s']/(r['cells']/1000):.2f},{r['peak']:.0f},{r['at_rest']:.0f}")
open(os.path.join(RES, "summary.csv"), "w").write("\n".join(csv) + "\n")

print("\n--- single linear fits ---")
wf, mf = {}, {}
for d in devs:
    wf[d] = linfit(table[d], "wall_s"); mf[d] = linfit(table[d], "peak")
    print(f"  {d:>4} runtime: {wf[d]['intercept']:7.1f} s + {wf[d]['slope']*1000:6.3f} s/1k  (R2={wf[d]['r2']:.5f})")
    print(f"  {d:>4} memory : {mf[d]['intercept']:7.0f} MB + {mf[d]['slope']*1000:6.3f} MB/1k = "
          f"{mf[d]['slope']*1024:.2f} KB/cell  (R2={mf[d]['r2']:.5f})")

# throttle envelope (CPU, all sizes)
rates = [p["wall_s"]/(p["cells"]/1000) for p in table["cpu"]]
print(f"\n  CPU s/1k range {min(rates):.2f}-{max(rates):.2f} -> throttle spread {max(rates)/min(rates)-1:+.0%}")
if "gpu" in table:
    gm = {p["cells"]: p["wall_s"] for p in table["gpu"]}
    ratios = [p["wall_s"]/gm[p["cells"]] for p in table["cpu"] if p["cells"] in gm]
    print(f"  CPU/GPU ratio {np.mean(ratios):.3f} (GPU faster), shared sizes {sorted(gm)}")

# ---- figure: runtime (left) + memory (right) ----
COL = {"cpu": "#0072B2", "gpu": "#D55E00"}
LAB = {"cpu": "CPU / WASM", "gpu": "GPU / WebGPU"}
fig, (axr, axm) = plt.subplots(1, 2, figsize=(12.5, 5.2))
xr = np.linspace(10000, 500000, 100)

for d in devs:
    pts = table[d]; xmax = max(p["cells"] for p in pts)
    x = [p["cells"] for p in pts]; y = [p["wall_s"]/60 for p in pts]
    fit = wf[d]
    # cooling-dependent band: coolest vs most-throttled observed per-cell rate about the fixed load
    # intercept. Its narrowness (~12%) IS the finding — fanless Air throttles only mildly here.
    if d == "cpu":
        rates = [p["wall_s"]/p["cells"] for p in pts]
        rmin, rmax = min(rates), max(rates); b = fit["intercept"]
        axr.fill_between(xr, (b+rmin*xr)/60, (b+rmax*xr)/60, color=COL[d], alpha=.15, zorder=1,
                         label=f"cooling-dependent band (+{rmax/rmin-1:.0%}: cool ↔ throttled)")
    axr.scatter(x, y, color=COL[d], s=48, zorder=5, label=f"{LAB[d]} (measured)")
    solid = xr[xr <= xmax]; dash = xr[xr >= xmax]
    axr.plot(solid, (fit["intercept"]+fit["slope"]*solid)/60, "-", color=COL[d], lw=1.8)
    if len(dash) > 1 and xmax < xr.max():
        axr.plot(dash, (fit["intercept"]+fit["slope"]*dash)/60, ":", color=COL[d], lw=1.8,
                 label=f"{LAB[d]} (projected)")
axr.set_xlabel("query cells"); axr.set_ylabel("wall-clock runtime (min)")
axr.set_title("Runtime scales linearly (mild ~10% throttle)")
axr.grid(alpha=.3); axr.legend(fontsize=8, loc="upper left"); axr.ticklabel_format(style="plain", axis="x")
axr.annotate(f"CPU: {wf['cpu']['intercept']:.0f}s + {wf['cpu']['slope']*1000:.1f} s/1k  (R²={wf['cpu']['r2']:.4f})\n"
             f"500k measured: {table['cpu'][-1]['wall_s']/60:.0f} min",
             xy=(.04,.62), xycoords="axes fraction", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=.9))

for d in devs:
    pts = table[d]
    axm.plot([p["cells"] for p in pts], [p["peak"]/1024 for p in pts], "o-",
             color=COL[d], ms=5, lw=1.4, label=f"{LAB[d]} peak RSS")
axm.axhline(16, color="#888", ls="--", lw=1); axm.text(12000, 16.2, "16 GB laptop RAM", fontsize=8, color="#555")
axm.set_ylim(0, 17); axm.set_xlabel("query cells"); axm.set_ylabel("peak renderer RSS (GB)")
axm.set_title("Memory grows gently with query size")
axm.grid(alpha=.3); axm.legend(fontsize=8, loc="upper left"); axm.ticklabel_format(style="plain", axis="x")
axm.annotate(f"~{mf['cpu']['intercept']/1024:.1f} GB base + {mf['cpu']['slope']*1024:.1f} KB/cell\n"
             f"500k: {table['cpu'][-1]['peak']/1024:.1f} GB (well under 16 GB)",
             xy=(.04,.70), xycoords="axes fraction", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=.9))

fig.suptitle("CytoVerse on MacBook Air M4 — full SCimilarity reference (25M cells, 815 MB index), metaatlas query",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(RES, "m4_benchmark_figure.png")
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
