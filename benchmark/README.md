# CytoVerse benchmarks — entry point (for the reviewer response)

This directory holds the benchmark evidence for the Cell Patterns revision (PATTERNS-D-26-00158),
reviewer points **R2-a** (memory stress test) and **R2-b** (runtime / peak-memory scaling vs #cells),
plus the CPU-vs-GPU question (**R1-4**).

**The final, authoritative study is in [`m4/`](m4/).** Everything else — the earlier sspsygene
harness, the discrete-NVIDIA (k8s) runs, and the WebGPU/BatchNorm microbenchmarks — is superseded and
moved to [`archive/`](archive/) (still useful as corroboration; see the last section).

To generate the reviewer response you need only this README plus [`m4/RESULTS.md`](m4/RESULTS.md) and
[`m4/results/`](m4/results/).

---

## What was measured

- **App:** the committed CytoVerse production build (onnxruntime-web **1.27**, WebGPU encoder with a
  runtime WASM fallback; commit `879fb03` on branch `benchmark`), served with `vite preview`.
- **Reference:** the **full SCimilarity** index — ~**23 M-cell** reference, **815 MB** product-quantized
  index (4,835 partitions), 28,230-gene encoder. Loaded fresh in every run (the fixed intercept).
- **Query:** deterministic (seed 42) subsamples of the 599,211-cell metaatlas
  (`adata_metaatlas_final_raw.h5ad`), **10k → 500k cells**. This is a runtime/memory study — labels
  are not evaluated.
- **Machine:** MacBook Air **M4** (fanless — the consumer-device reference).
- **Driver:** `m4/m4bench.mjs` (Playwright, **headed** — headless truncates the h5ad read to 1,000
  cells), renderer-process RSS sampled via macOS `ps`, wall-clock timed in-browser.

## Results (headline)

Runtime and peak renderer RSS, full cell counts, no WASM fallback (median over repeats; 10k n=3,
larger sizes n=1). Full table: [`m4/results/summary.csv`](m4/results/summary.csv); figure:
[`m4/results/m4_benchmark_figure.png`](m4/results/m4_benchmark_figure.png).

| cells | CPU/WASM wall | CPU peak RSS | GPU/WebGPU wall | GPU peak RSS |
|------:|--------------:|-------------:|----------------:|-------------:|
| 10k   | 135 s (2m15s) | 2,776 MB | 114 s | 2,712 MB |
| 25k   | 345 s | 2,915 MB | 303 s | 2,818 MB |
| 50k   | 723 s | 3,010 MB | 641 s | 2,960 MB |
| 100k  | 1,512 s (25m) | 3,264 MB | 1,314 s | 3,239 MB |
| 250k  | 3,754 s (63m) | 3,846 MB | 3,042 s (51m) | 3,775 MB |
| 500k  | 7,272 s (**2.0 h**) | **4,726 MB** | 6,084 s (**1.7 h**) | 4,516 MB |

**Three findings:**

1. **Runtime is linear in query size.** `wall ≈ 13 s + 14.6 s per 1,000 cells` (CPU/WASM,
   **R²=0.9997**); `27 s + 12.1 s/1k` (GPU/WebGPU, **R²=0.9997**). Sustained thermal throttling on the
   fanless Air costs only **~12%** (per-1k rate 13.5–15.1 s across the whole range) — there is **no
   ~2× "hot" regime**. 500k cells were measured directly (2.0 h), so no extrapolation is needed for
   runtime.

2. **Peak memory is bounded and grows gently.** `~2.7 GB fixed base + ~4.0 KB per query cell`
   (CPU; GPU ~3.7 KB/cell — marginally lighter, not heavier). 500k cells peak at **4.7 GB** — well under a 16 GB
   laptop, against a **23 M-cell** reference. This is the R2-a answer: memory *does* scale with query
   size, but gently, and stays a small fraction of laptop RAM even at half a million cells.

3. **GPU ≈ CPU on Apple Silicon (WebGPU ~1.17× faster).** CPU/GPU wall ratio is a stable
   **1.13–1.23×** (GPU faster) across all six sizes, 10k–500k. WebGPU genuinely runs the encoder (verified, no
   fallback) but is transfer-bound, so it ties multithreaded WASM. Every GPU size is measured;
   nothing is projected. This matches the discrete-NVIDIA result (below).

## Freeing memory (reviewer note)

After a run the tab holds the model + 815 MB index + labeled results (~3.2 GB at 100k, ~4.7 GB at
500k). **Reloading the page releases everything and returns to the ~0.9 GB baseline.** (We prototyped
an in-app soft reset that terminates the worker; it reclaims the model+index — ~0.84 GB at 100k — but
not the main-thread V8/WebGL high-water, so a reload is the reliable full-release path. Measurements
in [`m4/RESULTS.md`](m4/RESULTS.md) → "Freeing memory".)

## Where things are

| Path | What |
|---|---|
| [`m4/RESULTS.md`](m4/RESULTS.md) | Full write-up: table, fits, caveats, plan-corrections, memory-release detail. **Read this next.** |
| [`m4/results/summary.csv`](m4/results/summary.csv) | Tidy per-point table (cells, device, wall, s/1k, peak/at-rest RSS). |
| [`m4/results/m4_benchmark_figure.png`](m4/results/m4_benchmark_figure.png) | The combined figure (runtime with cooling band + memory panel). |
| [`m4/results/*.json`](m4/results/) | Raw per-run driver output (incl. `reset_100k_cpu.json`). |
| [`m4/notebooks/analysis.ipynb`](m4/notebooks/analysis.ipynb) | Executed notebook (regenerates table + figure). |
| `m4/analyze.py` | Rebuilds `summary.csv` + figure from `results/*.json`. |
| `m4/m4bench.mjs`, `run_one.sh`, `drive.sh` | The run harness. |
| `m4/subsample_metaatlas.py` | Deterministic query generator (seed 42). |

## Caveats to carry into the response (stay honest)

- Run on a **fanless M4**; runtime is only mildly cooling-dependent here (~12% throttle spread), less
  than a MacBook Pro / actively-cooled M4 would be. The figure shows this as a "cooling-dependent band".
- **GPU ≈ CPU** — WebGPU is at rough parity (transfer-bound encoder), not a speedup. Correct result.
- **Memory scales with query size** (~4 KB/cell) on top of the ~2.7 GB fixed base — bounded, not flat.
- Large sizes are **single runs** (n=1); 10k repeats bound variance at ±4% and R²≈0.9997 confirms
  consistency.

## Corrections vs. the original plan (`archive/M4_PLAN.md`)

The original plan expected a ~2× throttled regime and flat ~2 GB memory; **neither held.** Runtime is
one clean linear line (~12% throttle); memory grows to 4.7 GB at 500k. GPU≈CPU parity was confirmed.
A harness bug was also found and fixed: **headless Chromium truncates the h5ad query read to 1,000
cells**, so all runs are driven headed.

## The `archive/` folder (superseded, but useful corroboration)

- `archive/k8s/` — **discrete-NVIDIA validation** (NRP Nautilus). On an RTX 3090 / A10 the encoder on
  WebGPU is GPU-idle and **WASM is ~1.4× faster** — independent confirmation that WebGPU is
  transfer-bound and CPU/WASM is the right default, on discrete GPUs as well as Apple Silicon. Useful
  if the R1-4 response wants cross-platform evidence. See `archive/k8s/results/RESULTS.md`.
- `archive/microbench/` — root-caused the earlier WebGPU failure (ort-web 1.22 BatchNorm-on-WebGPU
  bug, fixed by the 1.27 upgrade). Explains *why* the app now runs WebGPU at all.
- `archive/` (sspsygene: `DESIGN.md`, `HANDOFF.md`, `run.spec.ts`, `results/`, etc.) — the earlier
  599k-reference harness, superseded by the full-SCimilarity M4 study here.
