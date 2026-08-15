# CytoVerse M4 Benchmark — Results (metaatlas query → full SCimilarity)

**Machine:** MacBook Air M4 (fanless). **App:** committed production build, onnxruntime-web **1.27**
(WebGPU with runtime WASM fallback), commit `879fb03`, rebuilt for these runs.
**Reference:** full SCimilarity — ~25M-cell reference, **815 MB** PQ index (4,835 partitions),
28,230-gene encoder. **Query:** deterministic (seed 42) subsamples of the 599,211-cell metaatlas
(`adata_metaatlas_final_raw.h5ad`), 10k→500k cells. Labels are not the point — this is a
runtime/memory scaling study. Each run reloads the full ~2 GB model+index (the fixed intercept).

Driver: `m4bench.mjs` (Playwright, **headed**), renderer-process RSS sampled via macOS `ps`,
wall-clock timed in-browser (MutationObserver on the status element). Raw per-run JSON in `results/`.

## Summary table

| cells | device | n | wall (s) | s / 1k | wall | peak RSS | at-rest RSS |
|------:|--------|--:|---------:|-------:|-----:|---------:|------------:|
| 10k   | CPU/WASM   | 3 | 135.0 | 13.50 | 2m15s | 2,776 MB | 2,658 MB |
| 25k   | CPU/WASM   | 1 | 345.0 | 13.80 | 5m45s | 2,915 MB | 2,690 MB |
| 50k   | CPU/WASM   | 1 | 722.6 | 14.45 | 12m03s | 3,010 MB | 2,941 MB |
| 100k  | CPU/WASM   | 1 | 1,511.8 | 15.12 | 25m12s | 3,264 MB | 3,210 MB |
| 250k  | CPU/WASM   | 1 | 3,753.7 | 15.01 | 62m34s | 3,846 MB | 3,784 MB |
| 500k  | CPU/WASM   | 1 | 7,271.7 | 14.54 | **2h01m** | **4,726 MB** | 4,564 MB |
| 10k   | GPU/WebGPU | 3 | 114.0 | 11.40 | 1m54s | 2,712 MB | 2,606 MB |
| 25k   | GPU/WebGPU | 1 | 303.1 | 12.12 | 5m03s | 2,818 MB | 2,701 MB |
| 50k   | GPU/WebGPU | 1 | 641.1 | 12.82 | 10m41s | 2,960 MB | 2,802 MB |
| 100k  | GPU/WebGPU | 1 | 1,314.1 | 13.14 | 21m54s | 3,239 MB | 3,101 MB |
| 250k  | GPU/WebGPU | 1 | 3,041.7 | 12.17 | 50m41s | 3,775 MB | 3,711 MB |
| 500k  | GPU/WebGPU | 1 | 6,083.9 | 12.17 | **1h41m** | 4,516 MB | 4,355 MB |

All runs finished with full cell counts and no WASM fallback (WebGPU genuinely engaged on GPU runs).
10k has 3 repeats (variance ±4%); large sizes are single long runs. Fits below.
**Both execution paths are now measured at all six query sizes — nothing is extrapolated.** The
GPU 250k/500k points were added after reviewer 2 objected to the projected curve in the original
figure; they are the same harness, same machine, same build, run later on the same Air.

## Findings

**1. Runtime scales linearly; there is no 2× "hot" regime.** A single line fits the whole
10k→500k range: **wall ≈ 13 s + 14.61 s per 1,000 cells** (CPU/WASM, R² = 0.9997);
**wall ≈ 27 s + 12.13 s/1k** (GPU/WebGPU, R² = 0.9997). Per-1k rate varies only **±12%** across the
entire range (CPU 13.5–15.1 s/1k) — sustained thermal throttling on the fanless Air costs ~10%,
not the ~2× seen on the older sspsygene runs. 500k cells label in **2.0 h** on CPU and **1.7 h** on
GPU, both measured directly. The small intercept is the fixed model+index load.

**2. Memory grows gently and linearly — it is bounded, not flat.** Peak renderer RSS =
**~2.7 GB fixed base + ~4.0 KB per query cell** (CPU, R² = 0.995); GPU is close behind at
**~3.7 KB/cell** (R² = 0.985). 500k cells peak at **4.7 GB** (CPU) and **4.5 GB** (GPU) —
comfortably within a 16 GB laptop. So the R2-a answer is: memory does scale with query size, but
gently; against a **23M-cell** reference the working set stays a small fraction of laptop RAM even
at half a million query cells.

**3. GPU ≈ CPU on Apple Silicon (WebGPU ~1.17× faster).** CPU/GPU wall ratio is a stable
**1.13–1.23×** across all six sizes (GPU faster, mean 1.17×). WebGPU genuinely runs the encoder (no
fallback) but is transfer-bound, so it ties multithreaded WASM to within ~20%. The GPU path is also
marginally *lighter* in peak RSS at every size, not heavier — the earlier "~800 MB more" and
"~5.9 KB/cell" claims were artifacts of fitting only the four small sizes, and are withdrawn.
All GPU sizes are now measured; nothing is projected.

## Caveats (for the figure caption)

- Run on a **MacBook Air M4** (fanless). Runtime is only mildly cooling-dependent here (~10%
  sustained-throttle spread) — much less than anticipated; an actively-cooled M4/Pro would sit at
  the low end of that band.
- **GPU ≈ CPU** on Apple Silicon: WebGPU is at rough parity (~1.17× faster, 1.13–1.23× across
  sizes), the encoder being transfer-bound. This is the correct result, consistent with the discrete-NVIDIA finding that
  WebGPU does not beat multithreaded WASM for this model.
- **Memory is bounded** (~2.7 GB base + ~4 KB/cell; 4.7 GB at 500k) against a 23M-cell reference —
  it scales with query size but stays well under laptop RAM.
- Large sizes (100k/250k/500k) are **single runs** (n=1); 10k repeats bound variance at ±4% and the
  R²≈0.9997 linear fit confirms consistency.

## Freeing memory (reviewer note)

After a run the tab holds the loaded model, the 815 MB IVFPQ index, and the labeled results — peak
renderer RSS ~2.7 GB + ~4 KB/cell (e.g. ~3.2 GB at 100k, ~4.7 GB at 500k). **To release all of it
and return to the ~0.9 GB baseline, reload the page** (to be noted in the app README).

We prototyped an in-app soft reset (terminate the worker + clear results) and measured it
(`m4bench_reset.mjs`, raw JSON in `results/reset_100k_cpu.json`) before deciding a page reload is the
cleaner reviewer answer:

| query | baseline | after run | after soft reset | released | residual over baseline |
|------:|---------:|----------:|-----------------:|---------:|-----------------------:|
| 10k   | 892 MB   | 2,641 MB  | 1,430 MB | 1,211 MB | 538 MB |
| 100k  | 899 MB   | 3,159 MB  | 2,319 MB |   840 MB | 1,420 MB |

Terminating the worker returns its WASM heap (encoder + index + h5wasm) to the OS, but the **main
renderer heap grows with query size and persists** (V8/WebGL high-water: ~0.5 GB at 10k → ~1.4 GB at
100k), so a soft reset only partially reclaims the tab — hence **reload** for a full return to
baseline. (This is why the button was dropped in favor of the reload note.)

## Corrections vs. the original PLAN.md assumptions

- Plan expected a **~2× throttled regime** and a cold/hot band → **not observed**; throttle is ~10%,
  runtime is one clean line.
- Plan expected **flat ~2 GB memory** → memory actually **grows** to 4.7 GB at 500k (~4 KB/cell).
  The earlier "flat ~2 GB" was partly the headless 1,000-cell read bug (see below).
- Plan expected **GPU ≈ CPU parity** → **confirmed** (GPU ~1.17× faster across all six sizes).
- **Harness bug found & fixed:** headless Chromium truncates the h5ad query read to 1,000 cells, so
  headless runs silently under-process (flat 22 s regardless of size). All runs here are **headed**.
