# uce-edge

Exploring whether [UCE](https://github.com/snap-stanford/UCE) (Universal Cell Embedding), a single-cell RNA-seq foundation model, can run efficiently on researchers' machines — including in a web browser via ONNX Runtime Web + WebGPU.

## Status

The full UCE-brain pipeline (log1p normalize → weighted sampling → chromosome ordering → CLS/CHROM/PAD inserts → protein embedding gather → 8-layer transformer) now runs end-to-end in the browser on WebGPU, validated against a Python reference at every stage. **111 ms/cell** in the browser (WebGPU) at seq_len=1072 vs **139 ms/cell** in native PyTorch (MPS) at the same shape — i.e. the browser is **1.26× faster than native PyTorch on the same Apple GPU**. See [plan.md](plan.md) for the phased build-out and remaining optimization roadmap.

## Repository structure

- **UCE/** — original UCE repo (submodule), 4-layer and 33-layer models
- **UCE-brain/** — newer [UCE-brain](https://github.com/snap-stanford/UCE-brain) repo (submodule), 8-layer model with smaller architecture (d_model=512 vs 1280)
- **model_files/** — pre-trained weights and supporting files (not checked in)
- **scripts/** — Python harnesses for baseline, ONNX export, quantization, and benchmarking
- **web/** — browser-based inference demo using ONNX Runtime Web

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone --recurse-submodules <repo-url> && cd uce-edge
uv sync
```

For browser benchmarks, install Playwright's Chromium:

```bash
.venv/bin/playwright install chromium
```

Model weights need to be placed in `model_files/`. The UCE-brain checkpoint downloads automatically from HuggingFace on first run:

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='KuanP/uce-brain-pilot-8l-512d', local_dir='model_files/uce-brain-pilot-8l-512d')
"
```

## Running

All experiments are available as Makefile targets:

```bash
# Original UCE (4-layer) — baseline, ONNX export, INT8 quantize, compare
make core-all

# UCE-brain (8-layer) — baseline, ONNX export, compare
make brain-all

# Automated browser benchmarks via Playwright (WebGPU + WASM)
make brain-web-bench

# Interactive browser demo (manual)
.venv/bin/python -m http.server 8765 -d web
# then open http://localhost:8765
```

End-to-end browser pipeline (Phases 0–6 — see [plan.md](plan.md)):

```bash
make web-install                 # one-time: npm deps + Playwright

# Phase 0-1: slice the embedding table, generate Python reference fixtures
make brain-extract-embeddings
make brain-reference-pipeline

# Each brain-phase* builds the web bundle and runs the Playwright validator
make brain-phase2                # transformer only (vs bit-exact reference)
make brain-phase3                # + browser gather
make brain-phase4                # + browser chrom-ordering / CLS-CHROM-PAD
make brain-phase5                # + browser weighted sampling
make brain-phase6                # + browser log1p + sum-to-1 normalize (full pipeline)

# Backend/options bench (WebGPU vs WASM, batching, int8, thread counts)
make brain-bench2
```

Individual steps:

| Target | Description |
|--------|-------------|
| `brain-baseline` | Run UCE-brain on MPS, save reference outputs |
| `brain-onnx-export` | Export to ONNX (opset 17, dynamo) |
| `brain-compare` | Compare PyTorch vs ONNX FP32 vs INT8 on CPU |
| `brain-web-bench` | Playwright-driven WebGPU + WASM benchmarks (synthetic) |
| `brain-phase{2..6}` | Phase-by-phase browser pipeline validation vs Python reference |
| `brain-bench2` | Backend/options benchmark (WebGPU batch size, WASM threads, int8) |

## Findings

### Architecture comparison

| | UCE original (4L) | UCE-brain (8L) |
|---|---|---|
| d_model | 1280 | 512 |
| Non-embedding params | 106M | 30M |
| ONNX FP32 size | 373 MB | 117 MB |
| ONNX INT8 size | 100 MB | 33 MB |

### Synthetic transformer-only benchmark (MacBook Air M4, 32GB)

UCE-brain 8-layer, seq_len=128, batch=1 (initial scoping benchmark, no preprocessing):

```
variant                  size      time      cosine vs reference
Python MPS (baseline)    —         215 ms    1.000000
Python ONNX FP32 CPU     117 MB    204 ms    1.000000
Python ONNX INT8 CPU      33 MB    201 ms    0.999672
Browser FP32 WebGPU      117 MB     14 ms    1.000000
Browser FP32 WASM        117 MB    143 ms    1.000000
Browser INT8 WebGPU       33 MB    173 ms    0.998644
Browser INT8 WASM         33 MB    145 ms    0.998706
```

### Full pipeline benchmark at real inference shape

End-to-end, raw counts → cell embedding, averaged over 100 cells from `allen-celltypes+human-cortex+m1-100.h5ad` (Phase 7, MacBook Air M4, WebGPU):

```
stage                          time
log1p + sum-to-1 normalize     0.1 ms
weighted sample + sentence     0.2 ms
gather + transformer (WebGPU)  110.7 ms
─────────────────────────────────────
total per cell                 ~111 ms   (seq_len=1071 valid of 2048 padded)
```

Apples-to-apples GPU comparison (transformer forward, same Apple M4 GPU, batch=1, FP32, `scripts/brain_baseline.py`):

```
shape                           PyTorch MPS   Browser WebGPU   browser speedup
seq_len=1072 (dynamic)          138.9 ms      110.7 ms         1.26×
seq_len=2048 (fixed pad)        295.1 ms      215   ms         1.37×
```

Browser is consistently faster than native PyTorch on the same GPU — ORT-Web's WebGPU shader kernels beat PyTorch's MPS kernels for this model at batch=1. Dynamic seq_len (Phase 7) wins for both backends.

Phase 6 vs Phase 7 on WebGPU (dynamic seq_len — skip padded tokens in attention):

```
config                          ms/cell
Phase 6 (fixed seq_len=2048)    215
Phase 7 (dynamic seq_len≈1071)  111   ← default, 1.9× faster
```

Backend comparison at seq_len=2048 (from `make brain-bench2`, pre-Phase-7):

```
config                          ms/cell
WebGPU FP32 batch=1             215   ← Phase 6 baseline
WebGPU FP32 batch=2             452   (O(L²) attention)
WebGPU FP32 batch=4             349
WebGPU INT8                     949
WASM SIMD 1 thread              1341
WASM SIMD 4 threads             1354  (no gain from threading)
WASM SIMD 10 threads            1394
```

### Key takeaways

1. **Full pipeline in the browser works.** Not just the transformer — log1p/normalize, weighted sampling with an in-JS RNG, chromosome ordering, and gather all run in TypeScript with cosine similarity against Python within the intrinsic RNG noise floor (per-cell cos 0.89–0.97 on the allen-cortex h5ad; Python-vs-Python at different seeds sits in the same range).
2. **Dynamic seq_len is the cheapest big win.** Real cells use ~52% of the padded 2048 tokens (mean 1071 valid). Since the exported ONNX graph already has dynamic axes, trimming src + mask to the valid prefix per cell cuts attention work ~3.65× and end-to-end time ~2× with zero accuracy cost. No re-export needed.
3. **WebGPU batch=1 FP32 is the right default.** Batching hurts per-cell (O(L²) attention), INT8 regresses (no GPU-native int8 kernels), and WASM threading is flat. Graph-optimization levels make no difference — the exported model is already fused.
4. **Browser is faster than native PyTorch on the same GPU.** 111 ms/cell WebGPU vs 139 ms/cell PyTorch MPS at the same shape (batch=1, seq_len=1072, FP32) — 1.26× on the Apple M4 GPU. 295 ms/cell MPS vs 215 ms/cell WebGPU at the pre-Phase-7 fixed 2048 shape — 1.37×. ORT-Web's WebGPU kernels beat PyTorch MPS for this workload. A 100-cell h5ad processes in ~11 s in-browser.
5. **Gather-upfront doesn't scale to 100 cells.** At seq_len=2048 × emb_dim=5120 × 100 cells × 4 bytes = 4.2 GB, which OOMs the tab. Moving gather inside the per-cell loop keeps the working set to ~22 MB per cell (and becomes the natural site for a future GPU-resident embedding table).
6. **First-visit cost is the real UX issue.** ~400 MB protein embedding table + 117 MB model download, then HTTP-cached. Runtime per-tab GPU peak ~1 GB.
7. **WASM threading requires cross-origin isolation.** Without COOP/COEP headers, `ort.env.wasm.numThreads` is silently ignored. The dev server in `scripts/brain_web_bench2.py` sends the right headers; a deployed app must too. Even with threading properly enabled it didn't help this model.
8. **UCE-brain's smaller architecture** (8 layers, d_model=512) is the right candidate for edge deployment — 3.5× smaller than the original UCE with equivalent design.

### Optimization headroom (remaining)

Ranked effort:payoff — see [plan.md](plan.md) for detail:

- ~~**Dynamic seq_len**~~ — **done (Phase 7, 1.9× speedup)**. Now at 111 ms/cell.
- **FP16 WebGPU weights**: transformer is memory-bandwidth-bound; halving weights roughly halves kernel time. Estimated **~60 ms/cell**.
- **GPU-resident embedding table + persistent session**: do the 5120-wide gather on GPU instead of shipping src through CPU; pool tensors across runs. Modest latency win, much lower memory churn.
- **`enableGraphCapture: true`** on WebGPU once shapes are stable. Would need to bucket-pad seq_len to a fixed grain first (valid length varies ±2 cell-to-cell on this dataset). 10–30% per ORT docs, untested here.

### Full 33-layer UCE feasibility

The original 33-layer UCE model is not a viable candidate for browser deployment:

- **Size**: The core transformer (excluding the 2.8 GB protein embedding table) is ~870M params = **3.3 GB FP32** as an ONNX file. This exceeds practical WebGPU memory budgets on most machines and is not a reasonable browser download, even cached. FP16 halves that to ~1.7 GB — borderline-feasible on 8 GB machines but still a painful first-visit download.
- **Architecture**: The original UCE uses `batch_first=False` (seq-first tensor layout), which produces ONNX graphs that fail on both CoreML and have not been validated on WebGPU. UCE-brain's `batch_first=True` layout produces a cleaner export that runs correctly.
- **Compute**: 33 layers at d_model=1280 is roughly **~26×** the compute of UCE-brain's 8 layers at d_model=512 — from the 8→33 layer ratio (4.1×) combined with (1280/512)² = 6.25× per-layer cost (attention + FFN both scale with d_model²). Projecting from our measured **111 ms/cell** WebGPU FP32 with the Phase 7 dynamic seq_len optimization: **~2.9 s/cell** FP32, or **~1.5 s/cell** with FP16 (Phase 8) once implemented. A 100-cell h5ad would take 2.5–5 minutes in-browser — usable for a "paste and wait" workflow, not interactive. This assumes the `batch_first=False` export can actually be made to run on WebGPU, which is itself unresolved.
- **Design intent**: The 33-layer model was designed for server-side GPU inference. UCE-brain was explicitly designed to be smaller while retaining the same architecture pattern, making it the right candidate for edge deployment.

### What's not yet covered

- Native h5ad parsing in the browser (the pipeline assumes the caller provides `(gene_symbols[], raw_counts[N,G])` in memory however they got there).
- Non-human species.
- The expression-prediction head (embedding extraction only).
- A polished demo UI — the phase*.html pages are validation harnesses, not product.
- The optimization phases listed above.
