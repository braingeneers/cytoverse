# UCE-brain browser deployment — phased plan

## Status: Phases 0–7 complete (2026-04-18)

All six preprocessing phases plus the dynamic seq_len optimization are implemented and passing on WebGPU. The full UCE-brain pipeline (stages 1–7 + transformer) now runs end-to-end in the browser from raw counts. End-to-end performance: **~111 ms/cell on WebGPU** (measured over 100 cells; was 215 ms/cell before Phase 7). Apples-to-apples against native PyTorch MPS on the same M4 GPU at the same shape: **139 ms/cell** — so browser is **1.26× faster than native PyTorch on the same Apple GPU**. See "Performance & optimization headroom" at the bottom for remaining optimizations.

## Goal

Run the full UCE-brain inference pipeline (not just the transformer) in a web browser, matching the Python reference output. The transformer already runs at 14ms FP32 on WebGPU with bit-perfect fidelity. This plan adds the preprocessing stages incrementally.

## Architecture

**Split**: JavaScript for preprocessing + data munging (stages 1–6), ONNX Runtime Web for the heavy numerical path (transformer). This matches UCE-brain's design where preprocessing is Python/control-flow and only the transformer is pure tensor compute.

**Assets shipped to browser**:
- `brain_8l_fp32.onnx` — 117 MB, already built, cacheable
- `human_protein_embeddings.bin` — ~400 MB FP32, 19656×5120, cached in IndexedDB on first load
- `human_gene_dict.json` — ~1 MB, gene symbol → (protein_id, chromosome, genomic_position)

**Hardcoded UCE-brain constants** (from `UCE-brain/src/uce_brain/data/sampler.py`):
- `cls_token_idx = 1`
- `chrom_token_offset = 1000` (chromosome boundary tokens start here)
- `chrom_token_right_idx = 2000` (CHROM_END marker)
- `pad_token_idx = 0`
- `sample_size = 1024` (genes sampled per cell)
- `pad_length = 2048` (final sequence length)
- `positive_sample_num = 100`, `negative_sample_num = 100` (only matter for expression prediction, not embedding extraction)
- `mask_prop = 0.0` (no masking during inference)

## Input

Test h5ad: `data/allen-celltypes+human-cortex+m1-100.h5ad` (100 cells, human cortex).

## Pipeline stages (from `UCE-brain/src/uce_brain/data/`)

```
h5ad raw counts
  │
  ├─ Stage 1: gene symbol → protein_id lookup         [dict munging, JS]
  │
  ├─ Stage 2: log1p + per-cell sum-to-1 normalize     [numeric, JS or ONNX]
  │
  ├─ Stage 3: weighted random sampling of 1024 genes  [RNG, JS]
  │
  ├─ Stage 4: chromosome ordering + CLS/PAD inserts   [integer munging, JS]
  │
  ├─ Stage 5: token indices → protein IDs             [gather, JS]
  │
  ├─ Stage 6: attention mask                          [trivial, JS]
  │
  ├─ Stage 7: protein embedding gather → (B, L, 5120) [gather, JS, asset-dependent]
  │
  └─ Transformer                                       [ONNX WebGPU, done]
```

Stages 2 and 7 could live inside the ONNX graph but we keep them in JS to keep the ONNX model unchanged from what we've already validated.

---

## Phase 0 — Extract human protein embedding table [DONE]

**Deliverables**:
- `scripts/extract_human_protein_embeddings.py`
- `web/human_protein_embeddings.bin` (FP32, shape 19656×5120, ~400 MB — gitignored)
- `web/human_gene_dict.json` (renumbered, ~1 MB)

**What**: Load the UCE-brain checkpoint, extract the 19,656 rows of the embedding table corresponding to genes in `UCE-brain/gene_data/human_gene_dict.json`, renumber protein IDs to a dense 0..19655 range, write two assets.

**Key detail**: the original gene dict has protein IDs that index into the full 145,469-row table. After slicing, we need a new dict with `new_protein_id = dense_index_in_sliced_table` so the browser can index into the 19,656-row buffer directly.

**Validation**:
- Round-trip test: pick 5 known gene symbols, look up original protein_id → vector in full table, look up new protein_id → vector in sliced table, assert element-wise equality.
- Shape check: sliced table is exactly (19656, 5120).
- Gene count check: renumbered dict has 19656 entries.

---

## Phase 1 — Python reference pipeline on real h5ad [DONE]

**Deliverables**:
- `scripts/brain_reference_pipeline.py`
- `data/brain_reference/` directory with per-stage fixture files (gitignored)

**What**: Run UCE-brain's actual pipeline (using `uce_brain.data.dataset`, `sampler`, `collator`) end-to-end on `data/allen-celltypes+human-cortex+m1-100.h5ad`. At each stage boundary, save the intermediate tensors.

**Fixtures to save** (one file per stage per batch):
- `ref_gene_symbols.json` — input gene symbols aligned with counts (Stage 1 output)
- `ref_normalized_weights.bin` — (N, G) float32, Stage 2 output
- `ref_sampled_gene_indices.bin` — (N, sample_size) int32, Stage 3 output
- `ref_ordered_token_indices.bin` — (N, pad_length) int32, Stage 4 output (with CLS/CHROM/PAD tokens)
- `ref_protein_ids.bin` — (N, pad_length) int32, Stage 5 output (mapped to protein IDs)
- `ref_attention_mask.bin` — (N, pad_length) float32, Stage 6 output
- `ref_src_embeddings.bin` — (N, pad_length, 5120) float32, Stage 7 output
- `ref_cell_embedding.bin` — (N, 512) float32, final transformer output

N = number of cells to test (start with 4 for manageable fixture sizes; src_embeddings at seq_len=2048 is ~40 MB per cell).

**Determinism**: use a fixed RNG seed for Stage 3 so we can reproduce.

**Validation**:
- End-to-end: run the UCE-brain notebook path on the same h5ad, confirm final cell embeddings match.
- Sanity: each fixture shape matches expected dimensions; no NaNs; cell embeddings are L2-normalized.

---

## Phase 2 — Browser: transformer only (from pre-embedded input) [DONE]

**Deliverables**:
- `web/phase2.html`
- `scripts/brain_web_phase2.py` (Playwright driver, optional)

**JS does**: load `ref_src_embeddings.bin` and `ref_attention_mask.bin`, run through `brain_8l_fp32.onnx` on WebGPU.

**Python did**: stages 1–7.

**Validation**: cosine(browser cell_embedding, `ref_cell_embedding.bin`) per cell. Expect cos > 0.999 (we've already seen 1.000000 on synthetic input).

**What this proves**: realistic activation distributions (not just synthetic noise) don't break the WebGPU path. Dependency: none beyond Phase 1.

---

## Phase 3 — Browser: protein embedding gather + transformer [DONE]

**Deliverables**:
- `web/phase3.html` — adds IndexedDB caching of the 400 MB embedding table
- JS gather function

**JS does**: 
1. Fetch + cache `human_protein_embeddings.bin` via IndexedDB (first load only)
2. Load `ref_protein_ids.bin` and `ref_attention_mask.bin`
3. Gather: for each (B, L) position, copy the 5120-float slice from embedding buffer
4. Feed into transformer

**Python did**: stages 1–5.

**Validation**:
- Reconstructed (B, L, 5120) array matches `ref_src_embeddings.bin` element-wise (0 tolerance — it's just an array copy).
- Cell embedding matches `ref_cell_embedding.bin` (cos > 0.999).

**First real test of IndexedDB flow.** Budget: initial load should be ≤ 30s on typical connection, subsequent loads ≤ 1s.

---

## Phase 4 — Browser: chromosome ordering + CLS/PAD + gather + transformer [DONE]

**Deliverables**:
- `web/phase4.html`
- JS: chromosome-aware ordering and token insertion

**JS does**: consume a pre-sampled list of gene references, sort within chromosomes by genomic position, insert CLS + chromosome boundary tokens + pad to 2048, map to protein IDs via the sliced `human_gene_dict.json`, gather, run transformer.

**Input from Python**: per cell, a list of `(gene_symbol, chromosome, genomic_position)` tuples representing the sampled genes (Stage 3 output, but as symbols for JS friendliness).

**Python did**: stages 1–3 (the sampling).

**Validation**:
- Ordered token indices match `ref_ordered_token_indices.bin` element-wise.
- Protein IDs match `ref_protein_ids.bin` element-wise.
- Cell embedding matches `ref_cell_embedding.bin` (cos > 0.999).

**Code**: straightforward integer array manipulation with the hardcoded constants above. No randomness yet.

---

## Phase 5 — Browser: weighted sampling + rest [DONE]

**Deliverables**:
- `web/phase5.html`
- JS: weighted sampling with replacement (standard algorithm — cumulative sum + binary search, or Vose alias method)

**JS does**: given (N, G) normalized weights + gene metadata table, weighted-sample 1024 genes per cell, then run Phase 4 logic.

**Python did**: stages 1–2 (normalized weights).

**Validation** — *this is the first phase where bit-identity fails*:
- Cannot expect element-wise match since JS RNG differs from numpy's `Generator`.
- Instead: for a batch of cells, measure cos(JS cell_embedding, Python cell_embedding). Targets are set against the **intrinsic seed-noise floor** — the cos you get running Python twice with different seeds. Empirically on this h5ad that floor is ~0.89–0.97 per cell (sparse cells are more gene-selection sensitive). So: mean cos > 0.90, individual cells > 0.88, and the JS results should not be materially worse than Python-vs-Python at the same seeds.
- Sanity: sampling distribution check. For a cell, run JS sampler 1000 times, compute histogram of sampled genes, compare shape to expected (`weights * trials * sample_size`). Pearson > 0.95 — this is the one that actually proves the sampler is correct. If histogram passes and per-cell cos is still low, the model is just gene-selection-sensitive (expected).

**Risk**: if histogram Pearson < 0.95, the sampler distribution is broken. If histogram passes but per-cell cos is dramatically worse than Python-vs-Python noise floor, something else (e.g. chrom-shuffle ordering) differs. Having both checks separates these cases cleanly.

---

## Phase 6 — Browser: full pipeline from raw counts [DONE]

**Deliverables**:
- `web/phase6.html` / `web/index.html` (this becomes the real demo)
- JS: log1p + per-row normalization

**JS does**: ingest `(gene_symbols, raw_counts)` per cell, apply `log1p` via numpy-equivalent (`Math.log1p` is fine), sum-to-1 normalize per cell, then run Phase 5.

**Python did**: nothing — just provides the raw h5ad-derived counts and the reference cell embedding.

**Validation**: same as Phase 5 (cos similarity, not element-wise). Normalization is deterministic, so the only source of drift remains the sampler from Phase 5.

---

## Phase 7 — Dynamic seq_len (skip padded tokens) [DONE]

**Deliverables**:
- `web/src/phase6.ts` updated to slice `src` + `mask` to the valid prefix length per cell before calling `runBrain`.
- `scripts/brain_reference_pipeline.py` scaled to 100 cells with a `--skip-src-embeddings` flag (the per-cell 40 MB fixture wasn't needed after Phase 3), and valid-token stats added to `manifest.json`.
- Per-cell `seq_len_used` recorded in `window.phase6Result`; Playwright driver logs min/max/mean.

**What**: Padding in UCE-brain is always at the sentence tail, so the valid prefix length is `sum(attention_mask)`. The exported ONNX graph already had dynamic `seq_len` axes (from `brain_onnx_export.py`), so nothing needed re-exporting — purely a JS-side change to slice inputs per cell.

**Gotcha discovered during implementation**: at `n_cells=100` the old "gather all cells upfront" path allocated `100 × 2048 × 5120 × 4 = 4.2 GB` for `src` and OOM'd the Chromium tab. Fixed by moving the gather *inside* the per-cell loop (one `validLen × 5120 × 4 ≈ 22 MB` buffer per cell instead).

**Measured** (100 cells, `allen-celltypes+human-cortex+m1-100.h5ad`, WebGPU, M4):

| metric | value |
|---|---|
| seq_len used | 1071 mean (1071–1073), pad=2048 → 52.3% utilization |
| mean transformer time/cell | **110.7 ms** (was 215 ms → 1.9× speedup) |
| min cosine vs Python | 0.891 |
| mean cosine vs Python | 0.953 |
| histogram pearson (sampler check) | 0.9975 |
| weights max\|diff\| (normalize check) | 2.3e-10 |

Speedup came in under the theoretical 3.65× attention-only ratio because MLPs and per-call overhead are O(L), not O(L²), and 100 cells is enough cells that per-call fixed costs (tensor setup, kernel dispatch) show up.

---

## Out of scope for this plan

- Loading h5ad directly in the browser. Assume the caller provides `(gene_symbols[], counts[N,G])` in memory, however it got there. (The existing project has h5ad reading solved.)
- Non-human species.
- Model fine-tuning or the expression prediction head.
- The 33-layer UCE (ruled out separately — see README).
- Batch size > 1 optimization. Browser demo does one cell at a time; batching can come later if throughput matters.

## File layout after all phases

```
scripts/
  extract_human_protein_embeddings.py   # Phase 0
  brain_reference_pipeline.py            # Phase 1
  brain_web_phase2.py ... phase6.py      # Playwright drivers (optional)

web/
  index.html                             # final demo (Phase 6)
  phase2.html ... phase5.html            # intermediate demos
  brain_8l_fp32.onnx                     # existing, gitignored
  human_protein_embeddings.bin           # Phase 0 output, gitignored
  human_gene_dict.json                   # Phase 0 output, renumbered
  (ref_*.bin loaded via fetch from data/brain_reference/)

data/
  allen-celltypes+human-cortex+m1-100.h5ad  # input
  brain_reference/                           # Phase 1 fixtures, gitignored
    ref_*.bin
    ref_*.json
```

## Validation philosophy

Each phase replaces one Python stage with a JS stage. The boundary between "Python did this" and "JS did this" shifts earlier each phase. The validation at each phase compares the JS-computed output against the corresponding Python fixture from Phase 1.

Until Phase 5, all comparisons are element-wise (no randomness in JS yet). Phase 5 is where we lose bit-identity due to RNG differences; from there we switch to cosine similarity on the final cell embedding plus distributional checks on the intermediate sampler.

---

## Performance & optimization headroom (post-Phase-7)

### Current numbers (measured, Apple Silicon M4, headless Chromium, 100 cells)

End-to-end, from raw counts to cell embedding:
- **111 ms/cell** on WebGPU (batch=1, FP32, dynamic seq_len ≈ 1071)
  - normalize (log1p + sum-to-1): 0.1 ms
  - sample + chrom-shuffle + sentence build: 0.2 ms
  - gather + transformer: 110.7 ms
- Pre-Phase-7 baseline (fixed pad_length=2048): 215 ms/cell → Phase 7 delivered **1.9× speedup**.

### Apples-to-apples: browser WebGPU vs native PyTorch MPS

Same Apple M4 GPU, transformer forward only, batch=1, FP32 (`scripts/brain_baseline.py`):

| shape | PyTorch MPS | Browser WebGPU | browser speedup |
|---|---|---|---|
| seq_len=1072 (Phase 7 dynamic) | 138.9 ms | 110.7 ms | **1.26×** |
| seq_len=2048 (fixed pad) | 295.1 ms | 215 ms | **1.37×** |

The browser is faster than native PyTorch on the same GPU. Why: ORT-Web's WebGPU kernels are more aggressively fused for transformer inference at batch=1 than PyTorch MPS kernels, which are tuned more for training-shaped workloads. This was the measurement we'd been missing — the earlier "browser is 3× faster than PyTorch" number was vs CPU FP32, which isn't the right baseline if the user has a GPU.

First-visit cost: ~400 MB model + embedding table download (then HTTP-cached). Per-tab GPU memory peak: ~1 GB.

### What we measured and ruled out

Backend bench harness: `web/src/bench.ts` + `scripts/brain_web_bench2.py`. All results below are per-cell, seq_len=2048, FP32 unless noted.

| Config | ms/cell | Notes |
|---|---|---|
| **WebGPU batch=1** | **215** | baseline |
| WebGPU batch=2 | 452 | O(L²) attention, batching hurts |
| WebGPU batch=4 | 349 | still hurts per cell |
| WebGPU int8 | 949 | quant model not GPU-optimized |
| WebGPU graph_opt=all | 215 | no gain (model already fused at export) |
| WASM simd 1 thread | 1341 | baseline CPU path |
| WASM simd 4 threads | 1354 | threading does not help |
| WASM simd 10 threads | 1394 | saturated / synchronization overhead |
| WASM int8 4 threads | 2967 | worse than fp32 |

**Gotcha**: WASM threading requires cross-origin isolation (COOP/COEP headers → SharedArrayBuffer). Without those, ORT silently falls back to single-thread regardless of `numThreads` setting. The dev server in `scripts/brain_web_bench2.py` sends these headers; a deployed app must too.

Conclusion: WebGPU batch=1 FP32 is the right default. Threading, batching, int8, and graph-opt levels do not move the needle at this model shape.

### Optimization roadmap (remaining phases, ranked effort:payoff)

**Phase 8 — FP16 WebGPU weights.** Transformer is memory-bandwidth-bound on GPU; halving weight bytes should roughly halve kernel time. ORT-Web has a mature FP16 WebGPU EP for transformers. **Estimated: ~60 ms/cell (on top of Phase 7).** Risk: accuracy drift — needs a calibration run and a per-cell cos check against FP32. Needs: re-export with FP16 conversion (`onnxconverter-common`) + session option `executionProviders: [{ name: 'webgpu', preferredLayout: 'NHWC' }]` or equivalent.

**Phase 9 — Persistent session + GPU-resident embedding table.** Two things: (a) allocate the 400 MB `human_protein_embeddings.bin` as a WebGPU buffer once, do the gather on GPU instead of shipping src through the CPU each call; (b) pool the src/mask tensors across runs to avoid per-call allocation. Already partially motivated by the Phase-7 gather-inside-loop refactor (the per-cell gather copy is now ~22 MB instead of 4.2 GB upfront, but still a CPU→GPU ship per call). **Estimated: 15–30 ms/cell savings + much lower CPU memory churn.** Risk: modest, just plumbing — `ort.Tensor.fromGpuBuffer` + a tiny gather shader. Needs: gather compute shader in WebGPU, session option `preferredOutputLocation: 'gpu-buffer'` for intermediate outputs, explicit `.getData()` for the final cell embedding.

**Phase 10 — Graph capture.** `sessionOpts: { enableGraphCapture: true }` reuses compiled shader bundles between runs when shapes are stable. ORT docs claim 10–30%. Valid seq_len varies by ±2 across cells (1071–1073 in the test set), so we'd need to pad to a small fixed bucket (e.g. round up to nearest 16) before this is useful. **Estimated: 10–30 ms/cell.** Risk: low, it's a flag + small padding logic.

**UX, not perf — service-worker cache the 400 MB bundle.** First-visit is the real UX cost. Worth doing once we have a real demo page.

### Not pursuing

- WASM threading: bench shows no gain and may hurt.
- Batching at the UCE-brain model's current shape: O(L²) attention makes per-cell cost worse.
- int8 quantization as it exists today: the shipped int8 model isn't GPU-optimized and hurts on both backends. Would need recalibration + static int8 WebGPU EP (not yet mature in ORT-Web).
- Native (non-web) deployment: out of scope; the point is browser inference.
