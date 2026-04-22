# UCE-Brain integration plan for cytoverse

Add UCE-Brain as a second foundation model alongside SCimilarity. The app currently assumes a single foundation-model runtime in `src/worker.ts`; this plan generalizes that, proves out the new model, then explores IVFPQ compression tradeoffs at the new 512-d embedding width.

Reference for UCE-Brain browser feasibility: `uce-edge/README.md` and `uce-edge/plan.md`. The full pipeline runs at ~111 ms/cell on WebGPU (Apple M4), cos > 0.9 vs Python reference. Exported artifacts already exist in `uce-edge/web/`.

## Architectural choice

Keep a single `worker.ts` with an if/then on `foundationModel`. The h5ad reading, gene extraction, IVFPQ search, PUMAP projection, and feature-importance plumbing are generic. Only "turn cells into embeddings" differs between models — that becomes a dispatched inner function. Rewriting two workers would duplicate ~80% of the code.

Source of truth for per-model config: a new `public/models/<name>/metadata.json`.

## Stage 1 — Metadata-driven model dispatch (no behavior change)

Foundation that ships independently of whether UCE-Brain works.

1. Add `public/models/<name>/metadata.json` for each existing model with `{ foundationModel: "scimilarity", embeddingDim: 128, displayName: "..." }`. Keep `pumap/metadata.json` for backward compatibility; this new file is for *foundation-model-level* metadata.
2. App fetches `metadata.json` per model to populate the dropdown with `displayName`, and passes `foundationModel` to the worker via the `start` message (App already knows the model id; fetching metadata there is one extra roundtrip).
3. Worker: extract the current SCimilarity embedding logic into `embedBatchScimilarity(batch) -> Float32Array`. Dispatch on `foundationModel`. For now only the SCimilarity branch exists — behavior is identical.
4. Keep `models.txt` as the flat list of model ids — it's the cheapest directory listing and already works.

**Gate**: existing app runs identically with `scimilarity`, `scimilarity-subset`, `sspsygene`.

## Stage 2 — UCE-Brain ONNX export script

Mirror of `scripts/scimilarity_export_model.py` but exports different artifacts. Copy the export code from `uce-edge/scripts/brain_onnx_export.py` and the protein-embedding extraction from `uce-edge/scripts/extract_human_protein_embeddings.py`.

5. `scripts/uce_brain_export_model.py` produces:
   - `<output>/embedding/transformer.onnx` — the 8-layer brain transformer (FP32, ~117 MB)
   - `<output>/embedding/protein_embeddings.bin` — 19656×5120 FP32, ~400 MB
   - `<output>/embedding/gene_dict.json` — ~1 MB, gene_symbol → (protein_id, chromosome, genomic_position) with dense protein ids
   - `<output>/metadata.json` — `{ foundationModel: "uce-brain", embeddingDim: 512, ... }`
6. Validation: same pattern as SCimilarity export — run Python reference pipeline on a small h5ad, assert ONNX concordance cos > 0.999. uce-edge already proved this; this step just repackages the check into the cytoverse export script.

**Gate**: cos > 0.999 vs Python reference. Low risk.

## Stage 3 — `h5ad_to_embeddings.py` supports UCE-Brain

Run the Python source pipeline (not ONNX) — server-side GPU is faster and this run generates the reference that ONNX will be validated against in Stage 2.

7. Add `--foundation-model scimilarity|uce-brain` to `scripts/h5ad_to_embeddings.py`. UCE-brain branch uses `uce_brain.data.dataset` / `sampler` / `collator` + brain transformer. Output still `embeddings.npy` + `labels.parquet`, just 512-d.
8. **Initial run: 1k–10k cells of sspsygene** to prove end-to-end wiring. Feed through `ivfpq_train.py` (unchanged — `d` is already data-driven) → `public/models/sspsygene-ucebrain/ivfpq/`. Run `pumap_train.py` → `public/models/sspsygene-ucebrain/pumap/`. Copy UCE-brain embedding assets from Stage 2 into `public/models/sspsygene-ucebrain/embedding/`.
9. After Stage 4 works end-to-end, run the full sspsygene reference (on a GPU box or overnight) to replace the small one.

**Gate**: all reference artifacts loadable by IVFPQ/PUMAP with `d=512`. Verify with a Python search test before any browser work — if IVFPQ chokes at 512-d, we learn it here cheaply.

## Stage 4 — Worker UCE-Brain embedding path

New branch in the if/then. Lifted from `uce-edge/web/src/phase6.ts`.

10. New `embedBatchUCEBrain(batch)` function. On model load, fetch and IndexedDB-cache `protein_embeddings.bin` (400 MB one-time); fetch `gene_dict.json` and `transformer.onnx`. Per-cell: log1p + sum-to-1 normalize → weighted-sample 1024 genes → chromosome-order + CLS/PAD inserts → protein id gather → 5120-wide src gather → transformer → 512-d embedding. Processes one cell at a time (batching hurts at O(L²) attention, confirmed in uce-edge).
11. Unchanged in the worker: `getCellNames`, `getSampleGenes`, `getRawCounts`, `precomputeInflationIndices`, IVFPQ labeling, PUMAP projection, batch-update messaging, cancellation.
12. Feature importance: stub out for UCE-brain initially. SCimilarity's gene-zeroing approach doesn't cleanly port — zeroing a gene changes sampling and ordering, making each perturbation expensive. UI shows "feature importance not available for this model". Can be a separate spike.

**Gate**: SSPsyGene → UCE-Brain appears in dropdown; h5ad upload labels cells with cos > 0.9 vs a Python reference pipeline run on the same cells. Matches uce-edge Phase 6 bar.

## Stage 5 — IVFPQ compression exploration at 512-d

Once UCE-brain is functional end-to-end, explore the PQ compression/accuracy tradeoff space opened up by the larger embedding width.

Current SCimilarity: d=128, pq_m=16, pq_k=256 (one byte per 8-d sub-vector → 16 bytes/vector, 16× compression before IVF bucketing).

At d=512 there's more room to play. Candidates:
- **pq_m=16, k=256** — same bytes/vector as SCimilarity (32× compression vs 512×4 raw) — aggressive, lossy, minimal disk
- **pq_m=32, k=256** — 2× the bytes, 16× compression — middle ground, matches SCimilarity's *ratio*
- **pq_m=64, k=256** — 8× compression — conservative, likely near-raw accuracy
- **raw (no PQ)** — 2048 bytes/vector, baseline recall target

13. Matrix: train each PQ config against the same sspsygene-ucebrain reference, measure (a) disk footprint per cell, (b) recall@50 vs raw brute-force neighbors, (c) label agreement vs raw, on a held-out validation set of labeled cells.
14. Pick the knee of the curve for the default `sspsygene-ucebrain/ivfpq/`. Document the tradeoff in `public/models/sspsygene-ucebrain/metadata.json` for future reference.

**Gate**: default config picked; label agreement within ~2% of raw.

## Out of scope for the lab meeting demo

- Feature importance for UCE-brain
- Full sspsygene reference (use 1k–10k cells; scale up after demo)
- Non-human species
- FP16 / graph-capture / GPU-resident embedding table (uce-edge Phase 8–10 roadmap — separate follow-up)

## Risk callouts

- **Stage 1 ships value even if UCE-brain fails** — merge it independently.
- **Stage 2–3 risk is low** — uce-edge proved concordance.
- **Stage 4 is the unknown** — integrating the 400 MB protein embedding asset load into the worker's boot path alongside the existing IndexedDB/`ready_for_feature_importance` flow. If this stalls before the lab meeting, fall back to demoing UCE-brain in the uce-edge harness and ship just Stages 1–3 in cytoverse.
- **Stage 5 is pure optimization** — no blocker for the demo.

## Order of operations

1. Stage 1 metadata dispatch → merge
2. Stage 2 export script → validate
3. Stage 3 small sspsygene-ucebrain reference (1k–10k cells)
4. Stage 3 IVFPQ / PUMAP training on small reference
5. Stage 4 worker UCE-brain branch → end-to-end demo
6. (post-demo) Stage 3 full sspsygene reference
7. (post-demo) Stage 5 IVFPQ compression matrix

## Cleanup

Followups captured as we go — not blocking the demo, but worth closing out so
the reference-generation pipeline stays self-consistent with the new metadata
contract.

- **`scripts/h5ad_to_embeddings.py` should emit `metadata.json`**. Today the
  root `metadata.json` at `public/models/<name>/metadata.json` is hand-written.
  The ingest script already knows `foundationModel` (via the new
  `--foundation-model` flag from Stage 3) and `embeddingDim` (from the
  generated embeddings' shape), so it should write the metadata file as part
  of its output alongside `embeddings.npy` / `labels.parquet`. Then re-verify
  the Makefile targets that run this end-to-end —
  `scimilarity-subset-embeddings`, equivalents for `scimilarity` and
  `sspsygene`, and the new `sspsygene-ucebrain` target — still produce a
  complete, browseable model directory with no manual steps.
