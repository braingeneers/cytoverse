/**
 * UCE-brain baseline concordance harness.
 *
 * Purpose: lock in a reference set of cell embeddings from the current
 * (known-good, in-browser-validated) TS pipeline + ONNX transformer, so that
 * future optimizations — e.g., folding normalization into ONNX — can be
 * validated by re-running the same 10 cells and comparing cosine similarity
 * against the baseline.
 *
 * This is intentionally a before/after gate, NOT a JS↔Python gate. The TS
 * sampler's PRNG (Mulberry32) doesn't match Python's PCG64, so byte-exact
 * JS↔Python concordance isn't possible at the sampler level. What *is*
 * possible — and what matters for optimization safety — is "does changing
 * the ONNX graph still produce the same embedding for the same TS inputs."
 *
 * On first run: writes tests/fixtures/uce_brain_baseline.json (commit it).
 * On subsequent runs: compares against the committed baseline.
 *
 * Run: npx vitest run tests/unit/uce_brain_baseline.test.ts
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import h5wasm from 'h5wasm';
import * as ort from 'onnxruntime-node';
import {
  UCE_PAD_LENGTH,
  UCE_SAMPLE_SIZE,
  RNG,
  normalizeCounts,
  weightedSample,
  buildSentence,
  chromOrderFromSample,
  gather,
  buildGeneTable,
  validPrefixLength,
  type UCEGeneDict,
  type UCESentenceTokens,
} from '../../src/uceBrain';

const FIXTURE = resolve(__dirname, '../fixtures/GSE136831_subsample_10.h5ad');
const BASELINE = resolve(__dirname, '../fixtures/uce_brain_baseline.bin');
const MODEL_DIR = resolve(__dirname, '../../public/models/sspsygene-ucebrain/embedding');

// Tolerance for "the ONNX graph still produces the same embedding." ORT-CPU
// vs ORT-CPU with a graph rewrite should be bit-identical modulo kernel
// fusion roundoff — 1e-5 is comfortably above that floor.
const COSINE_TOLERANCE = 1e-5;

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/**
 * Minimal h5ad reader for the 10-cell fixture. Mirrors worker.ts but is
 * trimmed to what the fixture actually has.
 */
function readH5AD(path: string, Module: any) {
  const name = path.split('/').pop()!;
  const buf = readFileSync(path);
  Module.FS.writeFile(name, new Uint8Array(buf));
  const file = new h5wasm.File(name, 'r');

  // Genes — fixture has var/index as flat strings (old-style h5ad).
  const varGroup = file.get('var') as any;
  const varKeys = varGroup.keys();
  const candidates = ['feature_name', 'symbol', 'gene_symbols', 'gene_name', '_index', 'index'];
  const geneKey = candidates.find((k) => varKeys.includes(k));
  if (!geneKey) throw new Error(`no gene key found in var; keys: ${varKeys.join(',')}`);
  const genes = (varGroup.get(geneKey) as any).value.map((g: string) => g.toUpperCase());

  // Expression: fixture is dense 2D X.
  const X = file.get('X') as any;
  const isSparse = 'keys' in X && X.keys().includes('indptr');
  let readCell: (i: number) => Float32Array;
  if (isSparse) {
    const indptr = X.get('indptr');
    const indices = X.get('indices');
    const data = X.get('data');
    readCell = (i: number) => {
      const ptr = indptr.slice([[i, i + 2]]);
      const s = Number(ptr[0]), e = Number(ptr[1]);
      const v = data.slice([[s, e]]);
      const vi = indices.slice([[s, e]]);
      const row = new Float32Array(genes.length);
      for (let j = 0; j < vi.length; j++) row[Number(vi[j])] = Number(v[j]);
      return row;
    };
  } else {
    readCell = (i: number) => {
      const row = X.slice([[i, i + 1], [0, genes.length]]) as number[];
      return Float32Array.from(row.map(Number));
    };
  }

  const nCells = isSparse
    ? (X.get('indptr').shape[0] - 1)
    : X.shape[0];

  return { genes, nCells, readCell, close: () => file.close() };
}

// Baseline file layout: raw Float32Array of shape (nCells, embDim), row-major.
// Cell count and embedding dim are implied by the current run; if either
// changes the byte length mismatches and the test fails loudly.

describe('UCE-brain baseline concordance', () => {
  let Module: any;
  let session: ort.InferenceSession;
  let geneDict: UCEGeneDict;
  let proteinEmbeddings: Float32Array;
  let sentenceTokens: UCESentenceTokens;

  beforeAll(async () => {
    Module = await h5wasm.ready;

    geneDict = JSON.parse(readFileSync(`${MODEL_DIR}/gene_dict.json`, 'utf-8'));
    const pbuf = readFileSync(`${MODEL_DIR}/protein_embeddings.bin`);
    // Node Buffer → Float32Array view (zero-copy over the underlying ArrayBuffer).
    proteinEmbeddings = new Float32Array(pbuf.buffer, pbuf.byteOffset, pbuf.byteLength / 4);

    const chromosomeTokenMap: Record<number, number> = {};
    for (const [k, v] of Object.entries(geneDict.chromosome_token_map)) {
      chromosomeTokenMap[parseInt(k, 10)] = v;
    }
    sentenceTokens = { specials: geneDict.specials, chromosomeTokenMap };

    session = await ort.InferenceSession.create(`${MODEL_DIR}/transformer.onnx`);
  }, 120_000);

  it('produces embeddings matching the committed baseline', async () => {
    const { genes, nCells, readCell, close } = readH5AD(FIXTURE, Module);
    const geneTable = buildGeneTable(genes, geneDict);
    const G = genes.length;
    const nAligned = geneTable.symbols.length;
    expect(nAligned).toBeGreaterThan(100);

    const rawCell = new Float32Array(G);
    const alignedCounts = new Float32Array(nAligned);
    const embDim = geneDict.embedding_dim;
    let outDim = -1;
    let allEmbeddings: Float32Array | null = null;

    for (let cellIdx = 0; cellIdx < nCells; cellIdx++) {
      rawCell.set(readCell(cellIdx));
      for (let j = 0; j < nAligned; j++) {
        alignedCounts[j] = rawCell[geneTable.sampleIndices[j]];
      }

      const weights = normalizeCounts(alignedCounts);
      const rng = new RNG(cellIdx);
      const sampleIndices = weightedSample(weights, UCE_SAMPLE_SIZE, rng);
      const chromOrder = chromOrderFromSample(sampleIndices, geneTable, rng);
      const { tokenIds, attentionMask } = buildSentence(
        sampleIndices, chromOrder, geneTable, sentenceTokens, UCE_PAD_LENGTH
      );

      const validLen = validPrefixLength(attentionMask);
      const idsSlice = tokenIds.subarray(0, validLen);
      const maskSlice = attentionMask.slice(0, validLen);
      const src = gather(proteinEmbeddings, geneDict.num_rows, embDim, idsSlice, 1, validLen);

      const srcTensor = new ort.Tensor('float32', src, [1, validLen, embDim]);
      const maskTensor = new ort.Tensor('float32', maskSlice, [1, validLen]);
      const out = await session.run({ src: srcTensor, mask: maskTensor });
      const cellEmb = out.cell_embedding.data as Float32Array;

      if (outDim === -1) {
        outDim = cellEmb.length;
        allEmbeddings = new Float32Array(nCells * outDim);
      }
      allEmbeddings!.set(cellEmb, cellIdx * outDim);
    }

    close();

    if (!existsSync(BASELINE)) {
      writeFileSync(BASELINE, Buffer.from(allEmbeddings!.buffer));
      console.log(`✅ Wrote baseline: ${BASELINE} (${nCells} × ${outDim})`);
      console.log('   Commit this file. Subsequent runs will compare against it.');
      return;
    }

    const baselineBuf = readFileSync(BASELINE);
    const baseline = new Float32Array(baselineBuf.buffer, baselineBuf.byteOffset, baselineBuf.byteLength / 4);
    expect(baseline.length).toBe(allEmbeddings!.length);

    const cosines: number[] = [];
    for (let i = 0; i < nCells; i++) {
      const a = baseline.subarray(i * outDim, (i + 1) * outDim);
      const b = allEmbeddings!.subarray(i * outDim, (i + 1) * outDim);
      const cos = cosine(a, b);
      cosines.push(cos);
    }

    const minCos = Math.min(...cosines);
    const meanCos = cosines.reduce((a, b) => a + b, 0) / cosines.length;
    console.log(`Cosines: min=${minCos.toFixed(8)} mean=${meanCos.toFixed(8)}`);
    console.log('Per-cell:', cosines.map((c) => c.toFixed(6)).join(' '));

    expect(1 - minCos).toBeLessThan(COSINE_TOLERANCE);
  }, 300_000);
});
