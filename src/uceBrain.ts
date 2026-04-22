/**
 * UCE-Brain browser-side pipeline helpers.
 *
 * Ported from uce-edge/web/src/lib/*.ts. The transformer runs in ONNX; these
 * are the JS stages that wrap it:
 *
 *   raw counts
 *     → log1p + sum-to-1 normalize (normalizeCounts)
 *     → weighted sampling with replacement (weightedSample)
 *     → chromosome-shuffled sentence with CLS/CHROM/CHROM_END/PAD (buildSentence)
 *     → protein-embedding gather (gather)
 *     → ONNX transformer (caller-side)
 *
 * All functions are pure and have no DOM dependencies so the worker can use
 * them directly.
 */

// Sampler constants — must match
// uce-edge/UCE-brain/src/uce_brain/data/sampler.py
export const UCE_PAD_LENGTH = 2048;
export const UCE_SAMPLE_SIZE = 1024;

export interface UCEGeneDict {
  num_rows: number;
  embedding_dim: number;
  specials: {
    pad_token_idx: number;
    cls_token_idx: number;
    chrom_token_right_idx: number;
  };
  chromosome_token_map: Record<string, number>;
  genes: Record<string, {
    protein_embedding_id: number;
    chromosome_id: number;
    location: number;
  }>;
}

export interface UCEGeneTable {
  /** new-space protein id for each aligned gene (length = numAlignedGenes). */
  proteinIdsNew: Int32Array;
  /** chromosome id for each aligned gene. */
  chroms: Int32Array;
  /** genomic start for each aligned gene. */
  starts: Int32Array;
  /** Indices of gene_dict-matched genes in the *sample* gene list. */
  sampleIndices: Int32Array;
  /** Aligned gene symbols, in table order. */
  symbols: string[];
}

export interface UCESentenceTokens {
  specials: UCEGeneDict['specials'];
  chromosomeTokenMap: Record<number, number>;
}

/**
 * Per-cell log1p + sum-to-1 normalize. Matches the Python reference:
 *   log_expr = torch.log1p(counts)
 *   sum = clamp(log_expr.sum(dim=1, keepdim=True), min=1e-8)
 *   weights = log_expr / sum
 * Deterministic; JS and Python match to Float32 roundoff.
 */
export function normalizeCounts(counts: Float32Array): Float32Array {
  const G = counts.length;
  const weights = new Float32Array(G);
  let sum = 0;
  for (let i = 0; i < G; i++) {
    const v = Math.log1p(counts[i]);
    weights[i] = v;
    sum += v;
  }
  const denom = Math.max(sum, 1e-8);
  for (let i = 0; i < G; i++) weights[i] = weights[i] / denom;
  return weights;
}

/** Mulberry32 — seedable, non-cryptographic. */
export class RNG {
  private state: number;
  constructor(seed: number) {
    this.state = (seed | 0) || 0x9e3779b9;
  }
  next(): number {
    let t = (this.state = (this.state + 0x6d2b79f5) | 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  nextInt(n: number): number {
    return Math.floor(this.next() * n);
  }
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.nextInt(i + 1);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }
}

/**
 * Weighted sampling with replacement — JS equivalent of
 * np.random.Generator.choice(p=weights, replace=True). Cumulative sum +
 * binary search: O(G) build, O(log G) per draw.
 */
export function weightedSample(
  weights: Float32Array,
  sampleSize: number,
  rng: RNG
): Int32Array {
  // Float64 prefix sum to avoid precision loss on ~18k terms.
  const cum = new Float64Array(weights.length);
  let acc = 0;
  for (let i = 0; i < weights.length; i++) {
    acc += weights[i];
    cum[i] = acc;
  }
  const total = cum[cum.length - 1];
  if (!(total > 0)) {
    throw new Error(`weights sum to non-positive total: ${total}`);
  }
  const out = new Int32Array(sampleSize);
  for (let i = 0; i < sampleSize; i++) {
    const u = rng.next() * total;
    // Binary search: smallest i such that cum[i] >= target.
    let lo = 0;
    let hi = cum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cum[mid] < u) lo = mid + 1;
      else hi = mid;
    }
    out[i] = lo;
  }
  return out;
}

/**
 * Build a UCE-brain cell sentence from sampled gene indices.
 * Returns tokenIds (padLength int32) + attentionMask (padLength float32, 1=valid, 0=pad).
 */
export function buildSentence(
  sampleIndices: Int32Array,
  chromOrder: Int32Array,
  table: UCEGeneTable,
  tokens: UCESentenceTokens,
  padLength: number
): { tokenIds: Int32Array; attentionMask: Float32Array } {
  const tokenIds = new Int32Array(padLength);
  const attentionMask = new Float32Array(padLength);

  tokenIds[0] = tokens.specials.cls_token_idx;
  attentionMask[0] = 1;
  let cursor = 1;

  const genesByChrom = new Map<number, number[]>();
  for (let i = 0; i < sampleIndices.length; i++) {
    const geneIdx = sampleIndices[i];
    const chrom = table.chroms[geneIdx];
    let list = genesByChrom.get(chrom);
    if (list === undefined) {
      list = [];
      genesByChrom.set(chrom, list);
    }
    list.push(geneIdx);
  }

  for (let k = 0; k < chromOrder.length; k++) {
    const chrom = chromOrder[k];
    if (chrom < 0) break;

    const chromToken = tokens.chromosomeTokenMap[chrom];
    if (chromToken === undefined) {
      throw new Error(`no token mapping for chromosome ${chrom}`);
    }
    tokenIds[cursor] = chromToken;
    attentionMask[cursor] = 1;
    cursor++;

    const geneIdxs = genesByChrom.get(chrom);
    if (geneIdxs === undefined || geneIdxs.length === 0) {
      throw new Error(`chromosome ${chrom} in order but no genes sampled`);
    }
    geneIdxs.sort((a, b) => {
      const sa = table.starts[a];
      const sb = table.starts[b];
      if (sa !== sb) return sa - sb;
      return a - b;
    });

    for (const geneIdx of geneIdxs) {
      tokenIds[cursor] = table.proteinIdsNew[geneIdx];
      attentionMask[cursor] = 1;
      cursor++;
    }

    tokenIds[cursor] = tokens.specials.chrom_token_right_idx;
    attentionMask[cursor] = 1;
    cursor++;
  }

  const pad = tokens.specials.pad_token_idx;
  for (let i = cursor; i < padLength; i++) {
    tokenIds[i] = pad;
    attentionMask[i] = 0;
  }

  return { tokenIds, attentionMask };
}

/**
 * Gather (batch, seq_len, embDim) from a (numRows, embDim) table by token IDs.
 * Pure memcpy — matches Python reference element-wise.
 */
export function gather(
  table: Float32Array,
  tableRows: number,
  embDim: number,
  tokenIds: Int32Array,
  batchSize: number,
  seqLen: number
): Float32Array {
  if (tokenIds.length !== batchSize * seqLen) {
    throw new Error(
      `tokenIds length ${tokenIds.length} != ${batchSize} * ${seqLen}`
    );
  }
  const out = new Float32Array(batchSize * seqLen * embDim);
  for (let i = 0; i < tokenIds.length; i++) {
    const id = tokenIds[i];
    if (id < 0 || id >= tableRows) {
      throw new Error(
        `token id ${id} at pos ${i} out of range [0, ${tableRows})`
      );
    }
    const srcOffset = id * embDim;
    const dstOffset = i * embDim;
    out.set(table.subarray(srcOffset, srcOffset + embDim), dstOffset);
  }
  return out;
}

/**
 * Build the per-sample GeneTable by intersecting the h5ad sample's genes with
 * the model's gene_dict. Gene symbol comparison is case-insensitive (sample
 * genes are already uppercased by worker.getSampleGenes).
 */
export function buildGeneTable(
  sampleGenes: string[],
  geneDict: UCEGeneDict
): UCEGeneTable {
  // Pre-build a case-insensitive lookup from the gene_dict keys.
  const dictByUpper = new Map<string, {
    protein_embedding_id: number;
    chromosome_id: number;
    location: number;
  }>();
  for (const [sym, v] of Object.entries(geneDict.genes)) {
    dictByUpper.set(sym.toUpperCase(), v);
  }

  const proteinIds: number[] = [];
  const chroms: number[] = [];
  const starts: number[] = [];
  const sampleIdx: number[] = [];
  const symbols: string[] = [];

  for (let i = 0; i < sampleGenes.length; i++) {
    const entry = dictByUpper.get(sampleGenes[i].toUpperCase());
    if (entry !== undefined) {
      proteinIds.push(entry.protein_embedding_id);
      chroms.push(entry.chromosome_id);
      starts.push(entry.location);
      sampleIdx.push(i);
      symbols.push(sampleGenes[i]);
    }
  }

  return {
    proteinIdsNew: Int32Array.from(proteinIds),
    chroms: Int32Array.from(chroms),
    starts: Int32Array.from(starts),
    sampleIndices: Int32Array.from(sampleIdx),
    symbols,
  };
}

/**
 * Compute the shuffled chromosome order for a sampled cell.
 * Matches the Python sampler:
 *   uq = np.unique(chrom_ids_for_sampled_genes)  # sorted ascending
 *   rng.shuffle(uq)
 */
export function chromOrderFromSample(
  sampleIndices: Int32Array,
  table: UCEGeneTable,
  rng: RNG
): Int32Array {
  const seen = new Set<number>();
  for (let i = 0; i < sampleIndices.length; i++) {
    seen.add(table.chroms[sampleIndices[i]]);
  }
  const arr = Array.from(seen).sort((a, b) => a - b);
  rng.shuffle(arr);
  return Int32Array.from(arr);
}

/**
 * Durable byte-level fetch through the browser Cache API. Used for the
 * ~400 MB protein_embeddings.bin so a reload doesn't re-download it.
 */
export async function cachedFetchBytes(
  url: string,
  cacheName: string = 'cytoverse-ucebrain-v1'
): Promise<{ buffer: ArrayBuffer; fromCache: boolean; ms: number }> {
  const t0 = performance.now();

  let cache: Cache | null = null;
  try {
    cache = await caches.open(cacheName);
  } catch {
    cache = null;
  }

  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buffer = await hit.arrayBuffer();
      return { buffer, fromCache: true, ms: performance.now() - t0 };
    }
  }

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);

  if (cache) {
    try {
      await cache.put(url, resp.clone());
    } catch (e) {
      console.warn(`cache.put failed for ${url}: ${(e as Error).message}`);
    }
  }

  const buffer = await resp.arrayBuffer();
  return { buffer, fromCache: false, ms: performance.now() - t0 };
}

/**
 * Sum an attention mask to find valid prefix length. Padding is always at the
 * tail, so trimming to this length feeds dynamic seq_len to the transformer.
 */
export function validPrefixLength(mask: Float32Array): number {
  let v = 0;
  for (let i = 0; i < mask.length; i++) v += mask[i];
  return v;
}
