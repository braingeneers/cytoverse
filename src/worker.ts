/**
 * Unified Web Worker for embedding and labeling cells
 *
 * This worker processes h5ad files, generates embeddings, maps to UMAP coordinates,
 * and labels cells using IVFPQ search in a streamlined pipeline.
 *
 * Data Flow:
 * h5ad file → embeddings → PUMAP coordinates → IVFPQ labels → App.vue
 */

import h5wasm from 'h5wasm';
import { InferenceSession, Tensor, env } from 'onnxruntime-web';
import { IVFPQ, SearchResults } from './ivfpq';
import { UserIVFPQ } from './userIvfpq';
import {
  UCE_PAD_LENGTH,
  UCE_SAMPLE_SIZE,
  UCEGeneDict,
  UCESentenceTokens,
  RNG,
  normalizeCounts,
  weightedSample,
  buildSentence,
  chromOrderFromSample,
  gather,
  buildGeneTable,
  cachedFetchBytes,
  validPrefixLength,
} from './uceBrain';

// ORT log level defaults to 'warning'. Bump to 'verbose' only when debugging
// a specific ONNX issue — every kernel logs per call otherwise.

// Configuration
const NUM_NEAREST_NEIGHBORS = 50;
const NUM_PARTITIONS_TO_SEARCH = 4;
const BATCH_SIZE = 32;

// TypeScript interfaces
type FoundationModel = 'scimilarity' | 'uce-brain';

interface ModelMetadata {
  foundationModel: FoundationModel;
  embeddingDim: number;
  displayName?: string;
}

interface UCEBrainModel {
  proteinEmbeddings: Float32Array;
  tableRows: number;
  embDim: number;
  geneDict: UCEGeneDict;
  sentenceTokens: UCESentenceTokens;
}

interface ModelInfo {
  modelID: string;
  foundationModel: FoundationModel;
  /** SCimilarity: list of model-input genes. UCE-brain: empty (alignment is per-sample via gene_dict). */
  genes: string[];
  embeddingSession: InferenceSession;
  mappingSession: InferenceSession;
  ivfpq: IVFPQ | UserIVFPQ;
  /** UCE-brain-only artifacts. Null for SCimilarity. */
  uceBrain: UCEBrainModel | null;
}

interface StartMessage {
  type: 'start';
  modelID: string;
  modelsURL: string;
  useUserIndex: boolean;
  userIndexId?: string;
  h5File: File;
  useWebGPU: boolean;
  maxTextureSize: number;
  labelIndices: Int16Array;
  featureImportanceOnly?: boolean; // If true, skip IVFPQ loading and cell processing
}

interface H5DataSet {
  type: string;
  value: unknown;
  shape: number[];
  keys(): string[];
  slice(ranges: number[][]): unknown;
}

interface H5Group {
  type: string;
  keys(): string[];
  get(path: string): H5DataSet | H5Group;
}

interface H5File {
  keys(): string[];
  get(path: string): H5DataSet | H5Group;
  close(): void;
}

interface RawCountsData {
  isSparse: boolean;
  data: H5DataSet;
  indices: H5DataSet | null;
  indptr: H5DataSet | null;
}

interface CellUpdate {
  cellId: string;
  x: number;
  y: number;
  label?: string;
  labelId?: number;
  confidence?: number;
}

interface CellBatchUpdate {
  type: 'cell_batch_update';
  cells: CellUpdate[];
}

interface StatusMessage {
  type: 'status';
  message: string;
}

interface ProgressMessage {
  type: 'progress';
  message: string;
  countFinished: number;
  totalToProcess: number;
}

interface ErrorMessage {
  type: 'error';
  error: string;
}

interface FinishedMessage {
  type: 'finished';
  datasetLabel: string;
  totalProcessed: number;
  cellNames: string[];
  partitionIds: Uint16Array;
  pqCodes: Uint8Array[];
}

interface FeatureImportanceRequestMessage {
  type: 'calculate_feature_importance';
  cellId: string;
  cellIndex: number;
  topN?: number;
}

interface FeatureImportanceProgressMessage {
  type: 'feature_importance_progress';
  cellId: string;
  progress: number;
}

interface FeatureImportanceResultMessage {
  type: 'feature_importance_result';
  cellId: string;
  genes: Array<{
    gene: string;
    importance: number;
    expression: number;
    index: number;
  }>;
}

// Global state
let model: ModelInfo | null = null;
// let labelIndices: Int16Array | null = null;

// Global state for h5 file and feature importance
// @ts-ignore - Used for future feature importance operations
let currentH5File: any = null;
// @ts-ignore - Used for future feature importance operations
let currentCellNames: string[] = [];
let currentRawData: {
  data: H5DataSet;
  indices: H5DataSet | null;
  indptr: H5DataSet | null;
  isSparse: boolean;
  sampleGenes: string[];
  inflationIndices: number[];
} | null = null;

// Feature importance cancellation
let currentFeatureImportanceCellId: string | null = null;
let cancelFeatureImportance = false;

// Handle messages from main thread
self.addEventListener(
  'message',
  async function (event: MessageEvent<StartMessage | FeatureImportanceRequestMessage>) {
    console.log('Worker received message:', event.data.type);

    if (event.data.type === 'start') {
      start(
        event.data.modelID,
        event.data.modelsURL,
        event.data.h5File,
        event.data.useWebGPU,
        event.data.maxTextureSize,
        event.data.labelIndices,
        event.data.useUserIndex,
        event.data.userIndexId,
        event.data.featureImportanceOnly || false
      );
    } else if (event.data.type === 'calculate_feature_importance') {
      console.log('Worker: calculate_feature_importance message received');
      // Cancel any existing feature importance calculation
      if (currentFeatureImportanceCellId !== null) {
        console.log(
          `Cancelling feature importance for ${currentFeatureImportanceCellId}, starting new calculation for ${event.data.cellId}`
        );
        cancelFeatureImportance = true;
        // Wait a bit for the cancellation to take effect
        await new Promise((resolve) => setTimeout(resolve, 50));
      }

      // Reset cancellation flag and start new calculation
      cancelFeatureImportance = false;
      currentFeatureImportanceCellId = event.data.cellId;

      // Don't await - let it run in background
      calculateFeatureImportance(
        event.data.cellId,
        event.data.cellIndex,
        event.data.topN || 10
      ).catch((error) => {
        console.error('Unhandled error in calculateFeatureImportance:', error);
        self.postMessage({
          type: 'error',
          error: `Feature importance failed: ${error instanceof Error ? error.message : String(error)}`,
        } as ErrorMessage);
      });
    }
  }
);

/**
 * Initialize model and IVFPQ components
 */
async function instantiateModel(
  modelsURL: string,
  modelID: string,
  useWebGPU: boolean,
  maxTextureSize: number,
  useUserIndex: boolean = false,
  userIndexId?: string,
  skipIVFPQ: boolean = false
): Promise<ModelInfo> {
  console.log(`Instantiating model ${modelID} from ${modelsURL}`);
  // self.postMessage({ type: 'status', message: 'Downloading model...' } as StatusMessage)

  // Fetch foundation-model metadata. Missing or malformed metadata is treated as
  // legacy SCimilarity for backward compatibility with existing deployments.
  let metadata: ModelMetadata = { foundationModel: 'scimilarity', embeddingDim: 128 };
  try {
    const metaResp = await fetch(`${modelsURL}/${modelID}/metadata.json`);
    if (metaResp.ok) {
      metadata = { ...metadata, ...(await metaResp.json()) };
    }
  } catch (e) {
    console.warn(`No metadata.json for ${modelID}, defaulting to scimilarity`, e);
  }
  console.log(`Foundation model: ${metadata.foundationModel}`);

  // Configure ONNX Runtime (shared by both foundation models)
  env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 4);
  env.wasm.proxy = true;

  // Branch on foundation model: each loads its own embedding session + any
  // model-specific assets. Mapping session + IVFPQ setup is shared below.
  let genes: string[] = [];
  let embeddingSession: InferenceSession;
  let uceBrain: UCEBrainModel | null = null;

  if (metadata.foundationModel === 'scimilarity') {
    ({ genes, embeddingSession } = await loadScimilarityEmbedding(
      modelsURL,
      modelID,
      useWebGPU,
      maxTextureSize
    ));
  } else if (metadata.foundationModel === 'uce-brain') {
    ({ embeddingSession, uceBrain } = await loadUCEBrainEmbedding(
      modelsURL,
      modelID,
      useWebGPU
    ));
  } else {
    throw new Error(`Unknown foundation model: ${metadata.foundationModel}`);
  }

  // Create mapping session (same config shape regardless of foundation model)
  let mappingSessionOptions: Record<string, unknown>;
  if (useWebGPU) {
    console.log('Configuring mapping session to use WebGPU...');
    mappingSessionOptions = {
      executionProviders: [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance',
        },
      ],
      graphOptimizationLevel: 'all',
    };
  } else {
    console.log('Configuring mapping session to use WebAssembly...');
    mappingSessionOptions = {
      executionProviders: ['wasm'],
      executionMode: 'parallel',
      graphOptimizationLevel: 'all',
    };
  }

  const mappingSession = await InferenceSession.create(
    `${modelsURL}/${modelID}/pumap/model.onnx`,
    mappingSessionOptions
  );
  console.log('Mapper Output names', mappingSession.outputNames);

  // Load IVFPQ or UserIVFPQ system (skip if only doing feature importance)
  let ivfpq: IVFPQ | UserIVFPQ;
  if (!skipIVFPQ) {
    const ivfpqBasePath = `${modelsURL}/${modelID}/ivfpq`;
    if (useUserIndex && userIndexId) {
      ivfpq = new UserIVFPQ(
        ivfpqBasePath,
        NUM_PARTITIONS_TO_SEARCH,
        NUM_NEAREST_NEIGHBORS
      );
      await (ivfpq as UserIVFPQ).loadUserIndex(userIndexId, useWebGPU);
      console.log('UserIVFPQ system loaded successfully');
    } else {
      ivfpq = new IVFPQ(ivfpqBasePath, NUM_PARTITIONS_TO_SEARCH, NUM_NEAREST_NEIGHBORS);
      await ivfpq.load(useWebGPU);
      console.log('IVFPQ system loaded successfully');
    }
  } else {
    console.log('Skipping IVFPQ loading (feature importance only mode)');
    // Create a dummy IVFPQ for type safety (won't be used for feature importance)
    // @ts-expect-error - IVFPQ not needed for feature importance calculations
    ivfpq = null;
  }

  return {
    modelID,
    foundationModel: metadata.foundationModel,
    genes,
    embeddingSession,
    mappingSession,
    ivfpq,
    uceBrain,
  };
}

/**
 * Load SCimilarity embedding model: genes.txt + model.onnx.
 * Progress is reported via ProgressMessage during the model download.
 */
async function loadScimilarityEmbedding(
  modelsURL: string,
  modelID: string,
  useWebGPU: boolean,
  maxTextureSize: number
): Promise<{ genes: string[]; embeddingSession: InferenceSession }> {
  let response = await fetch(`${modelsURL}/${modelID}/embedding/genes.txt`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const genes = (await response.text()).split('\n');
  console.log('Model Genes', genes.slice(0, 5));

  response = await fetch(`${modelsURL}/${modelID}/embedding/model.onnx`);
  if (!response.ok) {
    throw new Error(`Error fetching onnx file: ${response.status}`);
  }

  const contentLength = response.headers.get('content-length');
  const totalBytes = contentLength ? parseInt(contentLength, 10) : null;

  const chunks: Uint8Array[] = [];
  const reader = response.body!.getReader();
  let receivedBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedBytes += value.length;

    self.postMessage({
      type: 'progress',
      message: 'Downloading model...',
      countFinished: Math.round(receivedBytes / (1024 * 1024)),
      totalToProcess: totalBytes ? Math.round(totalBytes / (1024 * 1024)) : 0,
    } as ProgressMessage);
  }

  const modelArray = new Uint8Array(receivedBytes);
  let position = 0;
  for (const chunk of chunks) {
    modelArray.set(chunk, position);
    position += chunk.length;
  }

  self.postMessage({
    type: 'status',
    message: 'Instantiating model...',
  } as StatusMessage);

  let sessionOptions: Record<string, unknown>;
  if (useWebGPU && maxTextureSize >= genes.length) {
    console.log('Configuring embedding session to use WebGPU...');
    sessionOptions = {
      executionProviders: [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance',
        },
      ],
      graphOptimizationLevel: 'all',
    };
  } else {
    if (useWebGPU && maxTextureSize < genes.length) {
      console.warn(
        `Embeddings dimention of ${genes.length} exceeds max gpu texture size of ${maxTextureSize} (likely running on Apple Silicon).`
      );
      console.warn('Falling back to WebAssembly execution for embedding only');
    }
    console.log('Configuring embedding session to use WebAssembly...');
    sessionOptions = {
      executionProviders: ['wasm'],
      executionMode: 'parallel',
      graphOptimizationLevel: 'all',
    };
  }

  const embeddingSession = await InferenceSession.create(
    modelArray.buffer,
    sessionOptions
  );
  console.log('Model Output names', embeddingSession.outputNames);

  return { genes, embeddingSession };
}

/**
 * Load UCE-brain embedding model:
 *   transformer.onnx  (~117 MB)
 *   protein_embeddings.bin  (~400 MB; Cache API for durability across reloads)
 *   gene_dict.json  (~1.5 MB)
 *
 * The transformer *must* run on WebGPU for reasonable performance (CPU is 10x
 * slower at seq_len=1024), so we pick WebAssembly only when WebGPU is
 * unavailable. The protein embedding table lives in JS memory and feeds the
 * per-cell gather — it is not uploaded to the GPU.
 */
async function loadUCEBrainEmbedding(
  modelsURL: string,
  modelID: string,
  useWebGPU: boolean
): Promise<{ embeddingSession: InferenceSession; uceBrain: UCEBrainModel }> {
  const base = `${modelsURL}/${modelID}/embedding`;

  // Fetch gene_dict (small, not cached — spec may change during development)
  self.postMessage({
    type: 'status',
    message: 'Loading UCE-brain gene dict...',
  } as StatusMessage);
  const geneDictResp = await fetch(`${base}/gene_dict.json`);
  if (!geneDictResp.ok) {
    throw new Error(`Failed to fetch gene_dict.json: ${geneDictResp.status}`);
  }
  const geneDict = (await geneDictResp.json()) as UCEGeneDict;
  console.log(
    `UCE-brain gene_dict: ${Object.keys(geneDict.genes).length} genes, ` +
      `table ${geneDict.num_rows}×${geneDict.embedding_dim}`
  );

  // Fetch protein embedding table via Cache API. 400 MB — one-time download.
  self.postMessage({
    type: 'status',
    message: 'Loading protein embeddings (~400 MB)...',
  } as StatusMessage);
  const tableUrl = `${base}/protein_embeddings.bin`;
  const tableResult = await cachedFetchBytes(tableUrl);
  const proteinEmbeddings = new Float32Array(tableResult.buffer);
  const expectedLen = geneDict.num_rows * geneDict.embedding_dim;
  if (proteinEmbeddings.length !== expectedLen) {
    throw new Error(
      `protein_embeddings.bin length ${proteinEmbeddings.length} != ` +
        `expected ${expectedLen} (${geneDict.num_rows} × ${geneDict.embedding_dim})`
    );
  }
  console.log(
    `Protein embeddings loaded: ${(tableResult.buffer.byteLength / 1e6).toFixed(0)} MB, ` +
      `${tableResult.fromCache ? 'cache HIT' : 'cache MISS'}, ${tableResult.ms.toFixed(0)} ms`
  );

  // Fetch transformer.onnx with progress
  self.postMessage({
    type: 'status',
    message: 'Downloading UCE-brain transformer...',
  } as StatusMessage);
  const xfResp = await fetch(`${base}/transformer.onnx`);
  if (!xfResp.ok) {
    throw new Error(`Failed to fetch transformer.onnx: ${xfResp.status}`);
  }
  const xfContentLength = xfResp.headers.get('content-length');
  const xfTotalBytes = xfContentLength ? parseInt(xfContentLength, 10) : null;
  const xfChunks: Uint8Array[] = [];
  const xfReader = xfResp.body!.getReader();
  let xfReceived = 0;
  while (true) {
    const { done, value } = await xfReader.read();
    if (done) break;
    xfChunks.push(value);
    xfReceived += value.length;
    self.postMessage({
      type: 'progress',
      message: 'Downloading UCE-brain transformer...',
      countFinished: Math.round(xfReceived / (1024 * 1024)),
      totalToProcess: xfTotalBytes ? Math.round(xfTotalBytes / (1024 * 1024)) : 0,
    } as ProgressMessage);
  }
  const xfArray = new Uint8Array(xfReceived);
  let xfPos = 0;
  for (const chunk of xfChunks) {
    xfArray.set(chunk, xfPos);
    xfPos += chunk.length;
  }

  self.postMessage({
    type: 'status',
    message: 'Instantiating UCE-brain transformer...',
  } as StatusMessage);

  // WebGPU first, WASM fallback. The transformer is the performance-critical
  // path — without GPU this will be slow.
  let sessionOptions: Record<string, unknown>;
  if (useWebGPU) {
    console.log('Configuring UCE-brain embedding session to use WebGPU...');
    sessionOptions = {
      executionProviders: [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance',
        },
      ],
      graphOptimizationLevel: 'all',
    };
  } else {
    console.warn(
      'UCE-brain requires WebGPU for reasonable performance. ' +
        'Falling back to WebAssembly — expect 10x slower per-cell inference.'
    );
    sessionOptions = {
      executionProviders: ['wasm'],
      executionMode: 'parallel',
      graphOptimizationLevel: 'all',
    };
  }

  const embeddingSession = await InferenceSession.create(
    xfArray.buffer,
    sessionOptions
  );
  console.log('UCE-brain transformer output names:', embeddingSession.outputNames);

  // Build the sentence-tokens helper (numeric chromosome map — gene_dict has
  // string keys from JSON).
  const chromosomeTokenMap: Record<number, number> = {};
  for (const [k, v] of Object.entries(geneDict.chromosome_token_map)) {
    chromosomeTokenMap[parseInt(k, 10)] = v;
  }
  const sentenceTokens: UCESentenceTokens = {
    specials: geneDict.specials,
    chromosomeTokenMap,
  };

  return {
    embeddingSession,
    uceBrain: {
      proteinEmbeddings,
      tableRows: geneDict.num_rows,
      embDim: geneDict.embedding_dim,
      geneDict,
      sentenceTokens,
    },
  };
}

/**
 * Extract cell names from h5ad file
 */
function getCellNames(annData: H5File): string[] {
  if (!annData.keys().includes('obs')) {
    throw new Error('Unable to find cell names: Missing "obs" group in h5ad file');
  }

  const obs = annData.get('obs');

  if (obs.type === 'Dataset') {
    try {
      const obsValue = (obs as H5DataSet).value as Array<[string]>;
      return obsValue.map((e) => e[0]);
    } catch {
      throw new Error('Unable to extract cell names from obs Dataset');
    }
  }

  if (obs.type === 'Group') {
    const obsGroup = obs as H5Group;
    const obsKeys = obsGroup.keys();
    const indexKeys = [
      'index', // Standard Pandas/Scanpy export.
      'barcodes', // 10x Genomics standard.
      'Cells', // Seurat (R) conversion standard.
      'barcode', // Common manual naming.
      'cell_id', // Metadata/Atlas standard.
      '_index', // Automated backup/safety name.
      'cell_name', // Legacy/custom pipelines.
      '_barcodes', // Versioned/filtered backup.
    ];

    for (const key of indexKeys) {
      if (obsKeys.includes(key)) {
        try {
          return (obsGroup.get(key) as H5DataSet).value as string[];
        } catch {
          console.warn(`Failed to read cell names from obs/${key}`);
        }
      }
    }

    throw new Error(
      `Unable to find cell names in obs group. Available keys: ${obsKeys.join(', ')}`
    );
  }

  throw new Error('Unable to find cell names: obs is neither Dataset nor Group');
}

/**
 * Validate gene symbols
 */
function validateGeneSymbols(geneArray: string[]): boolean {
  const hugoMarkers = [
    'ACTB',   // Beta-actin, ubiquitous housekeeping gene
    'GAPDH',  // Classic housekeeping enzyme
    'MALAT1', // Highly abundant lncRNA in virtually all cells
    'B2M',    // Beta-2 microglobulin, present in all nucleated cells
    'PTPRC',  // CD45, pan-leukocyte marker
    'RPL13',  // Ribosomal protein, extremely abundant
    'TMSB4X', // Thymosin beta-4, one of most abundant transcripts
    'TP53',   // Tumor suppressor, widely studied
  ];
  const upperGenes = new Set(geneArray.map((g) => g.toUpperCase()));
  const matches = hugoMarkers.filter((marker) => upperGenes.has(marker)).length;
  return matches >= 2;
}

/**
 * Extract gene names from h5ad file
 */
function getSampleGenes(annData: H5File): string[] {
  if (!annData.keys().includes('var')) {
    throw new Error('Unable to find gene names: Missing "var" group in h5ad file');
  }

  const varData = annData.get('var');

  if (varData.type === 'Dataset') {
    try {
      const varValue = (varData as H5DataSet).value as Array<[string]>;
      return varValue.map((e) => e[0].toUpperCase());
    } catch {
      throw new Error('Unable to extract gene names from var Dataset');
    }
  }

  if (varData.type === 'Group') {
    const varGroup = varData as H5Group;
    const varKeys = varGroup.keys();
    const geneKeys = [
      'feature_name', // Direct 10x Genomics output; very common.
      'gene_symbols', // Standard naming convention in Python pipelines.
      'gene_name', // Standard naming in R-to-Python conversions.
      'symbol', // Common manual/shorthand naming.
      'index', // The fallback for reset dataframes.
      'gene_ids', // Usually Ensembl IDs; used if symbols are missing.
      '_index', // "The ""emergency"" fallback."
      // Other common variations to check for:
      'gene_symbol',
      'gene_names',
      'feature_name',
      'features',
      'gene_id',
      'Genes',
    ];

    for (const key of geneKeys) {
      if (varKeys.includes(key)) {
        try {
          const genes = (varGroup.get(key) as H5DataSet).value as string[];
          if (validateGeneSymbols(genes)) {
            console.log(`Found gene names in var/${key}`);
            return genes.map((g) => g.toUpperCase());
          }
        } catch {
          console.warn(`Failed to read gene names from var/${key}`);
        }
      }
    }

    if (varKeys.length > 0) {
      try {
        const firstKey = varKeys[0];
        const potentialGenes = (varGroup.get(firstKey) as H5DataSet).value as string[];
        if (Array.isArray(potentialGenes) && validateGeneSymbols(potentialGenes)) {
          console.warn('Falling back to using var index as gene names');
          return potentialGenes.map((g) => g.toUpperCase());
        }
      } catch {
        console.warn(
          'Failed to read potential gene names from first available var key'
        );
      }
    }

    throw new Error(
      `Unable to find gene names in var group. Available keys: ${varKeys.join(', ')}`
    );
  }

  throw new Error('Unable to find gene names: var is neither Dataset nor Group');
}

/**
 * Extract raw counts from h5ad file
 */
function getRawCounts(annData: H5File): RawCountsData {
  const topLevelKeys = annData.keys();

  // Check layers for raw counts
  if (topLevelKeys.includes('layers')) {
    const layers = annData.get('layers');
    if (layers.type === 'Group') {
      const layersGroup = layers as H5Group;
      const layerKeys = layersGroup.keys();
      const rawCountKeys = [
        'counts',
        'raw_counts',
        'raw',
        'count',
        'spliced',
        'unspliced',
      ];

      for (const key of rawCountKeys) {
        if (layerKeys.includes(key)) {
          console.log(`Found raw counts in layers/${key}`);
          const countsData = layersGroup.get(key);

          if (countsData.type === 'Dataset') {
            return {
              isSparse: false,
              data: countsData as H5DataSet,
              indices: null,
              indptr: null,
            };
          }

          if (countsData.type === 'Group') {
            const sparseGroup = countsData as H5Group;
            const sparseKeys = sparseGroup.keys();

            if (
              sparseKeys.includes('data') &&
              sparseKeys.includes('indices') &&
              sparseKeys.includes('indptr')
            ) {
              return {
                isSparse: true,
                data: sparseGroup.get('data') as H5DataSet,
                indices: sparseGroup.get('indices') as H5DataSet,
                indptr: sparseGroup.get('indptr') as H5DataSet,
              };
            }
          }
        }
      }
    }
  }

  // Check raw attribute
  if (topLevelKeys.includes('raw')) {
    const raw = annData.get('raw');
    if (raw.type === 'Group') {
      const rawGroup = raw as H5Group;
      if (rawGroup.keys().includes('X')) {
        const rawX = rawGroup.get('X');

        if (rawX.type === 'Dataset') {
          return {
            isSparse: false,
            data: rawX as H5DataSet,
            indices: null,
            indptr: null,
          };
        }

        if (rawX.type === 'Group') {
          const sparseGroup = rawX as H5Group;
          const sparseKeys = sparseGroup.keys();

          if (
            sparseKeys.includes('data') &&
            sparseKeys.includes('indices') &&
            sparseKeys.includes('indptr')
          ) {
            return {
              isSparse: true,
              data: sparseGroup.get('data') as H5DataSet,
              indices: sparseGroup.get('indices') as H5DataSet,
              indptr: sparseGroup.get('indptr') as H5DataSet,
            };
          }
        }
      }
    }
  }

  // Use main X matrix as last resort
  if (topLevelKeys.includes('X')) {
    console.warn(
      'Using main X matrix - this may contain normalized data instead of raw counts'
    );
    const X = annData.get('X');

    if (X.type === 'Dataset') {
      return {
        isSparse: false,
        data: X as H5DataSet,
        indices: null,
        indptr: null,
      };
    }

    if (X.type === 'Group') {
      const sparseGroup = X as H5Group;
      const sparseKeys = sparseGroup.keys();

      if (
        sparseKeys.includes('data') &&
        sparseKeys.includes('indices') &&
        sparseKeys.includes('indptr')
      ) {
        return {
          isSparse: true,
          data: sparseGroup.get('data') as H5DataSet,
          indices: sparseGroup.get('indices') as H5DataSet,
          indptr: sparseGroup.get('indptr') as H5DataSet,
        };
      }
    }
  }

  const availableKeys = topLevelKeys.join(', ');
  throw new Error(
    `Unable to find raw counts data. Looked in: layers/counts, layers/raw_counts, raw/X, and X. ` +
      `Available top-level keys in h5ad file: ${availableKeys}`
  );
}

/**
 * Precompute inflation indices for gene mapping
 */
function precomputeInflationIndices(
  currentModelGenes: string[],
  sampleGenes: string[]
): number[] {
  const inflationIndices: number[] = [];
  for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
    inflationIndices.push(currentModelGenes.indexOf(sampleGenes[geneIndex]));
  }
  const missingGenesInModel = inflationIndices.filter((x) => x === -1).length;
  console.log(`Missing genes in model: ${missingGenesInModel}`);
  return inflationIndices;
}

/**
 * Fill batch data and inflate to model gene space
 */
function fillBatchData(
  batchStart: number,
  currentBatchSize: number,
  data: H5DataSet,
  indices: H5DataSet | null,
  indptr: H5DataSet | null,
  isSparse: boolean,
  sampleGenes: string[],
  inflationIndices: number[],
  inflatedBatchData: Float32Array
): void {
  for (let batchSlot = 0; batchSlot < currentBatchSize; batchSlot++) {
    const cellIndex = batchStart + batchSlot;
    const batchOffset = batchSlot * model!.genes.length;

    if (isSparse) {
      const [start, end] = indptr!.slice([[cellIndex, cellIndex + 2]]) as number[];
      const values = data.slice([[start, end]]) as number[];
      const valueIndices = indices!.slice([[start, end]]) as number[];

      for (let j = 0; j < valueIndices.length; j++) {
        const sampleIndex = inflationIndices[valueIndices[j]];
        if (sampleIndex !== -1) {
          inflatedBatchData[batchOffset + sampleIndex] = Number(values[j]);
        }
      }
    } else {
      let sampleExpression: number[] | null = null;
      if (data.shape.length === 1) {
        sampleExpression = data.slice([
          [cellIndex * sampleGenes.length, (cellIndex + 1) * sampleGenes.length],
        ]) as number[];
      } else if (data.shape.length === 2) {
        sampleExpression = data.slice([
          [cellIndex, cellIndex + 1],
          [0, sampleGenes.length],
        ]) as number[];
      } else {
        throw new Error('Unsupported data shape');
      }
      for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
        const sampleIndex = inflationIndices[geneIndex];
        if (sampleIndex !== -1) {
          inflatedBatchData[batchOffset + sampleIndex] = sampleExpression[geneIndex];
        }
      }
    }
  }
}

/**
 * Label cells using IVFPQ search
 */
async function labelCells(
  embeddings: Float32Array,
  batchSize: number,
  embeddingDim: number,
  labelIndices: Int16Array,
  useUserIndex: boolean
): Promise<{
  labelIds: number[];
  confidences: number[];
  partitionIds: Uint16Array;
  pqCodes: Uint8Array[];
}> {
  const labelIds: number[] = [];
  const confidences: number[] = [];
  const partitionIds: Uint16Array = new Uint16Array(batchSize);
  const pqCodes: Uint8Array[] = new Array(batchSize);

  for (let i = 0; i < batchSize; i++) {
    try {
      // Extract embedding for this cell (zero-copy view)
      const offset = i * embeddingDim;
      const queryVector = embeddings.subarray(offset, offset + embeddingDim);

      // Search using IVFPQ with optional artifact retention
      const searchResults: SearchResults = await model!.ivfpq.search(queryVector);

      // Compute consensus label
      let consensusLabelId = -1;
      let consensusConfidence = 0;

      if (searchResults.indices.length > 0 && labelIndices) {
        const labelVotes: { [labelId: number]: number } = {};

        // UserIVFPQ returns label indices directly, IVFPQ returns train vector indices
        // Both work with the same labelIndices array passed from baseRef
        for (const indexValue of searchResults.indices) {
          if (indexValue !== -1 && indexValue < labelIndices.length) {
            const labelId = useUserIndex ? indexValue : labelIndices[indexValue];
            if (labelId >= 0) {
              labelVotes[labelId] = (labelVotes[labelId] || 0) + 1;
            }
          }
        }

        let maxVotes = 0;
        for (const [labelId, votes] of Object.entries(labelVotes)) {
          if (votes > maxVotes) {
            maxVotes = votes;
            consensusLabelId = parseInt(labelId);
          }
        }

        if (maxVotes > 0) {
          consensusConfidence = maxVotes / NUM_NEAREST_NEIGHBORS;
        }
      }

      labelIds.push(consensusLabelId);
      confidences.push(consensusConfidence);

      // Additional artifacts to create a user index
      partitionIds[i] = searchResults.partitionId;
      pqCodes[i] = searchResults.pqCode;
    } catch (error) {
      console.error(`Error processing vector ${i}:`, error);
    }
  }

  return { labelIds, confidences, partitionIds, pqCodes };
}

/**
 * Main processing function
 */
async function start(
  modelID: string,
  modelsURL: string,
  h5File: File,
  useWebGPU: boolean,
  maxTextureSize: number,
  labelIndices: Int16Array,
  useUserIndex: boolean = false,
  userIndexId?: string,
  featureImportanceOnly: boolean = false
): Promise<void> {
  console.log(
    `Starting unified worker for model ${modelID} with file ${h5File.name}${featureImportanceOnly ? ' (feature importance only)' : ''}`
  );

  self.postMessage({
    type: 'status',
    message: 'Loading libraries...',
  } as StatusMessage);
  const Module = await h5wasm.ready;
  const { FS } = Module;
  console.log('h5wasm loaded');

  try {
    // Load model if needed
    if (!model || model.modelID !== modelID) {
      model = await instantiateModel(
        modelsURL,
        modelID,
        useWebGPU,
        maxTextureSize,
        useUserIndex,
        userIndexId,
        featureImportanceOnly
      );
    }

    // Mount h5 file
    self.postMessage({
      type: 'status',
      message: 'Loading file...',
    } as StatusMessage);
    if (!FS.analyzePath('/work').exists) {
      FS.mkdir('/work');
    }
    FS.mount(FS.filesystems.WORKERFS, { files: [h5File] }, '/work');

    const annData = new h5wasm.File(`/work/${h5File.name}`, 'r') as H5File;
    console.log(`Top level keys: ${annData.keys()}`);

    // Store globally for feature importance
    currentH5File = annData;

    // Extract metadata
    let cellNames: string[];
    try {
      cellNames = getCellNames(annData);
      currentCellNames = cellNames;
      console.log(`Found ${cellNames.length} cells`);
    } catch (error) {
      throw new Error(
        `Failed to extract cell names: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }

    let sampleGenes: string[];
    try {
      sampleGenes = getSampleGenes(annData);
      console.log(`Found ${sampleGenes.length} genes`);
    } catch (error) {
      throw new Error(
        `Failed to extract gene names: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }

    // Extract raw counts
    let rawCountsData: RawCountsData;
    try {
      rawCountsData = getRawCounts(annData);
      console.log(`Found expression data (sparse: ${rawCountsData.isSparse})`);
    } catch (error) {
      throw new Error(
        `Failed to extract expression data: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }

    const { isSparse, data, indices, indptr } = rawCountsData;

    // SCimilarity needs gene-to-model alignment indices; UCE-brain does not
    // (its alignment happens via the gene_dict protein table).
    const inflationIndices =
      model.foundationModel === 'scimilarity'
        ? precomputeInflationIndices(model.genes, sampleGenes)
        : [];

    // Store data globally for feature importance. inflationIndices is empty
    // for UCE-brain — the feature-importance path short-circuits with a
    // "not supported" error for that foundation model.
    currentRawData = {
      data,
      indices,
      indptr,
      isSparse,
      sampleGenes,
      inflationIndices,
    };

    // Notify that we're ready for feature importance calculations
    self.postMessage({
      type: 'ready_for_feature_importance',
    });

    // If feature importance only mode, skip cell processing
    if (featureImportanceOnly) {
      console.log('Feature importance only mode - skipping cell processing');
      return;
    }

    self.postMessage({
      type: 'status',
      message: 'Processing cells...',
    } as StatusMessage);

    const allPartitionIds: Uint16Array = new Uint16Array(cellNames.length);
    const allPQCodes: Uint8Array[] = new Array(cellNames.length);

    if (model.foundationModel === 'scimilarity') {
      await processCellsScimilarity(
        cellNames,
        sampleGenes,
        inflationIndices,
        rawCountsData,
        labelIndices,
        useUserIndex,
        allPartitionIds,
        allPQCodes
      );
    } else if (model.foundationModel === 'uce-brain') {
      await processCellsUCEBrain(
        cellNames,
        sampleGenes,
        rawCountsData,
        labelIndices,
        useUserIndex,
        allPartitionIds,
        allPQCodes
      );
    } else {
      throw new Error(
        `Unknown foundation model at processing time: ${model.foundationModel}`
      );
    }

    // Clean up
    // Keep annData open for feature importance calculations
    // annData.close();
    // FS.unmount('/work');

    const finishedMessage: FinishedMessage = {
      type: 'finished',
      datasetLabel: h5File.name,
      totalProcessed: cellNames.length,
      cellNames: cellNames,
      partitionIds: allPartitionIds,
      pqCodes: allPQCodes,
    };

    self.postMessage(finishedMessage);
  } catch (error) {
    try {
      FS.unmount('/work');
    } catch (unmountError) {
      console.error('Failed to unmount file system:', unmountError);
    }

    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('Worker error:', errorMessage);
    self.postMessage({
      type: 'error',
      error: errorMessage,
    } as ErrorMessage);
  }
}

/**
 * Batched SCimilarity processing loop. Inflates each batch into the model's
 * gene space, runs the embedding + PUMAP sessions, and emits labels via IVFPQ.
 */
async function processCellsScimilarity(
  cellNames: string[],
  sampleGenes: string[],
  inflationIndices: number[],
  rawCountsData: RawCountsData,
  labelIndices: Int16Array,
  useUserIndex: boolean,
  allPartitionIds: Uint16Array,
  allPQCodes: Uint8Array[]
): Promise<void> {
  const { isSparse, data, indices, indptr } = rawCountsData;
  const batchDataBuffer = new Float32Array(BATCH_SIZE * model!.genes.length);

  for (let batchStart = 0; batchStart < cellNames.length; batchStart += BATCH_SIZE) {
    const batchEnd = Math.min(batchStart + BATCH_SIZE, cellNames.length);
    const currentBatchSize = batchEnd - batchStart;

    const batchData =
      currentBatchSize === BATCH_SIZE
        ? batchDataBuffer
        : batchDataBuffer.subarray(0, currentBatchSize * model!.genes.length);
    batchData.fill(0);
    fillBatchData(
      batchStart,
      currentBatchSize,
      data,
      indices,
      indptr,
      isSparse,
      sampleGenes,
      inflationIndices,
      batchData
    );

    const inputTensor = new Tensor('float32', batchData, [
      currentBatchSize,
      model!.genes.length,
    ]);
    const embeddingResults = await model!.embeddingSession.run({ input: inputTensor });
    const mappingResults = await model!.mappingSession.run({
      input: embeddingResults.output,
    });

    const coordinates: number[][] = [];
    for (let i = 0; i < mappingResults.output.dims[0]; i++) {
      const startIndex = i * 2;
      coordinates.push([
        Number(mappingResults.output.data[startIndex]),
        Number(mappingResults.output.data[startIndex + 1]),
      ]);
    }

    const embeddingDim = embeddingResults.output.dims[1] as number;
    const { labelIds, confidences, partitionIds, pqCodes } = await labelCells(
      embeddingResults.output.data as Float32Array,
      currentBatchSize,
      embeddingDim,
      labelIndices,
      useUserIndex
    );

    allPartitionIds.set(partitionIds, batchStart);
    for (let i = 0; i < currentBatchSize; i++) {
      allPQCodes[batchStart + i] = pqCodes[i];
    }

    emitBatchUpdates(
      cellNames,
      batchStart,
      currentBatchSize,
      coordinates,
      labelIds,
      confidences
    );

    self.postMessage({
      type: 'progress',
      message: 'Labeling',
      countFinished: batchEnd,
      totalToProcess: cellNames.length,
    } as ProgressMessage);

    embeddingResults.output.dispose();
    mappingResults.output.dispose();
  }
}

/**
 * Per-cell UCE-brain processing loop. Runs the full JS pipeline (normalize →
 * weighted-sample → sentence → gather) then the transformer, one cell at a
 * time. Per uce-edge empirical results, batching across cells hurts because
 * transformer attention is O(L²); larger L per cell dominates over any
 * per-step overhead from B=1 vs B>1.
 *
 * PUMAP and IVFPQ are still runnable on batched vectors — we accumulate a
 * small flush buffer and call them every N cells.
 */
async function processCellsUCEBrain(
  cellNames: string[],
  sampleGenes: string[],
  rawCountsData: RawCountsData,
  labelIndices: Int16Array,
  useUserIndex: boolean,
  allPartitionIds: Uint16Array,
  allPQCodes: Uint8Array[]
): Promise<void> {
  if (model!.uceBrain === null) {
    throw new Error('UCE-brain processing requested but uceBrain model is null');
  }
  const { proteinEmbeddings, tableRows, embDim, geneDict, sentenceTokens } =
    model!.uceBrain;
  const { isSparse, data, indices, indptr } = rawCountsData;
  const G = sampleGenes.length;

  // Align sample genes to the UCE-brain protein table (per-h5ad, once).
  const geneTable = buildGeneTable(sampleGenes, geneDict);
  const nAligned = geneTable.symbols.length;
  console.log(
    `UCE-brain gene alignment: ${nAligned}/${sampleGenes.length} sample genes ` +
      `matched to the protein table`
  );
  if (nAligned < 100) {
    throw new Error(
      `Only ${nAligned} of ${sampleGenes.length} sample genes match the UCE-brain ` +
        `protein table. Check gene-symbol normalization.`
    );
  }

  // Reusable per-cell buffers
  const rawCell = new Float32Array(G);
  const alignedCounts = new Float32Array(nAligned);

  // Flush cadence for PUMAP + IVFPQ. 64 is a round number that keeps UI
  // updates frequent while amortizing the batch session overhead.
  const FLUSH = 64;
  // embeddingDim for UCE-brain is the transformer output dim (typically 512).
  // We discover this from the first inference and grow the buffer on-demand.
  let flushBuffer: Float32Array | null = null;
  let embeddingDim = -1;
  let bufCount = 0;
  const bufCellIndices = new Int32Array(FLUSH);

  async function flushBatch(): Promise<void> {
    if (bufCount === 0 || flushBuffer === null) return;
    const batchView = flushBuffer.subarray(0, bufCount * embeddingDim);

    // PUMAP
    const mappingTensor = new Tensor('float32', batchView, [bufCount, embeddingDim]);
    const mappingResults = await model!.mappingSession.run({ input: mappingTensor });
    const coordinates: number[][] = [];
    for (let i = 0; i < mappingResults.output.dims[0]; i++) {
      const start = i * 2;
      coordinates.push([
        Number(mappingResults.output.data[start]),
        Number(mappingResults.output.data[start + 1]),
      ]);
    }

    // IVFPQ labeling (search one cell at a time, as in the SCimilarity path)
    const { labelIds, confidences, partitionIds, pqCodes } = await labelCells(
      batchView,
      bufCount,
      embeddingDim,
      labelIndices,
      useUserIndex
    );

    // Scatter artifacts back by original cell index
    for (let i = 0; i < bufCount; i++) {
      const cellIdx = bufCellIndices[i];
      allPartitionIds[cellIdx] = partitionIds[i];
      allPQCodes[cellIdx] = pqCodes[i];
    }

    // Emit labeled cells
    const labeledBatch: CellUpdate[] = [];
    for (let i = 0; i < bufCount; i++) {
      if (labelIds[i] !== -1) {
        const cellIdx = bufCellIndices[i];
        labeledBatch.push({
          cellId: cellNames[cellIdx],
          x: coordinates[i][0],
          y: coordinates[i][1],
          labelId: labelIds[i],
          confidence: confidences[i],
        });
      }
    }
    if (labeledBatch.length > 0) {
      self.postMessage({
        type: 'cell_batch_update',
        cells: labeledBatch,
      } as CellBatchUpdate);
    }

    mappingResults.output.dispose();
    bufCount = 0;
  }

  // Main per-cell loop
  for (let cellIdx = 0; cellIdx < cellNames.length; cellIdx++) {
    // 1. Read raw counts into rawCell (G-wide, sample-gene order)
    rawCell.fill(0);
    if (isSparse) {
      const [start, end] = indptr!.slice([[cellIdx, cellIdx + 2]]) as number[];
      const values = data.slice([[start, end]]) as number[];
      const valueIndices = indices!.slice([[start, end]]) as number[];
      for (let j = 0; j < valueIndices.length; j++) {
        rawCell[valueIndices[j]] = Number(values[j]);
      }
    } else {
      let sampleExpression: number[];
      if (data.shape.length === 1) {
        sampleExpression = data.slice([[cellIdx * G, (cellIdx + 1) * G]]) as number[];
      } else if (data.shape.length === 2) {
        sampleExpression = data.slice([
          [cellIdx, cellIdx + 1],
          [0, G],
        ]) as number[];
      } else {
        throw new Error('Unsupported data shape');
      }
      for (let j = 0; j < G; j++) rawCell[j] = Number(sampleExpression[j]);
    }

    // 2. Scatter to aligned-gene order (the order of geneTable.symbols)
    for (let j = 0; j < nAligned; j++) {
      alignedCounts[j] = rawCell[geneTable.sampleIndices[j]];
    }

    // 3. log1p + sum-to-1 normalize
    const weights = normalizeCounts(alignedCounts);

    // 4. Weighted sampling with replacement
    const rng = new RNG(cellIdx);
    const sampleIndices = weightedSample(weights, UCE_SAMPLE_SIZE, rng);

    // 5. Chromosome order (uq sorted + shuffled by the same rng)
    const chromOrder = chromOrderFromSample(sampleIndices, geneTable, rng);

    // 6. Build sentence (tokens + attention mask, padded to UCE_PAD_LENGTH)
    const { tokenIds, attentionMask } = buildSentence(
      sampleIndices,
      chromOrder,
      geneTable,
      sentenceTokens,
      UCE_PAD_LENGTH
    );

    // 7. Dynamic seq_len: trim to valid prefix — padding is always at the tail
    const validLen = validPrefixLength(attentionMask);
    const idsSlice = tokenIds.subarray(0, validLen);
    const maskSlice = attentionMask.slice(0, validLen);

    // 8. Gather protein embeddings (1, validLen, embDim)
    const src = gather(proteinEmbeddings, tableRows, embDim, idsSlice, 1, validLen);

    // 9. Run transformer
    const srcTensor = new Tensor('float32', src, [1, validLen, embDim]);
    const maskTensor = new Tensor('float32', maskSlice, [1, validLen]);
    const out = await model!.embeddingSession.run({
      src: srcTensor,
      mask: maskTensor,
    });
    const cellEmb = out.cell_embedding.data as Float32Array;

    // Discover output dim on first run
    if (embeddingDim === -1) {
      embeddingDim = cellEmb.length;
      flushBuffer = new Float32Array(FLUSH * embeddingDim);
      console.log(`UCE-brain cell embedding dim: ${embeddingDim}`);
    }

    // Copy into flush buffer
    flushBuffer!.set(cellEmb, bufCount * embeddingDim);
    bufCellIndices[bufCount] = cellIdx;
    bufCount++;

    out.cell_embedding.dispose();
    if (out.gene_embeddings) out.gene_embeddings.dispose();

    if (bufCount >= FLUSH) {
      await flushBatch();
    }

    // Progress every cell — per-cell time is ~100+ ms so this isn't chatty.
    if ((cellIdx + 1) % 10 === 0 || cellIdx === cellNames.length - 1) {
      self.postMessage({
        type: 'progress',
        message: 'Labeling',
        countFinished: cellIdx + 1,
        totalToProcess: cellNames.length,
      } as ProgressMessage);
    }
  }

  // Final flush
  await flushBatch();
}

/**
 * Emit a cell_batch_update for the current batch. Shared by SCimilarity +
 * (indirectly) the UCE-brain flush path.
 */
function emitBatchUpdates(
  cellNames: string[],
  batchStart: number,
  currentBatchSize: number,
  coordinates: number[][],
  labelIds: number[],
  confidences: number[]
): void {
  const labeledBatch: CellUpdate[] = [];
  for (let i = 0; i < currentBatchSize; i++) {
    const cellIndex = batchStart + i;
    if (labelIds[i] !== -1) {
      labeledBatch.push({
        cellId: cellNames[cellIndex],
        x: coordinates[i][0],
        y: coordinates[i][1],
        labelId: labelIds[i],
        confidence: confidences[i],
      });
    }
  }
  if (labeledBatch.length > 0) {
    self.postMessage({
      type: 'cell_batch_update',
      cells: labeledBatch,
    } as CellBatchUpdate);
  }
}

/**
 * Calculate feature importance for a specific cell
 */
async function calculateFeatureImportance(
  cellId: string,
  cellIndex: number,
  topN: number
) {
  try {
    console.log('Feature importance request received:', { cellId, cellIndex, topN });
    console.log('Model initialized:', !!model);
    console.log('Current raw data initialized:', !!currentRawData);

    if (!model || !currentRawData) {
      throw new Error(
        `Model or data not initialized (model: ${!!model}, data: ${!!currentRawData})`
      );
    }

    if (model.foundationModel !== 'scimilarity') {
      // UCE-brain's sampler-based pipeline doesn't admit SCimilarity's
      // gene-zeroing perturbation cleanly — zeroing a gene changes weighted
      // sampling and chromosome ordering, so each perturbation is as
      // expensive as a fresh inference pass. A separate importance method
      // (e.g., attention-roll-out or gradient-based) is a follow-up spike.
      currentFeatureImportanceCellId = null;
      self.postMessage({
        type: 'error',
        error: `Feature importance is not available for ${model.foundationModel} models.`,
      } as ErrorMessage);
      return;
    }

    console.log(
      `Calculating feature importance for cell ${cellId} (index ${cellIndex})`
    );

    // Extract cell expression data
    const cellExpression = new Float32Array(model.genes.length);
    const { data, indices, indptr, isSparse, inflationIndices } = currentRawData;

    if (isSparse) {
      const [start, end] = indptr!.slice([[cellIndex, cellIndex + 2]]) as number[];
      const values = data.slice([[start, end]]) as number[];
      const valueIndices = indices!.slice([[start, end]]) as number[];

      for (let j = 0; j < valueIndices.length; j++) {
        const sampleIndex = inflationIndices[valueIndices[j]];
        if (sampleIndex !== -1) {
          cellExpression[sampleIndex] = Number(values[j]);
        }
      }
    } else {
      let sampleExpression: number[] | null = null;
      if (data.shape.length === 1) {
        sampleExpression = data.slice([
          [
            cellIndex * currentRawData.sampleGenes.length,
            (cellIndex + 1) * currentRawData.sampleGenes.length,
          ],
        ]) as number[];
      } else if (data.shape.length === 2) {
        sampleExpression = data.slice([
          [cellIndex, cellIndex + 1],
          [0, currentRawData.sampleGenes.length],
        ]) as number[];
      } else {
        throw new Error('Unsupported data shape');
      }

      for (let j = 0; j < inflationIndices.length; j++) {
        const sampleIndex = inflationIndices[j];
        if (sampleIndex !== -1) {
          cellExpression[sampleIndex] = Number(sampleExpression[j]);
        }
      }
    }

    // Get baseline embedding
    const baselineTensor = new Tensor('float32', cellExpression, [
      1,
      model.genes.length,
    ]);
    const baselineResults = await model.embeddingSession.run({ input: baselineTensor });
    const baselineEmbedding = new Float32Array(
      baselineResults.output.data as Float32Array
    );

    // Find expressed genes (above threshold)
    const minExpression = 0.1;
    const expressedGenes: number[] = [];
    for (let i = 0; i < cellExpression.length; i++) {
      if (cellExpression[i] > minExpression) {
        expressedGenes.push(i);
      }
    }

    console.log(
      `Found ${expressedGenes.length} expressed genes out of ${cellExpression.length}`
    );

    // Calculate importance for each expressed gene
    const importances = new Float32Array(model.genes.length);
    const batchSize = 50;

    for (let i = 0; i < expressedGenes.length; i += batchSize) {
      // Check if calculation was cancelled
      if (cancelFeatureImportance) {
        console.log(`Feature importance calculation cancelled for cell ${cellId}`);
        baselineResults.output.dispose();
        currentFeatureImportanceCellId = null;
        return;
      }

      const batchGenes = expressedGenes.slice(
        i,
        Math.min(i + batchSize, expressedGenes.length)
      );

      for (const geneIdx of batchGenes) {
        // Check cancellation frequently
        if (cancelFeatureImportance) {
          console.log(`Feature importance calculation cancelled for cell ${cellId}`);
          baselineResults.output.dispose();
          currentFeatureImportanceCellId = null;
          return;
        }

        // Create perturbed version (zero out gene)
        const perturbed = new Float32Array(cellExpression);
        perturbed[geneIdx] = 0.0;

        // Get embedding for perturbed input
        const perturbedTensor = new Tensor('float32', perturbed, [
          1,
          model.genes.length,
        ]);
        const perturbedResults = await model.embeddingSession.run({
          input: perturbedTensor,
        });
        const perturbedEmbedding = new Float32Array(
          perturbedResults.output.data as Float32Array
        );

        // Calculate L2 distance
        let sum = 0;
        for (let j = 0; j < baselineEmbedding.length; j++) {
          const diff = baselineEmbedding[j] - perturbedEmbedding[j];
          sum += diff * diff;
        }
        importances[geneIdx] = Math.sqrt(sum);

        perturbedResults.output.dispose();
      }

      // Report progress
      const progress = Math.min(100, ((i + batchSize) / expressedGenes.length) * 100);
      self.postMessage({
        type: 'feature_importance_progress',
        cellId,
        progress,
      } as FeatureImportanceProgressMessage);

      // Yield to event loop
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    // Get top N genes
    const results: Array<{
      gene: string;
      importance: number;
      expression: number;
      index: number;
    }> = [];

    for (let i = 0; i < importances.length; i++) {
      if (importances[i] > 0) {
        results.push({
          gene: model.genes[i],
          importance: importances[i],
          expression: cellExpression[i],
          index: i,
        });
      }
    }

    results.sort((a, b) => b.importance - a.importance);
    const topGenes = results.slice(0, topN);

    console.log(`Top ${topN} genes for cell ${cellId}:`, topGenes);

    // Send results
    self.postMessage({
      type: 'feature_importance_result',
      cellId,
      genes: topGenes,
    } as FeatureImportanceResultMessage);

    baselineResults.output.dispose();
    currentFeatureImportanceCellId = null;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('Feature importance error:', errorMessage);
    currentFeatureImportanceCellId = null;
    self.postMessage({
      type: 'error',
      error: `Failed to calculate feature importance: ${errorMessage}`,
    } as ErrorMessage);
  }
}
