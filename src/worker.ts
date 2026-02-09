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

env.logLevel = 'verbose';
env.debug = true;

// Configuration
const NUM_NEAREST_NEIGHBORS = 50;
const NUM_PARTITIONS_TO_SEARCH = 4;
const BATCH_SIZE = 32;

// TypeScript interfaces
interface ModelInfo {
  modelID: string;
  genes: string[];
  embeddingSession: InferenceSession;
  mappingSession: InferenceSession;
  ivfpq: IVFPQ | UserIVFPQ;
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

  // Fetch model genes
  let response = await fetch(`${modelsURL}/${modelID}/embedding/genes.txt`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const genes = (await response.text()).split('\n');
  console.log('Model Genes', genes.slice(0, 5));

  // Fetch embedding model
  response = await fetch(`${modelsURL}/${modelID}/embedding/model.onnx`);
  if (!response.ok) {
    throw new Error(`Error fetching onnx file: ${response.status}`);
  }

  // Get total size from Content-Length header if available
  const contentLength = response.headers.get('content-length');
  const totalBytes = contentLength ? parseInt(contentLength, 10) : null;

  // Collect chunks dynamically (Content-Length may not be available in all servers)
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

  // Concatenate all chunks into a single array
  const modelArray = new Uint8Array(receivedBytes);
  let position = 0;
  for (const chunk of chunks) {
    modelArray.set(chunk, position);
    position += chunk.length;
  }

  // Configure ONNX Runtime
  self.postMessage({
    type: 'status',
    message: 'Instantiating model...',
  } as StatusMessage);
  env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 4);
  env.wasm.proxy = true;

  // Create inference sessions
  let sessionOptions = {};
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

  // Create mapping session
  if (useWebGPU) {
    console.log('Configuring mapping session to use WebGPU...');
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
    console.log('Configuring mapping session to use WebAssembly...');
    sessionOptions = {
      executionProviders: ['wasm'],
      executionMode: 'parallel',
      graphOptimizationLevel: 'all',
    };
  }

  const mappingSession = await InferenceSession.create(
    `${modelsURL}/${modelID}/pumap/model.onnx`,
    sessionOptions
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

  return { modelID, genes, embeddingSession, mappingSession, ivfpq };
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
  const knownGenes = ['TP53', 'BRCA1'];
  return knownGenes.some((gene) => geneArray.includes(gene));
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
      return varValue.map((e) => e[0]);
    } catch {
      throw new Error('Unable to extract gene names from var Dataset');
    }
  }

  if (varData.type === 'Group') {
    const varGroup = varData as H5Group;
    const varKeys = varGroup.keys();
    const geneKeys = [
      'symbol',
      'gene_symbol',
      'gene_symbols',
      'gene_name',
      'gene_names',
      'feature_name',
      'features',
      'index',
      '_index',
      'gene_id',
      'gene_ids',
    ];

    for (const key of geneKeys) {
      if (varKeys.includes(key)) {
        try {
          const genes = (varGroup.get(key) as H5DataSet).value as string[];
          if (validateGeneSymbols(genes)) {
            console.log(`Found gene names in var/${key}`);
            return genes;
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
          return potentialGenes;
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
    const inflationIndices = precomputeInflationIndices(model.genes, sampleGenes);

    // Store data globally for feature importance
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

    // Reusable batch data buffer (max size)
    const batchDataBuffer = new Float32Array(BATCH_SIZE * model.genes.length);

    // Process batches
    for (let batchStart = 0; batchStart < cellNames.length; batchStart += BATCH_SIZE) {
      // console.log(`Processing batch starting at cell ${batchStart}`);
      const batchEnd = Math.min(batchStart + BATCH_SIZE, cellNames.length);
      const currentBatchSize = batchEnd - batchStart;

      // Prepare batch data (reuse buffer, zero only what's needed)
      const batchData =
        currentBatchSize === BATCH_SIZE
          ? batchDataBuffer
          : batchDataBuffer.subarray(0, currentBatchSize * model.genes.length);
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

      // Run embedding model
      const inputTensor = new Tensor('float32', batchData, [
        currentBatchSize,
        model.genes.length,
      ]);

      const embeddingResults = await model.embeddingSession.run({
        input: inputTensor,
      });

      // Generate UMAP coordinates
      const mappingResults = await model.mappingSession.run({
        input: embeddingResults.output,
      });

      // Parse coordinates
      const coordinates: number[][] = [];
      for (let i = 0; i < mappingResults.output.dims[0]; i++) {
        const startIndex = i * 2;
        coordinates.push([
          Number(mappingResults.output.data[startIndex]),
          Number(mappingResults.output.data[startIndex + 1]),
        ]);
      }

      // Label cells using IVFPQ
      const embeddingDim = embeddingResults.output.dims[1] as number;
      const { labelIds, confidences, partitionIds, pqCodes } = await labelCells(
        embeddingResults.output.data as Float32Array,
        currentBatchSize,
        embeddingDim,
        labelIndices,
        useUserIndex
      );

      // Retain artifacts for user index
      allPartitionIds.set(partitionIds, batchStart);
      for (let i = 0; i < currentBatchSize; i++) {
        allPQCodes[batchStart + i] = pqCodes[i];
      }

      // Send labeled updates as batch
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
        // console.log(`Sending batch of ${labeledBatch.length} labeled cells`);
        self.postMessage({
          type: 'cell_batch_update',
          cells: labeledBatch,
        } as CellBatchUpdate);
      }

      // Send progress
      self.postMessage({
        type: 'progress',
        message: 'Labeling',
        countFinished: batchEnd,
        totalToProcess: cellNames.length,
      } as ProgressMessage);

      // Clean up tensors
      embeddingResults.output.dispose();
      mappingResults.output.dispose();
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
