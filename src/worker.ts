/**
 * Unified Web Worker for embedding and labeling cells
 * 
 * This worker processes h5ad files, generates embeddings, maps to UMAP coordinates,
 * and labels cells using IVFPQ search in a streamlined pipeline.
 * 
 * Data Flow:
 * h5ad file → embeddings → PUMAP coordinates → IVFPQ labels → App.vue
 */

import h5wasm from 'h5wasm'
import { InferenceSession, Tensor, env } from 'onnxruntime-web'
import { IVFPQ } from '@cytoverse/ivfpq'

// Configuration
const NUM_NEAREST_NEIGHBORS = 50
const NUM_PARTITIONS_TO_SEARCH = 2
const BATCH_SIZE = 16 

// TypeScript interfaces
interface ModelInfo {
  modelID: string
  genes: string[]
  embeddingSession: InferenceSession
  mappingSession: InferenceSession
  ivfpq: IVFPQ
}

interface StartMessage {
  type: 'start'
  modelID: string
  modelsURL: string
  h5File: File
  useWebGPU: boolean
  categoryData: Int32Array
  categoryDataLength: number
}

interface H5DataSet {
  type: string
  value: unknown
  shape: number[]
  keys(): string[]
  slice(ranges: number[][]): unknown
}

interface H5Group {
  type: string
  keys(): string[]
  get(path: string): H5DataSet | H5Group
}

interface H5File {
  keys(): string[]
  get(path: string): H5DataSet | H5Group
  close(): void
}

interface RawCountsData {
  isSparse: boolean
  data: H5DataSet
  indices: H5DataSet | null
  indptr: H5DataSet | null
}

interface CellResult {
  type: 'cell_update'
  cellId: string
  x: number
  y: number
  label?: string
  labelId?: number
  confidence?: number
  trainVectorId?: number
}

interface StatusMessage {
  type: 'status'
  message: string
}

interface ProgressMessage {
  type: 'progress'
  message: string
  countFinished: number
  totalToProcess: number
}

interface ErrorMessage {
  type: 'error'
  error: string
}

interface FinishedMessage {
  type: 'finished'
  datasetLabel: string
  totalProcessed: number
}

// Global state
let model: ModelInfo | null = null
let categoryData: Int32Array | null = null
let categoryDataLength = 0

// Handle messages from main thread
self.addEventListener('message', async function (event: MessageEvent<StartMessage>) {
  if (event.data.type === 'start') {
    start(
      event.data.modelID,
      event.data.modelsURL,
      event.data.h5File,
      event.data.useWebGPU,
      event.data.categoryData,
      event.data.categoryDataLength
    )
  }
})

/**
 * Initialize model and IVFPQ components
 */
async function instantiateModel(
  modelsURL: string,
  modelID: string,
  useWebGPU: boolean
): Promise<ModelInfo> {
  console.log(`Instantiating model ${modelID} from ${modelsURL}`)
  self.postMessage({ type: 'status', message: 'Downloading model...' } as StatusMessage)

  // Fetch model genes
  let response = await fetch(`${modelsURL}/${modelID}/embedding/genes.txt`)
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  const genes = (await response.text()).split('\n')
  console.log('Model Genes', genes.slice(0, 5))

  // Fetch embedding model
  response = await fetch(`${modelsURL}/${modelID}/embedding/model.onnx`)
  if (!response.ok) {
    throw new Error(`Error fetching onnx file: ${response.status}`)
  }
  const contentLength = response.headers.get('content-length')
  if (!contentLength) {
    throw new Error('Content-Length header is missing')
  }
  const totalBytes = parseInt(contentLength, 10)
  let loadedBytes = 0

  const reader = response.body!.getReader()
  const chunks: Uint8Array[] = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loadedBytes += value.length

    self.postMessage({
      type: 'progress',
      message: 'Downloading model...',
      countFinished: loadedBytes,
      totalToProcess: totalBytes,
    } as ProgressMessage)
  }

  const modelArray = new Uint8Array(loadedBytes)
  let position = 0
  for (const chunk of chunks) {
    modelArray.set(chunk, position)
    position += chunk.length
  }

  // Configure ONNX Runtime
  self.postMessage({ type: 'status', message: 'Instantiating model...' } as StatusMessage)
  env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency - 1)
  env.wasm.proxy = true

  let sessionOptions = {}
  if (useWebGPU) {
    console.log('Configuring ONNX Runtime to use WebGPU...')
    sessionOptions = {
      executionProviders: [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance',
        },
        'wasm',
      ],
      graphOptimizationLevel: 'all',
    }
  } else {
    console.log('Configuring ONNX Runtime to use WebAssembly...')
    sessionOptions = {
      executionProviders: ['wasm'],
      executionMode: 'parallel',
      graphOptimizationLevel: 'all',
    }
  }

  // Create inference sessions
  const embeddingSession = await InferenceSession.create(modelArray.buffer, sessionOptions)
  console.log('Model Output names', embeddingSession.outputNames)

  const mappingSession = await InferenceSession.create(
    `${modelsURL}/${modelID}/pumap/model.onnx`,
    sessionOptions
  )
  console.log('Mapper Output names', mappingSession.outputNames)

  // Load IVFPQ system
  const ivfpqBasePath = `${modelsURL}/${modelID}/ivfpq`
  const ivfpq = new IVFPQ(ivfpqBasePath)
  await ivfpq.load()
  console.log('IVFPQ system loaded successfully')

  return { modelID, genes, embeddingSession, mappingSession, ivfpq }
}

/**
 * Extract cell names from h5ad file
 */
function getCellNames(annData: H5File): string[] {
  if (!annData.keys().includes('obs')) {
    throw new Error('Unable to find cell names: Missing "obs" group in h5ad file')
  }

  const obs = annData.get('obs')

  if (obs.type === 'Dataset') {
    try {
      const obsValue = (obs as H5DataSet).value as Array<[string]>
      return obsValue.map((e) => e[0])
    } catch {
      throw new Error('Unable to extract cell names from obs Dataset')
    }
  }

  if (obs.type === 'Group') {
    const obsGroup = obs as H5Group
    const obsKeys = obsGroup.keys()
    const indexKeys = ['index', '_index', 'barcodes', '_barcodes', 'cell_id', 'cell_name']

    for (const key of indexKeys) {
      if (obsKeys.includes(key)) {
        try {
          return (obsGroup.get(key) as H5DataSet).value as string[]
        } catch {
          console.warn(`Failed to read cell names from obs/${key}`)
        }
      }
    }

    throw new Error(`Unable to find cell names in obs group. Available keys: ${obsKeys.join(', ')}`)
  }

  throw new Error('Unable to find cell names: obs is neither Dataset nor Group')
}

/**
 * Validate gene symbols
 */
function validateGeneSymbols(geneArray: string[]): boolean {
  const knownGenes = ['TP53', 'BRCA1']
  return knownGenes.some((gene) => geneArray.includes(gene))
}

/**
 * Extract gene names from h5ad file
 */
function getSampleGenes(annData: H5File): string[] {
  if (!annData.keys().includes('var')) {
    throw new Error('Unable to find gene names: Missing "var" group in h5ad file')
  }

  const varData = annData.get('var')

  if (varData.type === 'Dataset') {
    try {
      const varValue = (varData as H5DataSet).value as Array<[string]>
      return varValue.map((e) => e[0])
    } catch {
      throw new Error('Unable to extract gene names from var Dataset')
    }
  }

  if (varData.type === 'Group') {
    const varGroup = varData as H5Group
    const varKeys = varGroup.keys()
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
    ]

    for (const key of geneKeys) {
      if (varKeys.includes(key)) {
        try {
          const genes = (varGroup.get(key) as H5DataSet).value as string[]
          if (validateGeneSymbols(genes)) {
            console.log(`Found gene names in var/${key}`)
            return genes
          }
        } catch {
          console.warn(`Failed to read gene names from var/${key}`)
        }
      }
    }

    if (varKeys.length > 0) {
      try {
        const firstKey = varKeys[0]
        const potentialGenes = (varGroup.get(firstKey) as H5DataSet).value as string[]
        if (Array.isArray(potentialGenes) && validateGeneSymbols(potentialGenes)) {
          console.warn('Falling back to using var index as gene names')
          return potentialGenes
        }
      } catch {
        console.warn('Failed to read potential gene names from first available var key')
      }
    }

    throw new Error(`Unable to find gene names in var group. Available keys: ${varKeys.join(', ')}`)
  }

  throw new Error('Unable to find gene names: var is neither Dataset nor Group')
}

/**
 * Extract raw counts from h5ad file
 */
function getRawCounts(annData: H5File): RawCountsData {
  const topLevelKeys = annData.keys()

  // Check layers for raw counts
  if (topLevelKeys.includes('layers')) {
    const layers = annData.get('layers')
    if (layers.type === 'Group') {
      const layersGroup = layers as H5Group
      const layerKeys = layersGroup.keys()
      const rawCountKeys = ['counts', 'raw_counts', 'raw', 'count', 'spliced', 'unspliced']

      for (const key of rawCountKeys) {
        if (layerKeys.includes(key)) {
          console.log(`Found raw counts in layers/${key}`)
          const countsData = layersGroup.get(key)

          if (countsData.type === 'Dataset') {
            return {
              isSparse: false,
              data: countsData as H5DataSet,
              indices: null,
              indptr: null,
            }
          }

          if (countsData.type === 'Group') {
            const sparseGroup = countsData as H5Group
            const sparseKeys = sparseGroup.keys()

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
              }
            }
          }
        }
      }
    }
  }

  // Check raw attribute
  if (topLevelKeys.includes('raw')) {
    const raw = annData.get('raw')
    if (raw.type === 'Group') {
      const rawGroup = raw as H5Group
      if (rawGroup.keys().includes('X')) {
        const rawX = rawGroup.get('X')

        if (rawX.type === 'Dataset') {
          return {
            isSparse: false,
            data: rawX as H5DataSet,
            indices: null,
            indptr: null,
          }
        }

        if (rawX.type === 'Group') {
          const sparseGroup = rawX as H5Group
          const sparseKeys = sparseGroup.keys()

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
            }
          }
        }
      }
    }
  }

  // Use main X matrix as last resort
  if (topLevelKeys.includes('X')) {
    console.warn('Using main X matrix - this may contain normalized data instead of raw counts')
    const X = annData.get('X')

    if (X.type === 'Dataset') {
      return {
        isSparse: false,
        data: X as H5DataSet,
        indices: null,
        indptr: null,
      }
    }

    if (X.type === 'Group') {
      const sparseGroup = X as H5Group
      const sparseKeys = sparseGroup.keys()

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
        }
      }
    }
  }

  const availableKeys = topLevelKeys.join(', ')
  throw new Error(
    `Unable to find raw counts data. Looked in: layers/counts, layers/raw_counts, raw/X, and X. ` +
      `Available top-level keys in h5ad file: ${availableKeys}`
  )
}

/**
 * Precompute inflation indices for gene mapping
 */
function precomputeInflationIndices(currentModelGenes: string[], sampleGenes: string[]): number[] {
  const inflationIndices: number[] = []
  for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
    inflationIndices.push(currentModelGenes.indexOf(sampleGenes[geneIndex]))
  }
  const missingGenesInModel = inflationIndices.filter((x) => x === -1).length
  console.log(`Missing genes in model: ${missingGenesInModel}`)
  return inflationIndices
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
    const cellIndex = batchStart + batchSlot
    const batchOffset = batchSlot * model!.genes.length

    if (isSparse) {
      const [start, end] = indptr!.slice([[cellIndex, cellIndex + 2]])
      const values = data.slice([[start, end]])
      const valueIndices = indices!.slice([[start, end]])

      for (let j = 0; j < valueIndices.length; j++) {
        const sampleIndex = inflationIndices[valueIndices[j]]
        if (sampleIndex !== -1) {
          inflatedBatchData[batchOffset + sampleIndex] = Number(values[j])
        }
      }
    } else {
      let sampleExpression: number[] | null = null
      if (data.shape.length === 1) {
        sampleExpression = data.slice([
          [cellIndex * sampleGenes.length, (cellIndex + 1) * sampleGenes.length],
        ]) as number[]
      } else if (data.shape.length === 2) {
        sampleExpression = data.slice([
          [cellIndex, cellIndex + 1],
          [0, sampleGenes.length],
        ]) as number[]
      } else {
        throw new Error('Unsupported data shape')
      }
      for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
        const sampleIndex = inflationIndices[geneIndex]
        if (sampleIndex !== -1) {
          inflatedBatchData[batchOffset + sampleIndex] = sampleExpression[geneIndex]
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
  embeddingDim: number
): Promise<{ labelIds: number[], confidences: number[], trainVectorIds: number[] }> {
  const labelIds: number[] = []
  const confidences: number[] = []
  const trainVectorIds: number[] = []
  
  for (let i = 0; i < batchSize; i++) {
    try {
      // Extract embedding for this cell
      const queryVector = new Float32Array(embeddingDim)
      const offset = i * embeddingDim
      for (let j = 0; j < embeddingDim; j++) {
        queryVector[j] = embeddings[offset + j]
      }

      // Search using IVFPQ
      const searchResults = await model!.ivfpq.search(queryVector, {
        n_probe: NUM_PARTITIONS_TO_SEARCH,
        k: NUM_NEAREST_NEIGHBORS
      })

      const nearestTrainVectorId = searchResults.indices.length > 0 ? searchResults.indices[0] : -1

      // Compute consensus label
      let consensusLabelId = -1
      let consensusConfidence = 0

      if (searchResults.indices.length > 0 && categoryData) {
        const labelVotes: { [labelId: number]: number } = {}

        for (const trainVectorId of searchResults.indices) {
          if (trainVectorId !== -1 && trainVectorId < categoryDataLength) {
            const labelId = categoryData[trainVectorId]
            if (labelId >= 0) {
              labelVotes[labelId] = (labelVotes[labelId] || 0) + 1
            }
          }
        }

        let maxVotes = 0
        for (const [labelId, votes] of Object.entries(labelVotes)) {
          if (votes > maxVotes) {
            maxVotes = votes
            consensusLabelId = parseInt(labelId)
          }
        }

        if (maxVotes > 0) {
          consensusConfidence = maxVotes / NUM_NEAREST_NEIGHBORS
        }
      }

      trainVectorIds.push(nearestTrainVectorId)
      labelIds.push(consensusLabelId)
      confidences.push(consensusConfidence)
    } catch (error) {
      console.error(`Error processing vector ${i}:`, error)
      trainVectorIds.push(-1)
      labelIds.push(-1)
      confidences.push(0)
    }
  }

  return { labelIds, confidences, trainVectorIds }
}

/**
 * Main processing function
 */
async function start(
  modelID: string,
  modelsURL: string,
  h5File: File,
  useWebGPU: boolean,
  categoryDataIn: Int32Array,
  categoryDataLengthIn: number
): Promise<void> {
  console.log(`Starting unified worker for model ${modelID} with file ${h5File.name}`)
  
  // Store category data
  categoryData = categoryDataIn
  categoryDataLength = categoryDataLengthIn
  
  self.postMessage({ type: 'status', message: 'Loading libraries...' } as StatusMessage)
  const Module = await h5wasm.ready
  const { FS } = Module
  console.log('h5wasm loaded')

  try {
    // Load model if needed
    if (!model || model.modelID !== modelID) {
      model = await instantiateModel(modelsURL, modelID, useWebGPU)
    }

    // Mount h5 file
    self.postMessage({ type: 'status', message: 'Loading file...' } as StatusMessage)
    if (!FS.analyzePath('/work').exists) {
      FS.mkdir('/work')
    }
    FS.mount(FS.filesystems.WORKERFS, { files: [h5File] }, '/work')

    const annData = new h5wasm.File(`/work/${h5File.name}`, 'r') as H5File
    console.log(`Top level keys: ${annData.keys()}`)

    // Extract metadata
    let cellNames: string[]
    try {
      cellNames = getCellNames(annData)
      console.log(`Found ${cellNames.length} cells`)
    } catch (error) {
      throw new Error(
        `Failed to extract cell names: ${error instanceof Error ? error.message : String(error)}`
      )
    }

    let sampleGenes: string[]
    try {
      sampleGenes = getSampleGenes(annData)
      console.log(`Found ${sampleGenes.length} genes`)
    } catch (error) {
      throw new Error(
        `Failed to extract gene names: ${error instanceof Error ? error.message : String(error)}`
      )
    }

    // Extract raw counts
    let rawCountsData: RawCountsData
    try {
      rawCountsData = getRawCounts(annData)
      console.log(`Found expression data (sparse: ${rawCountsData.isSparse})`)
    } catch (error) {
      throw new Error(
        `Failed to extract expression data: ${
          error instanceof Error ? error.message : String(error)
        }`
      )
    }

    const { isSparse, data, indices, indptr } = rawCountsData
    const inflationIndices = precomputeInflationIndices(model.genes, sampleGenes)

    self.postMessage({ type: 'status', message: 'Processing cells...' } as StatusMessage)

    // Process batches
    for (let batchStart = 0; batchStart < cellNames.length; batchStart += BATCH_SIZE) {
      const batchEnd = Math.min(batchStart + BATCH_SIZE, cellNames.length)
      const currentBatchSize = batchEnd - batchStart
      
      // Prepare batch data
      const batchData = new Float32Array(currentBatchSize * model.genes.length)
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
      )

      // Run embedding model
      const inputTensor = new Tensor('float32', batchData, [
        currentBatchSize,
        model.genes.length,
      ])
      
      const embeddingResults = await model.embeddingSession.run({
        input: inputTensor,
      })

      // Generate UMAP coordinates
      const mappingResults = await model.mappingSession.run({
        input: embeddingResults.output,
      })

      // Parse coordinates
      const coordinates: number[][] = []
      for (let i = 0; i < mappingResults.output.dims[0]; i++) {
        const startIndex = i * 2
        coordinates.push([
          Number(mappingResults.output.data[startIndex]),
          Number(mappingResults.output.data[startIndex + 1]),
        ])
      }

      // First send unlabeled coordinates for immediate display
      for (let i = 0; i < currentBatchSize; i++) {
        const cellIndex = batchStart + i
        self.postMessage({
          type: 'cell_update',
          cellId: cellNames[cellIndex],
          x: coordinates[i][0],
          y: coordinates[i][1],
        } as CellResult)
      }

      // Label cells using IVFPQ
      const embeddingDim = embeddingResults.output.dims[1] as number
      const { labelIds, confidences, trainVectorIds } = await labelCells(
        embeddingResults.output.data as Float32Array, 
        currentBatchSize, 
        embeddingDim
      )

      // Send labeled updates
      for (let i = 0; i < currentBatchSize; i++) {
        const cellIndex = batchStart + i
        if (labelIds[i] !== -1) {
          self.postMessage({
            type: 'cell_update',
            cellId: cellNames[cellIndex],
            x: coordinates[i][0],
            y: coordinates[i][1],
            labelId: labelIds[i],
            confidence: confidences[i],
            trainVectorId: trainVectorIds[i],
          } as CellResult)
        }
      }

      // Send progress
      self.postMessage({
        type: 'progress',
        countFinished: batchEnd,
        totalToProcess: cellNames.length,
      } as ProgressMessage)

      // Clean up tensors
      embeddingResults.output.dispose()
      mappingResults.output.dispose()
    }

    // Clean up
    annData.close()
    FS.unmount('/work')

    self.postMessage({
      type: 'finished',
      datasetLabel: h5File.name,
      totalProcessed: cellNames.length,
    } as FinishedMessage)
  } catch (error) {
    try {
      FS.unmount('/work')
    } catch (unmountError) {
      console.error('Failed to unmount file system:', unmountError)
    }

    const errorMessage = error instanceof Error ? error.message : String(error)
    console.error('Worker error:', errorMessage)
    self.postMessage({
      type: 'error',
      error: errorMessage,
    } as ErrorMessage)
  }
}