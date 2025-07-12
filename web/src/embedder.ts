/**
 * Browser Web Worker that runs embedding and parametric pumap models
 *
 * The h5 file is read using h5wasm which is a WebAssembly of the h5 library.
 * We utilize its ability to map the file into the browsers file system and
 * thereby read the gene expression data incrementally to support unlimited file
 * sizes. Note this requires the expression data to be stored in column-major
 * order - which most h5ad files are. The worker can read sparse matrices and
 * expand them as well as map into the model's gene space in a single operation.
 * The prediction is run in multiple threads fed by filling double buffers from
 * the h5 file in a separate thread towards keeping all the threads busy.
 */
import h5wasm from 'h5wasm'

import { InferenceSession, Tensor, env } from 'onnxruntime-web'
import { loadPQModel, ProductQuantizer } from './ivfpq/pq'

// Define TypeScript interfaces for the worker's data structures
interface ModelInfo {
  modelID: string
  genes: string[]
  embeddingSession: InferenceSession
  mappingSession: InferenceSession
  pqModel: ProductQuantizer
}

interface Buffer {
  size: number
  data: Float32Array
}

interface StartMessage {
  type: 'start'
  modelID: string
  modelsURL: string
  h5File: File
  cellRangePercent: number
  useWebGPU: boolean
}

interface H5DataSet {
  type: string
  value: any
  shape: number[]
  keys(): string[]
  slice(ranges: any[][]): any
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

interface EmbeddingOutput {
  output: Tensor
}

interface MappingOutput {
  output: Tensor
}

// Dictionary with various model information (id, genes, session)
let model = null as ModelInfo | null

// Number of threads to use for inference. Use all but one for the GUI to run in
const numThreads = navigator.hardwareConcurrency - 1

// Tuned batch size - if I/O, pre-processing and inflation is fast relative to
// the model inference then increase the batch size so that inference is never
// waiting on data. If the model is very large and inference is slow then
// reduce the batch size so that inference can be parallelized across more
// threads. The ONNX model supports variable size batches which plays into
// this as well.
const batchSize = 2 * numThreads

console.log(`Number of threads: ${numThreads}`)
console.log(`Batch size: ${batchSize}`)

// Handle messages from the main thread
self.addEventListener('message', async function (event: MessageEvent<StartMessage>) {
  if (event.data.type === 'start') {
    start(
      event.data.modelID,
      event.data.modelsURL,
      event.data.h5File,
      event.data.cellRangePercent,
      event.data.useWebGPU
    )
  }
})

/**
 * Create an ONNX Runtime session for the selected model
 * @param {string} modelsURL - The URL of the model
 * @param {string} modelID - The id of the model to load
 * @returns {Promise<ModelInfo>} - A promise that resolves to a model session dictionary
 */
async function instantiateModel(
  modelsURL: string,
  modelID: string,
  useWebGPU: boolean
): Promise<ModelInfo> {
  console.log(`Instantiating model ${modelID} from ${modelsURL}`)
  self.postMessage({ type: 'status', message: 'Downloading model...' })

  // Fetch the model gene list
  // REMIND: Switch to .gz and have browser de-compress
  let response = await fetch(`${modelsURL}/${modelID}/embedding/genes.txt`)
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  const genes = (await response.text()).split('\n')
  console.log('Model Genes', genes.slice(0, 5))

  // Fetch the model ONNX file incrementally to show progress
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

  // Read the response body as a stream
  const reader = response.body!.getReader()
  const chunks: Uint8Array[] = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loadedBytes += value.length

    // Send progress update to the main thread
    self.postMessage({
      type: 'progress',
      message: 'Downloading model...',
      countFinished: loadedBytes,
      totalToProcess: totalBytes,
    })
  }

  // Combine all chunks into a single ArrayBuffer
  const modelArray = new Uint8Array(loadedBytes)
  let position = 0
  for (const chunk of chunks) {
    modelArray.set(chunk, position)
    position += chunk.length
  }

  // Initialize ONNX Runtime environment
  self.postMessage({ type: 'status', message: 'Instantiating model...' })
  // See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
  env.wasm.numThreads = numThreads
  env.wasm.proxy = true

  // Configure execution providers based on WebGPU availability
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
      // executionMode: 'parallel',
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

  // Create the InferenceSession with the model ArrayBuffer we fetched incrementally
  const embeddingSession = await InferenceSession.create(modelArray.buffer, sessionOptions)
  console.log('Model Output names', embeddingSession.outputNames)

  // Create the MappingSession
  const mappingSession = await InferenceSession.create(
    `${modelsURL}/${modelID}/pumap/model.onnx`,
    sessionOptions
  )
  console.log('Mapper Output names', mappingSession.outputNames)

  // Load the PQ model for encoding embeddings
  const pqModel = await loadPQModel(`${modelsURL}/${modelID}/pq`)
  console.log('PQ Model loaded successfully')

  return { modelID, genes, embeddingSession, mappingSession, pqModel }
}

/*
 * Precompute the inflation indices for the sample gene list
 * @param {string[]} currentModelGenes - The gene list of the model
 * @param {string[]} sampleGenes - The gene list of the sample
 * @returns {number[]} - The inflation indices
 * These are used in fillBatchData inflate each sample from sample gene list space
 * into the model's gene list space
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
 * Extract cell names (barcodes) from an h5ad file
 * @param {H5File} annData - The h5ad file
 * @returns {string[]} - Array of cell names
 * @throws {Error} - If cell names cannot be found
 */
function getCellNames(annData: H5File): string[] {
  // Check if obs exists
  if (!annData.keys().includes('obs')) {
    throw new Error('Unable to find cell names: Missing "obs" group in h5ad file')
  }

  const obs = annData.get('obs')
  
  // Case 1: obs is a Dataset (structured array)
  if (obs.type === 'Dataset') {
    try {
      return (obs as H5DataSet).value.map((e: any) => e[0])
    } catch (e) {
      throw new Error('Unable to extract cell names from obs Dataset')
    }
  }
  
  // Case 2: obs is a Group
  if (obs.type === 'Group') {
    const obsGroup = obs as H5Group
    const obsKeys = obsGroup.keys()
    
    // Try common locations for cell names/barcodes
    const indexKeys = ['index', '_index', 'barcodes', '_barcodes', 'cell_id', 'cell_name']
    
    for (const key of indexKeys) {
      if (obsKeys.includes(key)) {
        try {
          return (obsGroup.get(key) as H5DataSet).value
        } catch (e) {
          console.warn(`Failed to read cell names from obs/${key}`)
        }
      }
    }
    
    throw new Error(`Unable to find cell names in obs group. Available keys: ${obsKeys.join(', ')}`)
  }
  
  throw new Error('Unable to find cell names: obs is neither Dataset nor Group')
}

/**
 * Validate if an array contains gene symbols by checking for known genes
 * @param {string[]} geneArray - Array of potential gene symbols
 * @returns {boolean} - True if array appears to contain gene symbols
 */
function validateGeneSymbols(geneArray: string[]): boolean {
  const knownGenes = ['TP53', 'BRCA1']
  return knownGenes.some(gene => geneArray.includes(gene))
}

/**
 * Extract gene names/symbols from an h5ad file
 * @param {H5File} annData - The h5ad file
 * @returns {string[]} - Array of gene names
 * @throws {Error} - If gene names cannot be found
 */
function getSampleGenes(annData: H5File): string[] {
  // Check if var exists
  if (!annData.keys().includes('var')) {
    throw new Error('Unable to find gene names: Missing "var" group in h5ad file')
  }

  const varData = annData.get('var')
  
  // Case 1: var is a Dataset (structured array)
  if (varData.type === 'Dataset') {
    try {
      return (varData as H5DataSet).value.map((e: any) => e[0])
    } catch (e) {
      throw new Error('Unable to extract gene names from var Dataset')
    }
  }
  
  // Case 2: var is a Group
  if (varData.type === 'Group') {
    const varGroup = varData as H5Group
    const varKeys = varGroup.keys()
    
    // Try common locations for gene names in order of preference
    // symbol/gene_symbol is preferred over index as it contains gene names rather than IDs
    const geneKeys = [
      'symbol', 'gene_symbol', 'gene_symbols', 'gene_name', 'gene_names',
      'feature_name', 'features', 'index', '_index', 'gene_id', 'gene_ids'
    ]
    
    for (const key of geneKeys) {
      if (varKeys.includes(key)) {
        try {
          const genes = (varGroup.get(key) as H5DataSet).value
          if (validateGeneSymbols(genes)) {
            return genes
          }
        } catch (e) {
          console.warn(`Failed to read gene names from var/${key}`)
        }
      }
    }
    
    // Fallback: Check if there's an index in the var group that might contain gene names
    if (varKeys.length > 0) {
      try {
        // Try to get the first available key as a potential gene name source
        const firstKey = varKeys[0]
        const potentialGenes = (varGroup.get(firstKey) as H5DataSet).value
        if (Array.isArray(potentialGenes) && validateGeneSymbols(potentialGenes)) {
          return potentialGenes
        }
      } catch (e) {
        console.warn('Failed to read potential gene names from first available var key')
      }
    }
    
    throw new Error(`Unable to find gene names in var group. Available keys: ${varKeys.join(', ')}`)
  }
  
  throw new Error('Unable to find gene names: var is neither Dataset nor Group')
}

/**
 * Extract raw counts expression data from an h5ad file
 * @param {H5File} annData - The h5ad file
 * @returns {RawCountsData} - Object containing expression data and metadata
 * @throws {Error} - If raw counts cannot be found
 */
function getRawCounts(annData: H5File): RawCountsData {
  const topLevelKeys = annData.keys()
  
  // Strategy 1: Check layers for raw counts (most common location)
  if (topLevelKeys.includes('layers')) {
    const layers = annData.get('layers')
    if (layers.type === 'Group') {
      const layersGroup = layers as H5Group
      const layerKeys = layersGroup.keys()
      
      // Try common raw count layer names
      const rawCountKeys = ['counts', 'raw_counts', 'raw', 'count', 'spliced', 'unspliced']
      
      for (const key of rawCountKeys) {
        if (layerKeys.includes(key)) {
          const countsData = layersGroup.get(key)
          
          // Handle dense matrix
          if (countsData.type === 'Dataset') {
            return {
              isSparse: false,
              data: countsData as H5DataSet,
              indices: null,
              indptr: null
            }
          }
          
          // Handle sparse matrix
          if (countsData.type === 'Group') {
            const sparseGroup = countsData as H5Group
            const sparseKeys = sparseGroup.keys()
            
            if (sparseKeys.includes('data') && sparseKeys.includes('indices') && sparseKeys.includes('indptr')) {
              return {
                isSparse: true,
                data: sparseGroup.get('data') as H5DataSet,
                indices: sparseGroup.get('indices') as H5DataSet,
                indptr: sparseGroup.get('indptr') as H5DataSet
              }
            }
          }
        }
      }
    }
  }
  
  // Strategy 2: Check raw attribute (older format)
  if (topLevelKeys.includes('raw')) {
    const raw = annData.get('raw')
    if (raw.type === 'Group') {
      const rawGroup = raw as H5Group
      if (rawGroup.keys().includes('X')) {
        const rawX = rawGroup.get('X')
        
        // Handle dense matrix
        if (rawX.type === 'Dataset') {
          return {
            isSparse: false,
            data: rawX as H5DataSet,
            indices: null,
            indptr: null
          }
        }
        
        // Handle sparse matrix
        if (rawX.type === 'Group') {
          const sparseGroup = rawX as H5Group
          const sparseKeys = sparseGroup.keys()
          
          if (sparseKeys.includes('data') && sparseKeys.includes('indices') && sparseKeys.includes('indptr')) {
            return {
              isSparse: true,
              data: sparseGroup.get('data') as H5DataSet,
              indices: sparseGroup.get('indices') as H5DataSet,
              indptr: sparseGroup.get('indptr') as H5DataSet
            }
          }
        }
      }
    }
  }
  
  // Strategy 3: Use main X matrix as last resort (might be normalized)
  if (topLevelKeys.includes('X')) {
    console.warn('Using main X matrix - this may contain normalized data instead of raw counts')
    const X = annData.get('X')
    
    // Handle dense matrix
    if (X.type === 'Dataset') {
      return {
        isSparse: false,
        data: X as H5DataSet,
        indices: null,
        indptr: null
      }
    }
    
    // Handle sparse matrix
    if (X.type === 'Group') {
      const sparseGroup = X as H5Group
      const sparseKeys = sparseGroup.keys()
      
      if (sparseKeys.includes('data') && sparseKeys.includes('indices') && sparseKeys.includes('indptr')) {
        return {
          isSparse: true,
          data: sparseGroup.get('data') as H5DataSet,
          indices: sparseGroup.get('indices') as H5DataSet,
          indptr: sparseGroup.get('indptr') as H5DataSet
        }
      }
    }
  }
  
  // If we get here, we couldn't find any expression data
  const availableKeys = topLevelKeys.join(', ')
  throw new Error(
    `Unable to find raw counts data. Looked in: layers/counts, layers/raw_counts, raw/X, and X. ` +
    `Available top-level keys in h5ad file: ${availableKeys}`
  )
}

/**
 * Fill the batch data and inflate it into the model's gene list space
 * @param {number} batchStart - The start index of the batch
 * @param {number} currentBatchSize - The size of the batch
 * @param {H5DataSet} data - The data array
 * @param {H5DataSet} indices - The indices array
 * @param {H5DataSet} indptr - The indptr array
 * @param {boolean} isSparse - Whether the data is sparse
 * @param {string[]} sampleGenes - The gene list of the sample
 * @param {number[]} inflationIndices - The inflation indices
 * @param {Float32Array} inflatedBatchData - The inflated batch data
 * This function fills the batch data and inflates it into the model's gene list space
 * in one step. It also handles both sparse and non-sparse data.
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
  // Fill batchData and inflate in one step
  for (let batchSlot = 0; batchSlot < currentBatchSize; batchSlot++) {
    const cellIndex = batchStart + batchSlot
    const batchOffset = batchSlot * model!.genes.length

    if (isSparse) {
      // Sparse data stored column major
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
      // Non-sparse stored column major
      // Load up an intermediate buffer with h5wasm slice so we don't
      // call into h5wasm for every value
      let sampleExpression: any = null
      if (data.shape.length === 1) {
        // Direct 1D dense array mapping
        sampleExpression = data.slice([
          [cellIndex * sampleGenes.length, (cellIndex + 1) * sampleGenes.length],
        ])
      } else if (data.shape.length === 2) {
        // Direct 2D matrix mapping
        sampleExpression = data.slice([
          [cellIndex, cellIndex + 1],
          [0, sampleGenes.length],
        ])
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
 * Run the prediction on the model and store the results in IndexedDB
 * @param {string} modelID - The id of the model
 * @param {string} modelsURL - The URL of the model
 * @param {File} h5File - The h5ad file
 * @param {number} cellRangePercent - The percentage of cells to process
 * When finished, it sends a "finishedPrediction" message to the main thread.
 */
async function start(
  modelID: string,
  modelsURL: string,
  h5File: File,
  cellRangePercent: number,
  useWebGPU: boolean
): Promise<void> {
  console.log(`Starting embedding for model ${modelID} with file ${h5File.name}`)
  self.postMessage({ type: 'status', message: 'Loading libraries...' })
  const Module = await h5wasm.ready
  const { FS } = Module
  console.log('h5wasm loaded')

  try {
    // Load the model if it's not already loaded
    if (!model || model.modelID !== modelID) {
      model = await instantiateModel(modelsURL, modelID, useWebGPU)
    }

    // Load the h5ad file mapping it to the /work directory so we can read
    // it with h5wasm incrementally to support unlimited file sizes
    // We also figure out the list of genes in the sample and the list of cell names
    self.postMessage({ type: 'status', message: 'Loading file' })
    if (!FS.analyzePath('/work').exists) {
      FS.mkdir('/work')
    }
    FS.mount(FS.filesystems.WORKERFS, { files: [h5File] }, '/work')

    const annData = new h5wasm.File(`/work/${h5File.name}`, 'r') as H5File
    console.log(annData)

    console.log(`Top level keys: ${annData.keys()}`)

    // Extract cell names using the new function
    let cellNames: string[]
    try {
      cellNames = getCellNames(annData)
      console.log(`Found ${cellNames.length} cells`)
    } catch (error) {
      throw new Error(`Failed to extract cell names: ${error instanceof Error ? error.message : String(error)}`)
    }

    // Extract gene names using the new function
    let sampleGenes: string[]
    try {
      sampleGenes = getSampleGenes(annData)
      console.log(`Found ${sampleGenes.length} genes`)
    } catch (error) {
      throw new Error(`Failed to extract gene names: ${error instanceof Error ? error.message : String(error)}`)
    }

    const totalNumCells = cellNames.length

    // Limit the number of cells to process based on % slider
    cellNames = cellNames.slice(0, (cellRangePercent * cellNames.length) / 100)

    // Extract raw counts using the new function
    let rawCountsData: RawCountsData
    try {
      rawCountsData = getRawCounts(annData)
      console.log(`Found expression data (sparse: ${rawCountsData.isSparse})`)
    } catch (error) {
      throw new Error(`Failed to extract expression data: ${error instanceof Error ? error.message : String(error)}`)
    }

    // Destructure the raw counts data for use in the rest of the function
    const { isSparse, data, indices, indptr } = rawCountsData

    // const coordinates: number[][] = []

    const inflationIndices = precomputeInflationIndices(model.genes, sampleGenes)

    const startTime = Date.now() // Record start time

    // Initialize double buffers of batches
    const buffers: Buffer[] = [
      {
        size: 0,
        data: new Float32Array(Math.min(batchSize, cellNames.length) * model.genes.length),
      },
      {
        size: 0,
        data: new Float32Array(Math.min(batchSize, cellNames.length) * model.genes.length),
      },
    ]
    let activeBuffer = 0

    // Fill the first buffer to kickstart the process whereby while prediction runs
    // on the first buffer, the second buffer is filled with
    // the next batch of cells.
    buffers[activeBuffer].size = Math.min(batchSize, cellNames.length)
    fillBatchData(
      0,
      buffers[activeBuffer].size,
      data,
      indices,
      indptr,
      isSparse,
      sampleGenes,
      inflationIndices,
      buffers[activeBuffer].data
    )

    // Begin processing batches of cells double buffer style
    for (let batchStart = 0; batchStart < cellNames.length; batchStart += batchSize) {
      // Start inference async on the active buffer
      const inputTensor = new Tensor('float32', buffers[activeBuffer].data, [
        buffers[activeBuffer].size,
        model.genes.length,
      ])
      const inferencePromise = model.embeddingSession.run({
        input: inputTensor,
      }) as unknown as Promise<EmbeddingOutput>

      // Fill next buffer while inference runs asynchronously
      const nextBuffer = (activeBuffer + 1) % 2
      const nextStart = batchStart + batchSize
      if (nextStart < cellNames.length) {
        const nextEnd = Math.min(nextStart + batchSize, cellNames.length)
        const nextSize = nextEnd - nextStart
        buffers[nextBuffer].size = Math.min(nextSize, cellNames.length - nextStart)
        if (nextSize < Math.min(batchSize, cellNames.length)) {
          // On the last batch and its less then full size so we need to
          // resize the Float32Array for the Tensor creator
          buffers[nextBuffer].data = new Float32Array(nextSize * model.genes.length)
        }
        fillBatchData(
          nextStart,
          buffers[nextBuffer].size,
          data,
          indices,
          indptr,
          isSparse,
          sampleGenes,
          inflationIndices,
          buffers[nextBuffer].data
        )
      }

      // Wait for inference to complete on the current buffer
      const results = await inferencePromise

      // Encode embeddings using PQ model
      const pqCodes = await model.pqModel.encode(results.output.data as Float32Array)

      // Map the embeddings to 2D coordinates
      const mappingPromise = model.mappingSession.run({
        input: results.output,
      }) as unknown as Promise<MappingOutput>

      const mappings = await mappingPromise

      // Reshape into an array of 2D for the plotting packages
      const coordinates: number[][] = []
      for (let i = 0; i < mappings.output.dims[0]; i++) {
        const startIndex = i * 2 // Calculate startIndex for [x, y] pair
        coordinates.push([
          Number(mappings.output.data[startIndex]),
          Number(mappings.output.data[startIndex + 1]),
        ])
      }

      // Calculate test vector IDs for this batch using actual cell names from h5ad file
      const testVectorIds: string[] = []
      for (let i = 0; i < buffers[activeBuffer].size; i++) {
        const cellIndex = batchStart + i
        testVectorIds.push(cellNames[cellIndex])
      }

      // Send embedding data: (test_vector_id, pq_embedding, umap_coordinates)
      self.postMessage({
        type: 'embedding',
        test_vector_id: testVectorIds,
        pq_embedding: pqCodes,
        umap_coordinates: coordinates,
      })

      self.postMessage({
        type: 'progress',
        message: `Embedded and mapped ${cellNames.length} out of ${totalNumCells}...`,
        countFinished: nextStart,
        totalToProcess: cellNames.length,
      })

      // Swap buffers
      activeBuffer = nextBuffer
    }

    // All done so unmount the h5 file from the browsers file system
    annData.close()
    FS.unmount('/work')

    // Record end time and calculate elapsed time of prediction only
    const endTime = Date.now() // Record end time
    const elapsedTime = (endTime - startTime) / 60000 // Calculate elapsed time in minutes

    // Let the main thread know we're done and results are ready in IndexDB
    self.postMessage({
      type: 'finished',
      datasetLabel: h5File.name,
      elapsedTime,
      totalProcessed: cellNames.length,
      totalNumCells,
    })
  } catch (error) {
    // Try to unmount the file system if it was mounted
    try {
      FS.unmount('/work')
    } catch (unmountError) {
      console.error('Failed to unmount file system:', unmountError)
    }
    
    // Send error message to main thread
    const errorMessage = error instanceof Error ? error.message : String(error)
    console.error('Embedder error:', errorMessage)
    self.postMessage({ 
      type: 'error', 
      error: errorMessage 
    })
  }
}
