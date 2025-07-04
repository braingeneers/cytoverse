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
          inflatedBatchData[batchOffset + sampleIndex] = values[j]
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

    let cellNames: string[] = []
    if (annData.get('obs').type == 'Dataset') {
      cellNames = (annData.get('obs') as H5DataSet).value.map((e: any) => e[0])
    } else if (annData.get('obs').type == 'Group') {
      const obsGroup = annData.get('obs') as H5Group
      if (obsGroup.keys().includes('index')) {
        cellNames = (obsGroup.get('index') as H5DataSet).value
      } else if (obsGroup.keys().includes('_index')) {
        cellNames = (obsGroup.get('_index') as H5DataSet).value
      } else {
        throw new Error('Could not find cell names')
      }
    } else {
      throw new Error('Could not find cell names')
    }

    let sampleGenes: string[] = []
    if (annData.get('var').type == 'Dataset') {
      sampleGenes = (annData.get('var') as H5DataSet).value.map((e: any) => e[0])
    } else if (annData.get('var').type == 'Group') {
      const varGroup = annData.get('var') as H5Group
      if (varGroup.keys().includes('index')) {
        sampleGenes = (varGroup.get('index') as H5DataSet).value
      } else if (varGroup.keys().includes('_index')) {
        sampleGenes = (varGroup.get('_index') as H5DataSet).value
      } else {
        throw new Error('Could not find genes')
      }
    } else {
      throw new Error('Could not find genes')
    }

    const totalNumCells = cellNames.length

    // Limit the number of cells to process based on % slider
    cellNames = cellNames.slice(0, (cellRangePercent * cellNames.length) / 100)

    // Setup all the data structures for reading the expression data incrementally
    // including dealing with sparse and inflating the sample gene list into the
    // model's gene list.
    let isSparse = false
    let data: H5DataSet
    let indices: H5DataSet | null = null
    let indptr: H5DataSet | null = null
    if (annData.get('X').type == 'Dataset') {
      isSparse = false
      data = annData.get('X') as H5DataSet
    } else if (annData.get('X').type == 'Group') {
      isSparse = true
      const xGroup = annData.get('X') as H5Group
      data = xGroup.get('data') as H5DataSet
      indices = xGroup.get('indices') as H5DataSet
      indptr = xGroup.get('indptr') as H5DataSet
    } else {
      throw new Error('Could not find expression data')
    }

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
    // FS.unmount("/work");
    self.postMessage({ type: 'predictionError', error: error })
  }
}
