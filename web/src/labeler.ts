/**
 * Labeler Web Worker for real-time cell type prediction using IVFPQ search.
 *
 * This worker receives PQ-encoded embeddings from the embedder worker and performs
 * IVF search to find the closest training vector IDs, enabling real-time cell
 * type labeling during embedding processing.
 *
 * Data Flow:
 * embedder.ts → (test_vector_ids, pq_embeddings, umap_coordinates) → labeler.ts
 * labeler.ts → (test_vector_ids, pq_embeddings, umap_coordinates, train_vector_ids) → App.tsx
 */

import { InvertedFileIndex } from './ivfpq/ivf'
import { ProductQuantizer } from './ivfpq/pq'

const NUM_NEAREST_NEIGHBORS = 50 // Number of nearest neighbors to return for each query
const NUM_PARTITIONS_TO_SEARCH = 2 // Number of partitions to probe for each query

// Message types
interface StartLabelerMessage {
  type: 'start'
  modelsURL: string
  modelID: string
  categoryData: Int32Array
  categoryDataLength: number
}

interface EmbeddingMessage {
  type: 'embedding'
  test_vector_ids: string[]
  pq_embeddings: Uint8Array
  umap_coordinates: number[][]
  start_index: number
  end_index: number
}

interface LabelerResultMessage {
  type: 'labeled'
  test_vector_ids: string[]
  umap_coordinates: number[][]
  train_vector_ids: number[]
  label_ids: number[]
  confidences: number[]
  start_index: number
  end_index: number
}

interface StatusMessage {
  type: 'status'
  message: string
}

interface ErrorMessage {
  type: 'error'
  error: string
}

// Global state
let ivf: InvertedFileIndex | null = null
let pq: ProductQuantizer | null = null
let categoryData: Int32Array | null = null
let categoryDataLength = 0
let isInitialized = false

// Handle messages from main thread
self.addEventListener('message', async (event: MessageEvent) => {
  try {
    switch (event.data.type) {
      case 'start':
        await handleStart(event.data as StartLabelerMessage)
        break
      case 'embedding':
        await handleEmbedding(event.data as EmbeddingMessage)
        break
      default:
        console.warn('Unknown message type:', event.data.type)
    }
  } catch (error) {
    console.error('Labeler worker error:', error)
    self.postMessage({
      type: 'error',
      error: error instanceof Error ? error.message : String(error),
    } as ErrorMessage)
  }
})

/**
 * Initialize the labeler with IVFPQ models
 */
async function handleStart(message: StartLabelerMessage): Promise<void> {
  console.log('Initializing labeler worker...')

  self.postMessage({
    type: 'status',
    message: 'Loading IVFPQ models...',
  } as StatusMessage)

  try {
    // Store category data
    categoryData = message.categoryData
    categoryDataLength = message.categoryDataLength

    // Initialize IVF index
    const ivfBasePath = `${message.modelsURL}/${message.modelID}/ivfpq`
    ivf = new InvertedFileIndex(ivfBasePath)
    await ivf.loadMetadata()

    // Initialize PQ model
    const pqBasePath = `${message.modelsURL}/${message.modelID}/pq`
    pq = await import('./ivfpq/pq').then((module) => module.loadPQModel(pqBasePath))

    isInitialized = true

    self.postMessage({
      type: 'status',
      message: 'Labeler initialized successfully',
    } as StatusMessage)

    console.log('Labeler worker initialized successfully')
  } catch (error) {
    console.error('Failed to initialize labeler:', error)
    throw new Error(`Labeler initialization failed: ${error}`)
  }
}

/**
 * Process embedding batch and find closest training vector IDs
 */
async function handleEmbedding(message: EmbeddingMessage): Promise<void> {
  if (!isInitialized || !ivf || !pq || !categoryData) {
    throw new Error('Labeler not initialized')
  }

  const { test_vector_ids, pq_embeddings, umap_coordinates } = message
  const batchSize = test_vector_ids.length

  console.log(`Processing batch of ${batchSize} embeddings for labeling`)
  console.log(`PQ embedding length: ${pq_embeddings.length}`)

  // For each PQ embedding, find the closest training vector
  const trainVectorIds: number[] = []
  const labelIds: number[] = []
  const confidences: number[] = []
  const k = NUM_NEAREST_NEIGHBORS // Number of nearest neighbors to find

  // Reshape PQ codes for processing (assuming pq_embeddings contains codes for all vectors in batch)
  const codesPerVector = pq.m // number of subquantizers
  const totalCodes = pq_embeddings.length
  const expectedCodes = batchSize * codesPerVector

  console.log(
    `Expected codes: ${expectedCodes}, actual codes: ${totalCodes}, codes per vector: ${codesPerVector}`
  )

  // Determine how many vectors we can actually process
  const actualBatchSize = Math.min(batchSize, Math.floor(totalCodes / codesPerVector))
  console.log(`Processing ${actualBatchSize} out of ${batchSize} vectors`)

  if (totalCodes !== expectedCodes) {
    console.error(`PQ embedding size mismatch: expected ${expectedCodes}, got ${totalCodes}`)
  }

  // Process each vector in the batch
  for (let i = 0; i < actualBatchSize; i++) {
    try {
      // Extract PQ codes for this vector
      const vectorCodes = new Uint8Array(codesPerVector)
      const offset = i * codesPerVector
      for (let j = 0; j < codesPerVector; j++) {
        vectorCodes[j] = pq_embeddings[offset + j]
      }

      // Decode PQ codes to approximate vector for IVF search
      const reconstructed = pq.decode(vectorCodes)

      // Search for relevant partitions
      const nProbe = NUM_PARTITIONS_TO_SEARCH
      const selectedPartitions = ivf.searchPartitions(reconstructed, nProbe)

      // Collect all candidates from all selected partitions
      const candidates: Array<[number, number]> = [] // [trainVectorId, distance] pairs

      for (const partitionId of selectedPartitions) {
        const partitionData = await ivf.loadPartition(partitionId)

        if (partitionData.size === 0) continue

        // Use pre-flattened partition codes directly for asymmetric distance computation
        const partitionCodes = partitionData.pq_codes

        // Compute asymmetric distances for all vectors in partition
        const distances = pq.asymmetricDistance(reconstructed, partitionCodes)

        // Collect all candidates from this partition
        for (let j = 0; j < partitionData.size; j++) {
          // const distance = Math.sqrt(distances[j]) 
          // // asymmetricDistance returns squared distances but we just use to rank
          // candidates.push([partitionData.vector_ids[j], distance])
          candidates.push([partitionData.vector_ids[j], distances[j]])
        }
      }

      // Sort all candidates by distance and take top k
      candidates.sort((a, b) => a[1] - b[1])
      const topK = candidates.slice(0, k)

      // Store the nearest neighbor ID for backward compatibility
      const nearestTrainVectorId = topK.length > 0 ? topK[0][0] : -1

      // Convert k nearest neighbor IDs to label IDs and compute consensus
      let consensusLabelId = -1
      let consensusConfidence = 0

      if (topK.length > 0) {
        const labelVotes: { [labelId: number]: number } = {}

        // Convert each neighbor ID to label ID and count votes
        for (const [trainVectorId] of topK) {
          if (trainVectorId !== -1 && trainVectorId < categoryDataLength) {
            const labelId = categoryData[trainVectorId]
            if (labelId >= 0) {
              labelVotes[labelId] = (labelVotes[labelId] || 0) + 1
            }
          }
        }

        // Find label with most votes
        let maxVotes = 0
        for (const [labelId, votes] of Object.entries(labelVotes)) {
          if (votes > maxVotes) {
            maxVotes = votes
            consensusLabelId = parseInt(labelId)
          }
        }

        // Calculate confidence as count of consensus label / k
        if (maxVotes > 0) {
          consensusConfidence = maxVotes / k
        }
      }

      if (nearestTrainVectorId === -1) {
        console.log(
          `❌ No nearest neighbor found for test vector ${i} in batch. Selected partitions: ${selectedPartitions.join(
            ', '
          )}`
        )
      }

      trainVectorIds.push(nearestTrainVectorId)
      labelIds.push(consensusLabelId)
      confidences.push(consensusConfidence)

      // Release reconstructed vector memory
      reconstructed.fill(0)
    } catch (vectorError) {
      console.error(`Error processing vector ${i}:`, vectorError)
      trainVectorIds.push(-1) // Use -1 to indicate failure
      labelIds.push(-1)
      confidences.push(0)
    }
  }

  // Fill remaining slots with -1 if we couldn't process all vectors
  while (trainVectorIds.length < batchSize) {
    trainVectorIds.push(-1)
    labelIds.push(-1)
    confidences.push(0)
  }

  console.log(`Processed ${actualBatchSize} vectors, returning ${trainVectorIds.length} results`)

  // Send results back to main thread
  self.postMessage({
    type: 'labeled',
    test_vector_ids,
    umap_coordinates,
    train_vector_ids: trainVectorIds,
    label_ids: labelIds,
    confidences: confidences,
    start_index: message.start_index,
    end_index: message.end_index,
  } as LabelerResultMessage)

  console.log(
    `Labeled batch: found ${trainVectorIds.filter((id) => id !== -1).length}/${batchSize} matches`
  )
}

console.log('Labeler worker loaded')
