/**
 * Labeler Web Worker for real-time cell type prediction using IVFPQ search.
 *
 * This worker receives PQ-encoded embeddings from the embedder worker and performs
 * IVF search to find the closest training vector IDs, enabling real-time cell
 * type labeling during embedding processing.
 *
 * Data Flow:
 * embedder.ts → (test_vector_id, pq_embedding, umap_coordinates) → labeler.ts
 * labeler.ts → (test_vector_id, pq_embedding, umap_coordinates, train_vector_id) → App.tsx
 */

import { InvertedFileIndex } from './ivfpq/ivf'
import { ProductQuantizer } from './ivfpq/pq'

// Message types
interface StartLabelerMessage {
  type: 'start'
  modelsURL: string
  modelID: string
}

interface EmbeddingMessage {
  type: 'embedding'
  test_vector_id: string[]
  pq_embedding: Uint8Array
  umap_coordinates: number[][]
}

interface LabelerResultMessage {
  type: 'labeled'
  test_vector_id: string[]
  pq_embedding: Uint8Array
  umap_coordinates: number[][]
  train_vector_id: number[]
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
  if (!isInitialized || !ivf || !pq) {
    throw new Error('Labeler not initialized')
  }

  const { test_vector_id, pq_embedding, umap_coordinates } = message
  const batchSize = test_vector_id.length

  console.log(`Processing batch of ${batchSize} embeddings for labeling`)

  // For each PQ embedding, find the closest training vector
  const trainVectorIds: number[] = []

  // Reshape PQ codes for processing (assuming pq_embedding contains codes for all vectors in batch)
  const codesPerVector = pq.m // number of subquantizers
  const totalCodes = pq_embedding.length
  const expectedCodes = batchSize * codesPerVector

  if (totalCodes !== expectedCodes) {
    throw new Error(`PQ embedding size mismatch: expected ${expectedCodes}, got ${totalCodes}`)
  }

  // Process each vector in the batch
  for (let i = 0; i < batchSize; i++) {
    try {
      // Extract PQ codes for this vector
      const vectorCodes = new Uint8Array(codesPerVector)
      const offset = i * codesPerVector
      for (let j = 0; j < codesPerVector; j++) {
        vectorCodes[j] = pq_embedding[offset + j]
      }

      // Decode PQ codes to approximate vector for IVF search
      const reconstructed = pq.decode(vectorCodes)

      // Search for relevant partitions
      const nProbe = 4 // Configurable parameter for search quality vs speed
      const selectedPartitions = ivf.searchPartitions(reconstructed, nProbe)

      // Search within selected partitions
      let bestDistance = Infinity
      let bestTrainVectorId = -1

      for (const partitionId of selectedPartitions) {
        const partitionData = await ivf.loadPartition(partitionId)

        if (partitionData.size === 0) continue

        // Prepare codes for asymmetric distance computation
        const partitionCodes = new Uint8Array(partitionData.size * pq.m)
        for (let k = 0; k < partitionData.size; k++) {
          const vectorCodes = partitionData.pq_codes[k][0]
          for (let j = 0; j < pq.m; j++) {
            partitionCodes[k * pq.m + j] = vectorCodes[j]
          }
        }

        // Compute asymmetric distances for all vectors in partition
        const distances = pq.asymmetricDistance(reconstructed, partitionCodes)

        // Find best match in this partition
        for (let k = 0; k < partitionData.size; k++) {
          const distance = Math.sqrt(distances[k]) // asymmetricDistance returns squared distances
          if (distance < bestDistance) {
            bestDistance = distance
            bestTrainVectorId = partitionData.vector_ids[k]
          }
        }
      }

      trainVectorIds.push(bestTrainVectorId)
    } catch (vectorError) {
      console.error(`Error processing vector ${i}:`, vectorError)
      trainVectorIds.push(-1) // Use -1 to indicate failure
    }
  }

  // Send results back to main thread
  self.postMessage({
    type: 'labeled',
    test_vector_id,
    pq_embedding,
    umap_coordinates,
    train_vector_id: trainVectorIds,
  } as LabelerResultMessage)

  console.log(
    `Labeled batch: found ${trainVectorIds.filter((id) => id !== -1).length}/${batchSize} matches`
  )
}

console.log('Labeler worker loaded')
