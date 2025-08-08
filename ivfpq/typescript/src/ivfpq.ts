/**
 * Integrated IVFPQ (Inverted File with Product Quantization) system for approximate nearest neighbor search.
 * 
 * This module combines IVF partitioning with PQ compression to enable fast ANN search
 * on large datasets using residual vectors for improved accuracy.
 * 
 * Supports:
 * - Loading pre-trained IVFPQ models from Python pipeline
 * - ONNX model inference for IVF partition selection
 * - Residual-based PQ distance computation
 * - Browser and Node.js environments
 * - HTTP fetching for browser-based workers
 */

import * as ort from 'onnxruntime-web'
import { InvertedFileIndex } from './ivf'
import { ProductQuantizer, SearchResults, PQMetadata, loadPQMetadata } from './pq'

/**
 * IVFPQ search configuration
 */
export interface IVFPQConfig {
  n_probe: number  // Number of partitions to search
  k: number        // Number of nearest neighbors to return
  basePath?: string // Base path for loading model artifacts
}

/**
 * IVFPQ metadata structure
 */
export interface IVFPQMetadata {
  ivf: {
    n_partitions: number
    d: number
    version: string
  }
  pq: PQMetadata
}

/**
 * IVFPQ search results with global indices
 */
export interface IVFPQSearchResults {
  indices: number[]    // Global vector indices in original dataset
  distances: number[]  // Distances to query vector
  partitions?: number[] // Partition IDs where results were found (optional)
}

/**
 * Integrated IVFPQ system for ANN search using residual vectors
 */
export class IVFPQ {
  private ivf: InvertedFileIndex
  private pq: ProductQuantizer
  private ivfSession: ort.InferenceSession | null = null
  private metadata: IVFPQMetadata | null = null
  private basePath: string
  
  constructor(basePath: string) {
    this.basePath = basePath
    this.ivf = new InvertedFileIndex(basePath)
    this.pq = null as any // Will be initialized during load
  }
  
  /**
   * Load complete IVFPQ system from pre-trained models
   */
  async load(): Promise<void> {
    console.log(`Loading IVFPQ system from ${this.basePath}`)
    
    // Load IVF metadata and centroids
    await this.ivf.loadMetadata()
    
    // Load PQ metadata and create PQ instance
    const pqMetadata = await loadPQMetadata(`${this.basePath}/pq_metadata.json`)
    this.pq = new ProductQuantizer(pqMetadata)
    
    // Load PQ codebooks
    const codebooksResponse = await fetch(`${this.basePath}/pq_codebooks.bin`)
    if (!codebooksResponse.ok) {
      throw new Error(`Failed to load PQ codebooks: ${codebooksResponse.statusText}`)
    }
    const codebooksBuffer = await codebooksResponse.arrayBuffer()
    const codebooks = new Float32Array(codebooksBuffer)
    this.pq.loadCodebooks(codebooks)
    
    // Load PQ distance model for residual search
    await this.pq.loadDistanceModel(`${this.basePath}/pq_distance.onnx`)
    
    // Try to load IVF ONNX model if available
    try {
      await this.loadIVFModel(`${this.basePath}/ivf_search.onnx`)
    } catch (error) {
      console.warn('IVF ONNX model not found, using CPU-based search')
    }
    
    // Store combined metadata
    this.metadata = {
      ivf: {
        n_partitions: this.ivf['metadata'].n_partitions,
        d: this.ivf['metadata'].d,
        version: this.ivf['metadata'].version || 'residual-1.0'
      },
      pq: pqMetadata
    }
    
    console.log('IVFPQ system loaded successfully')
    console.log(`  IVF: ${this.metadata.ivf.n_partitions} partitions, ${this.metadata.ivf.d}D`)
    console.log(`  PQ: m=${pqMetadata.m}, k=${pqMetadata.k}, compression=${pqMetadata.compression_ratio}x`)
  }
  
  /**
   * Load IVF ONNX model for accelerated partition search
   */
  async loadIVFModel(modelPath: string): Promise<void> {
    try {
      this.ivfSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        logSeverityLevel: 3,
      })
      console.log(`Loaded IVF ONNX model from ${modelPath}`)
    } catch (error) {
      console.error('Failed to load IVF model:', error)
      throw error
    }
  }
  
  /**
   * Search for k nearest neighbors using IVFPQ with residual vectors
   * 
   * @param queryVectors - Query vectors as Float32Array [batch_size, d]
   * @param config - Search configuration
   * @returns Search results with global indices and distances
   */
  async search(
    queryVectors: Float32Array,
    config: Partial<IVFPQConfig> = {}
  ): Promise<IVFPQSearchResults[]> {
    if (!this.metadata) {
      throw new Error('IVFPQ system not loaded. Call load() first.')
    }
    
    const { n_probe = 1, k = 10 } = config
    const d = this.metadata.ivf.d
    const batchSize = queryVectors.length / d
    
    if (queryVectors.length % d !== 0) {
      throw new Error(`Query vectors length ${queryVectors.length} not divisible by dimension ${d}`)
    }
    
    const results: IVFPQSearchResults[] = []
    
    // Process each query vector
    for (let q = 0; q < batchSize; q++) {
      const queryOffset = q * d
      const queryVector = queryVectors.slice(queryOffset, queryOffset + d)
      
      // Step 1: Find top partitions and compute query residuals
      const { partitionIds, queryResiduals } = await this.searchPartitions(queryVector, n_probe)
      
      // Step 2: Search within each partition using residual-based PQ
      const partitionResults: { indices: number[], distances: number[], partition: number }[] = []
      
      for (let i = 0; i < partitionIds.length; i++) {
        const partitionId = partitionIds[i]
        const queryResidual = queryResiduals[i]
        
        // Load partition data
        const partitionData = await this.ivf.loadPartition(partitionId)
        
        if (partitionData.size === 0) {
          continue // Skip empty partitions
        }
        
        // Search using PQ with residual vectors
        const searchResults = await this.pq.findNearestNeighborsResidual(
          queryResidual,
          partitionData.pq_codes,
          Math.min(k, partitionData.size) // Don't request more than partition size
        )
        
        // Convert local indices to global indices
        const globalIndices = searchResults.indices.map(localIdx => 
          partitionData.vector_ids[localIdx]
        )
        
        partitionResults.push({
          indices: globalIndices,
          distances: searchResults.distances,
          partition: partitionId
        })
      }
      
      // Step 3: Merge results from all partitions and select top-k
      const mergedResults = this.mergePartitionResults(partitionResults, k)
      results.push(mergedResults)
    }
    
    return results
  }
  
  /**
   * Search for relevant partitions using ONNX model or CPU fallback
   */
  private async searchPartitions(
    queryVector: Float32Array,
    nProbe: number
  ): Promise<{ partitionIds: number[], queryResiduals: Float32Array[] }> {
    
    if (this.ivfSession) {
      // Use ONNX model for partition search
      const queryTensor = new ort.Tensor('float32', queryVector, [1, queryVector.length])
      const nProbeTensor = new ort.Tensor('int64', [BigInt(nProbe)], [1])
      
      const outputs = await this.ivfSession.run({
        query: queryTensor,
        n_probe: nProbeTensor
      })
      
      // Extract partition IDs and residuals
      const partitionIds = Array.from(outputs.partition_ids.data as BigInt64Array).map(Number)
      const residualsData = outputs.query_residuals.data as Float32Array
      
      // Reshape residuals [n_probe, d]
      const d = queryVector.length
      const queryResiduals: Float32Array[] = []
      for (let i = 0; i < nProbe; i++) {
        const residual = residualsData.slice(i * d, (i + 1) * d)
        queryResiduals.push(residual)
      }
      
      return { partitionIds, queryResiduals }
    } else {
      // Fallback to CPU-based search
      return this.ivf.searchPartitions(queryVector, nProbe)
    }
  }
  
  /**
   * Merge results from multiple partitions and select top-k globally
   */
  private mergePartitionResults(
    partitionResults: { indices: number[], distances: number[], partition: number }[],
    k: number
  ): IVFPQSearchResults {
    // Combine all results
    const allResults: { index: number, distance: number, partition: number }[] = []
    
    for (const pr of partitionResults) {
      for (let i = 0; i < pr.indices.length; i++) {
        allResults.push({
          index: pr.indices[i],
          distance: pr.distances[i],
          partition: pr.partition
        })
      }
    }
    
    // Sort by distance
    allResults.sort((a, b) => a.distance - b.distance)
    
    // Take top-k
    const topK = allResults.slice(0, k)
    
    return {
      indices: topK.map(r => r.index),
      distances: topK.map(r => r.distance),
      partitions: topK.map(r => r.partition)
    }
  }
  
  /**
   * Get system metadata
   */
  getMetadata(): IVFPQMetadata | null {
    return this.metadata
  }
  
  /**
   * Check if system is loaded and ready
   */
  isReady(): boolean {
    return this.metadata !== null && this.pq !== null
  }
}

/**
 * Factory function to create and load an IVFPQ system
 */
export async function createIVFPQ(
  basePath: string,
  config?: Partial<IVFPQConfig>
): Promise<IVFPQ> {
  const ivfpq = new IVFPQ(config?.basePath || basePath)
  await ivfpq.load()
  return ivfpq
}

/**
 * Load IVFPQ system for browser-based workers with HTTP fetching
 */
export async function loadIVFPQForBrowser(
  baseUrl: string,
  config?: Partial<IVFPQConfig>
): Promise<IVFPQ> {
  // Ensure we're using HTTP/HTTPS URLs for browser context
  if (!baseUrl.startsWith('http://') && !baseUrl.startsWith('https://')) {
    throw new Error('Browser context requires HTTP/HTTPS URLs for model loading')
  }
  
  return createIVFPQ(baseUrl, config)
}

/**
 * Utility to validate IVFPQ database structure
 */
export async function validateIVFPQDatabase(basePath: string): Promise<boolean> {
  try {
    // Check for required files
    const requiredFiles = [
      'centroids.bin',
      'pq_metadata.json',
      'pq_codebooks.bin',
      'pq_distance.onnx'
    ]
    
    for (const file of requiredFiles) {
      const response = await fetch(`${basePath}/${file}`)
      if (!response.ok) {
        console.error(`Missing required file: ${file}`)
        return false
      }
    }
    
    // Try to load metadata to validate format
    const pqMetadata = await loadPQMetadata(`${basePath}/pq_metadata.json`)
    if (!pqMetadata.version || !pqMetadata.version.startsWith('residual')) {
      console.warn('PQ metadata version not compatible with residual vectors')
    }
    
    return true
  } catch (error) {
    console.error('Failed to validate IVFPQ database:', error)
    return false
  }
}

/**
 * Export types and interfaces for external use
 */
export type { 
  SearchResults,
  PQMetadata
} from './pq'

export { InvertedFileIndex } from './ivf'
export { ProductQuantizer } from './pq'