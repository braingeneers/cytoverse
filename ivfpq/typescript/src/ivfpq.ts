/**
 * IVFPQ (Inverted File with Product Quantization) implementation for approximate nearest neighbor search.
 * 
 * - Uses pre-trained IVFPQ models from the Python pipeline to perform ANN search
 * - Combines IVF partitioning with PQ compression
 * - Supports using ONNX models for distance computation with residual vectors.
 * - Supports both HTTP and file system access for model artifacts.
 * - Support using a previously searched index as a 'user generated' index
 */

import { PQDistance, PQMetadata } from './pq'

/**
 * IVFPQ metadata structure matching Python output
 */
export interface IVFPQMetadata {
  d: number
  n_partitions: number
  pq_m: number
  pq_k: number
  total_vectors: number
  max_iterations: number
  inertia: number
  partition_sizes: { [key: string]: number }
  centroids_shape: number[]
  version: string
}

/**
 * Partition data loaded from binary files
 */
interface PartitionData {
  vector_ids: Int32Array
  pq_codes: Uint8Array
  size: number
}

/**
 * Search configuration
 */
export interface SearchConfig {
  n_probe: number
  k: number
}

/**
 * Search results
 */
export interface SearchResults {
  indices: number[]
  distances: number[]
  artifacts?: CellArtifact  // Optional artifacts when retainArtifacts is true
}

/**
 * Minimal artifact for a single cell - can be used to build user IVFPQ index
 */
export interface CellArtifact {
  partitionId: number      // Index of nearest centroid
  pqCode: Uint8Array       // PQ encoded residual (m bytes)
  labelIndex: number       // Index of assigned label via consensus
}

/**
 * User IVFPQ index built from collected artifacts
 */
export interface UserIVFPQIndex {
  modelId: string                              // Base model used
  labelCategory?: string                       // Category used for consensus labeling (added by App.vue)
  labels?: string[]                            // Actual label strings (added by App.vue)
  partitions: Map<number, {                    // Partition ID -> cells in that partition
    pqCodes: Uint8Array[]                     // PQ codes for cells
    labelIndices: number[]                     // Label indices for cells
  }>
  cellCount: number                            // Total number of cells indexed
}

/**
 * IVFPQ system for ANN search using pre-trained models
 */
export class IVFPQ {
  private basePath: string
  private useHttp: boolean
  private metadata: IVFPQMetadata | null = null
  private centroids: Float32Array | null = null
  private pqDistance: PQDistance | null = null
  private partitionCache: Map<number, PartitionData> = new Map()
  
  constructor(basePath: string) {
    this.basePath = basePath
    this.useHttp = basePath.startsWith('http://') || basePath.startsWith('https://')
  }
  
  /**
   * Load the IVFPQ system from artifacts
   */
  async load(): Promise<void> {
    console.log(`Loading IVFPQ system from ${this.basePath}`)
    
    // Load IVF metadata
    await this.loadMetadata()
    
    // Load IVF centroids
    await this.loadCentroids()
    
    // Initialize and load PQ distance calculator
    this.pqDistance = new PQDistance(this.basePath)
    await this.pqDistance.load()
    
    // Load PQ distance ONNX model
    await this.pqDistance.loadModel()

    // Load PQ encode model for building user indexes
    await this.pqDistance.loadEncodeModel()
    
    console.log(`IVFPQ system loaded: ${this.metadata!.n_partitions} partitions, ${this.metadata!.d}D`)
  }
  
  /**
   * Load IVF metadata from JSON file
   */
  private async loadMetadata(): Promise<void> {
    const data = await this.fetchData('ivf_metadata.json', 'json')
    this.metadata = data as IVFPQMetadata
  }
  
  
  /**
   * Load IVF centroids from binary file
   */
  private async loadCentroids(): Promise<void> {
    const buffer = await this.fetchData('ivf_centroids.bin', 'arraybuffer') as ArrayBuffer
    const view = new DataView(buffer)
    
    // Read header
    const n_partitions = view.getUint32(0, true)
    const d = view.getUint32(4, true)
    
    // Validate dimensions
    if (n_partitions !== this.metadata!.n_partitions || d !== this.metadata!.d) {
      throw new Error('Centroids dimensions mismatch with metadata')
    }
    
    // Read centroids data
    this.centroids = new Float32Array(buffer, 8)
  }
  
  
  
  /**
   * Load a partition from binary file
   */
  private async loadPartition(partitionId: number): Promise<PartitionData> {
    // Check cache first
    if (this.partitionCache.has(partitionId)) {
      return this.partitionCache.get(partitionId)!
    }
    
    const path = `partitions/partition_${partitionId.toString().padStart(4, '0')}.bin`
    const buffer = await this.fetchData(path, 'arraybuffer') as ArrayBuffer
    const view = new DataView(buffer)
    
    // Read header
    const num_vectors = view.getUint32(0, true)
    const m = view.getUint32(4, true)
    
    // Validate m dimension
    const pqDims = this.pqDistance!.getDimensions()!
    if (m !== pqDims.m) {
      throw new Error(`Partition m=${m} doesn't match PQ m=${pqDims.m}`)
    }
    
    // Read interleaved data
    const vector_ids = new Int32Array(num_vectors)
    const pq_codes = new Uint8Array(num_vectors * m)
    
    let offset = 8
    for (let i = 0; i < num_vectors; i++) {
      vector_ids[i] = view.getInt32(offset, true)
      offset += 4
      
      for (let j = 0; j < m; j++) {
        pq_codes[i * m + j] = view.getUint8(offset)
        offset += 1
      }
    }
    
    const partitionData = {
      vector_ids,
      pq_codes,
      size: num_vectors
    }
    
    // Cache the partition
    this.partitionCache.set(partitionId, partitionData)
    
    return partitionData
  }
  
  /**
   * Find nearest partitions for a query vector
   */
  private findNearestPartitions(queryVector: Float32Array, nProbe: number): number[] {
    if (!this.centroids || !this.metadata) {
      throw new Error('IVFPQ not loaded')
    }
    
    const n_partitions = this.metadata.n_partitions
    const d = this.metadata.d
    
    // Compute distances to all centroids
    const distances = new Float32Array(n_partitions)
    for (let i = 0; i < n_partitions; i++) {
      let dist = 0
      for (let j = 0; j < d; j++) {
        const diff = queryVector[j] - this.centroids[i * d + j]
        dist += diff * diff
      }
      distances[i] = dist
    }
    
    // Find top nProbe partitions
    const indices = Array.from({length: n_partitions}, (_, i) => i)
    indices.sort((a, b) => distances[a] - distances[b])
    
    return indices.slice(0, nProbe)
  }
  
  /**
   * Search for k nearest neighbors with optional artifact retention
   */
  async search(
    queryVector: Float32Array, 
    config: Partial<SearchConfig> = {},
    retainArtifacts: boolean = false
  ): Promise<SearchResults> {
    if (!this.metadata || !this.centroids || !this.pqDistance || !this.pqDistance.isReady()) {
      throw new Error('IVFPQ system not loaded')
    }
    
    const { n_probe = 1, k = 10 } = config
    const d = this.metadata.d
    
    if (queryVector.length !== d) {
      throw new Error(`Query vector dimension ${queryVector.length} doesn't match index dimension ${d}`)
    }
    
    // Step 1: Find nearest partitions
    const partitionIds = this.findNearestPartitions(queryVector, n_probe)
    const nearestPartition = partitionIds[0]  // Closest partition for artifacts
    
    // Prepare for artifact retention if requested
    let artifactPqCode: Uint8Array | undefined
    if (retainArtifacts && this.pqDistance['encodeSession']) {
      // Compute residual for nearest partition
      const centroidOffset = nearestPartition * d
      const nearestResidual = new Float32Array(d)
      for (let i = 0; i < d; i++) {
        nearestResidual[i] = queryVector[i] - this.centroids[centroidOffset + i]
      }
      // Encode the residual
      try {
        artifactPqCode = await this.pqDistance.encode(nearestResidual)
      } catch {
        // Encode model might not be loaded, continue without artifact
        console.warn('PQ encode model not loaded, artifacts will be incomplete')
      }
    }
    
    // Step 2: Search within each partition and collect all distances
    const allCandidates: Array<{index: number, distance: number}> = []
    
    for (const partitionId of partitionIds) {
      // Load partition data
      const partition = await this.loadPartition(partitionId)
      
      if (partition.size === 0) {
        continue
      }
      
      // Compute query residual for this partition
      const centroidOffset = partitionId * d
      const queryResidual = new Float32Array(d)
      for (let i = 0; i < d; i++) {
        queryResidual[i] = queryVector[i] - this.centroids[centroidOffset + i]
      }
      
      // PQ codes are already Uint8Array, which is what the ONNX model expects
      
      // Use PQ distance to get distances to all vectors in this partition
      const partitionResults = await this.pqDistance.search(
        queryResidual,
        partition.pq_codes
      )
      
      // Convert local indices to global indices and add all distances
      for (let i = 0; i < partitionResults.indices.length; i++) {
        const localIdx = partitionResults.indices[i]
        const globalIdx = partition.vector_ids[localIdx]
        allCandidates.push({
          index: globalIdx,
          distance: partitionResults.distances[i]
        })
      }
    }
    
    // Step 3: Sort all candidates and return top k
    allCandidates.sort((a, b) => a.distance - b.distance)
    const topK = allCandidates.slice(0, k)
    
    const result: SearchResults = {
      indices: topK.map(c => c.index),
      distances: topK.map(c => c.distance)
    }
    
    // Add artifacts if requested and available
    if (retainArtifacts && artifactPqCode) {
      // labelIndex will be set by the worker after consensus
      result.artifacts = {
        partitionId: nearestPartition,
        pqCode: artifactPqCode,
        labelIndex: -1  // Placeholder, will be set by worker after consensus
      }
    }
    
    return result
  }
  
  /**
   * Fetch data from HTTP or file system
   */
  private async fetchData(path: string, type: 'json' | 'arraybuffer'): Promise<unknown> {
    const fullPath = `${this.basePath}/${path}`
    
    if (this.useHttp) {
      const response = await fetch(fullPath)
      if (!response.ok) {
        throw new Error(`Failed to fetch ${fullPath}: ${response.statusText}`)
      }
      return type === 'json' ? response.json() : response.arrayBuffer()
    } else {
      // For Node.js file system access
      const fs = await import('fs')
      const fsPath = await import('path')
      const resolvedPath = fsPath.resolve(fullPath)
      
      if (type === 'json') {
        const data = await fs.promises.readFile(resolvedPath, 'utf-8')
        return JSON.parse(data)
      } else {
        const buffer = await fs.promises.readFile(resolvedPath)
        return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
      }
    }
  }
  
  /**
   * Get metadata
   */
  getMetadata(): { ivf: IVFPQMetadata, pq: PQMetadata } | null {
    if (!this.metadata || !this.pqDistance) {
      return null
    }
    const pqMetadata = this.pqDistance.getMetadata()
    if (!pqMetadata) {
      return null
    }
    return { ivf: this.metadata, pq: pqMetadata }
  }
  
  /**
   * Check if system is ready
   */
  isReady(): boolean {
    return !!(this.metadata && this.centroids && this.pqDistance && this.pqDistance.isReady())
  }
  
  /**
   * Get reference components for UserIVFPQ initialization
   */
  getReferenceComponents(): {
    centroids: Float32Array
    pqDistance: PQDistance
    d: number
    n_partitions: number
  } | null {
    if (!this.metadata || !this.centroids || !this.pqDistance) {
      return null
    }
    return {
      centroids: this.centroids,
      pqDistance: this.pqDistance,
      d: this.metadata.d,
      n_partitions: this.metadata.n_partitions
    }
  }
}

/**
 * Create and load an IVFPQ system
 */
export async function createIVFPQ(basePath: string): Promise<IVFPQ> {
  const ivfpq = new IVFPQ(basePath)
  await ivfpq.load()
  return ivfpq
}