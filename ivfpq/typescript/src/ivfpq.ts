/**
 * IVFPQ (Inverted File with Product Quantization) implementation for approximate nearest neighbor search.
 * 
 * This module uses pre-trained IVFPQ models from the Python pipeline to perform ANN search
 * using ONNX models for distance computation with residual vectors.
 * 
 * Supports both HTTP and file system access for model artifacts.
 */

import { PQDistance } from './pq'

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
 * PQ metadata structure
 */
export interface PQMetadata {
  d: number
  m: number
  k: number
  d_sub: number
  compression_ratio: number
  codebooks_shape: number[]
  codebooks_size: number
  training_samples: number
  max_iterations: number
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
}

/**
 * IVFPQ system for ANN search using pre-trained models
 */
export class IVFPQ {
  private basePath: string
  private useHttp: boolean
  private metadata: IVFPQMetadata | null = null
  private pqMetadata: PQMetadata | null = null
  private centroids: Float32Array | null = null
  private codebooks: Float32Array | null = null
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
    
    // Load PQ metadata
    await this.loadPQMetadata()
    
    // Load PQ codebooks
    await this.loadCodebooks()
    
    // Initialize PQ distance calculator
    this.pqDistance = new PQDistance(
      this.pqMetadata!.m,
      this.pqMetadata!.k,
      this.pqMetadata!.d_sub
    )
    
    // Load PQ distance ONNX model
    await this.loadPQDistanceModel()
    
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
   * Load PQ metadata from JSON file
   */
  private async loadPQMetadata(): Promise<void> {
    const data = await this.fetchData('pq_metadata.json', 'json')
    this.pqMetadata = data as PQMetadata
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
   * Load PQ codebooks from binary file
   */
  private async loadCodebooks(): Promise<void> {
    const buffer = await this.fetchData('pq_codebooks.bin', 'arraybuffer') as ArrayBuffer
    
    // The file contains raw float32 array without header
    this.codebooks = new Float32Array(buffer)
    
    // Validate the size matches expected dimensions
    const expectedSize = this.pqMetadata!.m * this.pqMetadata!.k * this.pqMetadata!.d_sub
    if (this.codebooks.length !== expectedSize) {
      throw new Error(
        `Codebooks size mismatch: got ${this.codebooks.length}, expected ${expectedSize} ` +
        `(m=${this.pqMetadata!.m}, k=${this.pqMetadata!.k}, d_sub=${this.pqMetadata!.d_sub})`
      )
    }
  }
  
  /**
   * Load PQ distance ONNX model
   */
  private async loadPQDistanceModel(): Promise<void> {
    const modelPath = `${this.basePath}/pq_distance.onnx`
    await this.pqDistance!.loadModel(modelPath)
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
    if (m !== this.pqMetadata!.m) {
      throw new Error(`Partition m=${m} doesn't match PQ m=${this.pqMetadata!.m}`)
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
   * Search for k nearest neighbors
   */
  async search(queryVector: Float32Array, config: Partial<SearchConfig> = {}): Promise<SearchResults> {
    if (!this.metadata || !this.pqMetadata || !this.centroids || !this.codebooks || !this.pqDistance) {
      throw new Error('IVFPQ system not loaded')
    }
    
    const { n_probe = 1, k = 10 } = config
    const d = this.metadata.d
    
    if (queryVector.length !== d) {
      throw new Error(`Query vector dimension ${queryVector.length} doesn't match index dimension ${d}`)
    }
    
    // Step 1: Find nearest partitions
    const partitionIds = this.findNearestPartitions(queryVector, n_probe)
    
    // Step 2: Search within each partition
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
      
      // Use PQ distance to find nearest neighbors in this partition
      const partitionResults = await this.pqDistance.search(
        queryResidual,
        partition.pq_codes,
        this.codebooks,
        Math.min(k, partition.size)
      )
      
      // Convert local indices to global indices
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
    
    return {
      indices: topK.map(c => c.index),
      distances: topK.map(c => c.distance)
    }
  }
  
  /**
   * Fetch data from HTTP or file system
   */
  private async fetchData(path: string, type: 'json' | 'arraybuffer'): Promise<any> {
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
    if (!this.metadata || !this.pqMetadata) {
      return null
    }
    return { ivf: this.metadata, pq: this.pqMetadata }
  }
  
  /**
   * Check if system is ready
   */
  isReady(): boolean {
    return !!(this.metadata && this.pqMetadata && this.centroids && this.codebooks && this.pqDistance)
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