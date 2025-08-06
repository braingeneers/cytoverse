/**
 * Browser-side Inverted File Index (IVF) implementation for IVFPQ search with residual vectors.
 *
 * This module provides IVF search functionality for finding the most relevant
 * partitions given a query vector, computing query residuals, and loading partition
 * data from custom binary files. Works with residual vectors for improved accuracy.
 */

/**
 * IVF metadata loaded from centroids.bin and metadata.json
 */
interface IVFMetadata {
  n_partitions: number
  d: number
  centroids: Float32Array // [n_partitions, d] flattened
  version?: string
}

/**
 * Partition data loaded from partition files with residual vector support
 */
interface PartitionData {
  vector_ids: Int32Array
  pq_codes: Uint8Array // Pre-flattened codes: [N * m] for direct use in asymmetric distance
  size: number
}

/**
 * IVF search implementation for browser with residual vector support
 */
export class InvertedFileIndex {
  private metadata: IVFMetadata | null = null
  private basePath: string
  private partitionCache: Map<number, PartitionData> = new Map()

  constructor(basePath: string) {
    this.basePath = basePath
  }

  /**
   * Load IVF metadata from centroids.bin and metadata.json
   */
  async loadMetadata(): Promise<void> {
    console.log(`Loading IVF metadata from ${this.basePath}`)

    try {
      // Load binary centroids
      const centroidsResponse = await fetch(`${this.basePath}/centroids.bin`)
      if (!centroidsResponse.ok) {
        throw new Error(`Failed to load centroids: ${centroidsResponse.status}`)
      }

      const arrayBuffer = await centroidsResponse.arrayBuffer()
      const dataView = new DataView(arrayBuffer)
      
      // Read header
      const nPartitions = dataView.getUint32(0, true) // little-endian
      const d = dataView.getUint32(4, true) // little-endian
      
      // Read centroids data
      const centroids = new Float32Array(arrayBuffer, 8) // Skip 8-byte header

      // Try to load additional metadata from JSON file
      let version = 'legacy'
      try {
        const metadataResponse = await fetch(`${this.basePath}/metadata.json`)
        if (metadataResponse.ok) {
          const jsonMetadata = await metadataResponse.json()
          version = jsonMetadata.version || 'legacy'
        }
      } catch (error) {
        console.warn('Could not load metadata.json, using legacy mode')
      }

      this.metadata = {
        n_partitions: nPartitions,
        d: d,
        centroids: centroids,
        version: version,
      }

      console.log(
        `Loaded IVF metadata: ${this.metadata.n_partitions} partitions, ${this.metadata.d}D, version: ${this.metadata.version}`
      )
    } catch (error) {
      console.error('Failed to load IVF metadata:', error)
      throw error
    }
  }

  /**
   * Find the most relevant partitions for a query vector and compute residuals
   */
  searchPartitions(queryVector: Float32Array, nProbe: number = 1): {partitionIds: number[], queryResiduals: Float32Array[]} {
    if (!this.metadata) {
      throw new Error('IVF metadata not loaded. Call loadMetadata() first.')
    }

    const { n_partitions, d, centroids } = this.metadata

    if (queryVector.length !== d) {
      throw new Error(
        `Query vector dimension ${queryVector.length} does not match IVF dimension ${d}`
      )
    }

    // Compute distances to all centroids
    const distances: { distance: number; partitionId: number }[] = []

    for (let p = 0; p < n_partitions; p++) {
      let distance = 0
      const centroidOffset = p * d

      for (let i = 0; i < d; i++) {
        const diff = queryVector[i] - centroids[centroidOffset + i]
        distance += diff * diff
      }

      distances.push({ distance: Math.sqrt(distance), partitionId: p })
    }

    // Sort by distance and return top nProbe partitions
    distances.sort((a, b) => a.distance - b.distance)
    const topPartitions = distances.slice(0, nProbe)
    
    // Extract partition IDs and compute query residuals
    const partitionIds: number[] = []
    const queryResiduals: Float32Array[] = []
    
    for (const partition of topPartitions) {
      const partitionId = partition.partitionId
      partitionIds.push(partitionId)
      
      // Compute query residual: query - centroid
      const queryResidual = new Float32Array(d)
      const centroidOffset = partitionId * d
      
      for (let i = 0; i < d; i++) {
        queryResidual[i] = queryVector[i] - centroids[centroidOffset + i]
      }
      
      queryResiduals.push(queryResidual)
    }

    return { partitionIds, queryResiduals }
  }

  /**
   * Load partition data from disk
   */
  async loadPartition(partitionId: number): Promise<PartitionData> {
    // Check cache first
    if (this.partitionCache.has(partitionId)) {
      return this.partitionCache.get(partitionId)!
    }

    const partitionFile = `${this.basePath}/partitions/partition_${partitionId
      .toString()
      .padStart(4, '0')}.bin`
    console.log(`Loading partition ${partitionId} from ${partitionFile}`)

    try {
      const response = await fetch(partitionFile)
      if (!response.ok) {
        // Return empty partition for missing files
        if (response.status === 404) {
          const emptyPartition: PartitionData = {
            vector_ids: new Int32Array(0),
            pq_codes: new Uint8Array(0),
            size: 0,
          }
          this.partitionCache.set(partitionId, emptyPartition)
          return emptyPartition
        }
        throw new Error(`Failed to load partition ${partitionId}: ${response.status}`)
      }

      const arrayBuffer = await response.arrayBuffer()
      const dataView = new DataView(arrayBuffer)
      
      // Read header
      const numVectors = dataView.getUint32(0, true) // little-endian
      const m = dataView.getUint32(4, true) // little-endian
      
      console.log(`Partition ${partitionId}: ${numVectors} vectors, ${m} subquantizers`)
      
      if (numVectors === 0) {
        // Handle empty partition
        const emptyPartition: PartitionData = {
          vector_ids: new Int32Array(0),
          pq_codes: new Uint8Array(0),
          size: 0,
        }
        this.partitionCache.set(partitionId, emptyPartition)
        return emptyPartition
      }
      
      // Pre-allocate arrays for optimal performance
      const vectorIds = new Int32Array(numVectors)
      const pqCodes = new Uint8Array(numVectors * m) // Pre-flattened for direct use
      
      // Read interleaved data efficiently
      let offset = 8 // Skip header
      for (let i = 0; i < numVectors; i++) {
        // Read vector ID
        vectorIds[i] = dataView.getUint32(offset, true)
        offset += 4
        
        // Read PQ codes directly into flattened array
        const codesOffset = i * m
        for (let j = 0; j < m; j++) {
          pqCodes[codesOffset + j] = dataView.getUint8(offset++)
        }
      }

      const partitionData: PartitionData = {
        vector_ids: vectorIds,
        pq_codes: pqCodes, // Already in optimal flattened format for residual processing
        size: numVectors,
      }

      // Cache the partition data
      this.partitionCache.set(partitionId, partitionData)

      console.log(`Loaded partition ${partitionId}: ${partitionData.size} vectors`)
      return partitionData
    } catch (error) {
      console.error(`Failed to load partition ${partitionId}:`, error)
      throw error
    }
  }

  /**
   * Search for nearest neighbors using residual vectors
   * 
   * @param queryVector Query vector
   * @param nProbe Number of partitions to search
   * @param k Number of neighbors to return
   * @param pqModel PQ model instance for distance computation
   * @returns Promise with search results
   */
  async searchWithResiduals(
    queryVector: Float32Array,
    nProbe: number,
    k: number,
    pqModel: any // ProductQuantizer instance
  ): Promise<{vectorIds: number[], distances: number[]}> {
    // Find relevant partitions and compute query residuals
    const { partitionIds, queryResiduals } = this.searchPartitions(queryVector, nProbe)
    
    console.log(`Searching ${partitionIds.length} partitions for ${k} nearest neighbors`)
    
    const allCandidates: {vectorId: number, distance: number}[] = []
    
    // Search each partition using PQ distance on residuals
    for (let i = 0; i < partitionIds.length; i++) {
      const partitionId = partitionIds[i]
      const queryResidual = queryResiduals[i]
      
      // Load partition data
      const partitionData = await this.loadPartition(partitionId)
      
      if (partitionData.size === 0) {
        continue
      }
      
      // Use PQ model to find nearest neighbors in this partition
      const partitionResults = await pqModel.findNearestNeighborsResidual(
        queryResidual,
        partitionData.pq_codes,
        Math.min(k * 2, partitionData.size) // Get more candidates per partition
      )
      
      // Add partition results to global candidates
      for (let j = 0; j < partitionResults.indices.length; j++) {
        const localIndex = partitionResults.indices[j]
        const globalVectorId = partitionData.vector_ids[localIndex]
        const distance = partitionResults.distances[j]
        
        allCandidates.push({
          vectorId: globalVectorId,
          distance: distance
        })
      }
    }
    
    // Sort all candidates by distance and take top k
    allCandidates.sort((a, b) => a.distance - b.distance)
    const topCandidates = allCandidates.slice(0, k)
    
    const vectorIds = topCandidates.map(c => c.vectorId)
    const distances = topCandidates.map(c => c.distance)
    
    return { vectorIds, distances }
  }

  /**
   * Get metadata information
   */
  getMetadata(): IVFMetadata | null {
    return this.metadata
  }

  /**
   * Get centroid for a specific partition (useful for residual computation)
   */
  getCentroid(partitionId: number): Float32Array | null {
    if (!this.metadata || partitionId < 0 || partitionId >= this.metadata.n_partitions) {
      return null
    }
    
    const { d, centroids } = this.metadata
    const centroidOffset = partitionId * d
    
    return centroids.slice(centroidOffset, centroidOffset + d)
  }

  /**
   * Clear partition cache to free memory
   */
  clearCache(): void {
    this.partitionCache.clear()
  }
}