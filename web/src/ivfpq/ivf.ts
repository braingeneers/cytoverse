/**
 * Browser-side Inverted File Index (IVF) implementation for IVFPQ search.
 *
 * This module provides IVF search functionality for finding the most relevant
 * partitions given a query vector, and loading partition data from custom binary files.
 */

/**
 * IVF metadata loaded from centroids.arrow
 */
interface IVFMetadata {
  n_partitions: number
  d: number
  centroids: Float32Array // [n_partitions, d] flattened
}

/**
 * Partition data loaded from partition files
 */
interface PartitionData {
  vector_ids: Int32Array
  pq_codes: Uint8Array // Pre-flattened codes: [N * m] for direct use in asymmetric distance
  size: number
}

/**
 * IVF search implementation for browser
 */
export class InvertedFileIndex {
  private metadata: IVFMetadata | null = null
  private basePath: string
  private partitionCache: Map<number, PartitionData> = new Map()

  constructor(basePath: string) {
    this.basePath = basePath
  }

  /**
   * Load IVF metadata from centroids.bin
   */
  async loadMetadata(): Promise<void> {
    console.log(`Loading IVF metadata from ${this.basePath}/centroids.bin`)

    try {
      const response = await fetch(`${this.basePath}/centroids.bin`)
      if (!response.ok) {
        throw new Error(`Failed to load centroids: ${response.status}`)
      }

      const arrayBuffer = await response.arrayBuffer()
      const dataView = new DataView(arrayBuffer)
      
      // Read header
      const nPartitions = dataView.getUint32(0, true) // little-endian
      const d = dataView.getUint32(4, true) // little-endian
      
      // Read centroids data
      const centroids = new Float32Array(arrayBuffer, 8) // Skip 8-byte header

      this.metadata = {
        n_partitions: nPartitions,
        d: d,
        centroids: centroids,
      }

      console.log(
        `Loaded IVF metadata: ${this.metadata.n_partitions} partitions, ${this.metadata.d}D`
      )
    } catch (error) {
      console.error('Failed to load IVF metadata:', error)
      throw error
    }
  }

  /**
   * Find the most relevant partitions for a query vector
   */
  searchPartitions(queryVector: Float32Array, nProbe: number = 1): number[] {
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
    return distances.slice(0, nProbe).map((d) => d.partitionId)
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
        throw new Error(`Failed to load partition ${partitionId}: ${response.status}`)
      }

      const arrayBuffer = await response.arrayBuffer()
      const dataView = new DataView(arrayBuffer)
      
      // Read header
      const numVectors = dataView.getUint32(0, true) // little-endian
      const m = dataView.getUint32(4, true) // little-endian
      
      console.log(`Partition ${partitionId}: ${numVectors} vectors, ${m} subquantizers`)
      
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
        pq_codes: pqCodes, // Already in optimal flattened format
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
   * Get metadata information
   */
  getMetadata(): IVFMetadata | null {
    return this.metadata
  }
}
