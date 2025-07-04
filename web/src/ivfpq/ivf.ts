/**
 * Browser-side Inverted File Index (IVF) implementation for IVFPQ search.
 *
 * This module provides IVF search functionality for finding the most relevant
 * partitions given a query vector, and loading partition data from Arrow files.
 */

import * as arrow from 'apache-arrow'

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
  pq_codes: Uint8Array[][] // [N, m] where each pq_codes[i] is array of m codes
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
   * Load IVF metadata from centroids.arrow
   */
  async loadMetadata(): Promise<void> {
    console.log(`Loading IVF metadata from ${this.basePath}/centroids.arrow`)

    try {
      const response = await fetch(`${this.basePath}/centroids.arrow`)
      if (!response.ok) {
        throw new Error(`Failed to load centroids: ${response.status}`)
      }

      const arrayBuffer = await response.arrayBuffer()
      const table = arrow.tableFromIPC(new Uint8Array(arrayBuffer))

      // Extract centroids data
      const centroidCoordsColumn = table.getChild('centroid_coords')
      if (!centroidCoordsColumn) {
        throw new Error('centroid_coords column not found in centroids.arrow')
      }

      // Convert centroids to Float32Array
      const nPartitions = centroidCoordsColumn.length
      const firstCentroid = centroidCoordsColumn.get(0)
      const d = firstCentroid.length

      const centroids = new Float32Array(nPartitions * d)
      let offset = 0
      for (let i = 0; i < nPartitions; i++) {
        const centroid = centroidCoordsColumn.get(i)
        for (let j = 0; j < d; j++) {
          centroids[offset++] = centroid[j]
        }
      }

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
      .padStart(4, '0')}.arrow`
    console.log(`Loading partition ${partitionId} from ${partitionFile}`)

    try {
      const response = await fetch(partitionFile)
      if (!response.ok) {
        throw new Error(`Failed to load partition ${partitionId}: ${response.status}`)
      }

      const arrayBuffer = await response.arrayBuffer()
      const table = arrow.tableFromIPC(new Uint8Array(arrayBuffer))

      // Extract vector IDs
      const vectorIdsColumn = table.getChild('vector_id')
      if (!vectorIdsColumn) {
        throw new Error(`vector_id column not found in partition ${partitionId}`)
      }

      const vectorIds = new Int32Array(vectorIdsColumn.length)
      for (let i = 0; i < vectorIdsColumn.length; i++) {
        vectorIds[i] = vectorIdsColumn.get(i)
      }

      // Extract PQ codes (assuming they are stored as code_0, code_1, ..., code_m-1)
      const pqCodes: Uint8Array[][] = []
      const firstCodeColumn = table.getChild('code_0')
      if (!firstCodeColumn) {
        throw new Error(`PQ code columns not found in partition ${partitionId}`)
      }

      // Determine number of subquantizers by checking available code columns
      let m = 0
      while (table.getChild(`code_${m}`)) {
        m++
      }

      for (let i = 0; i < vectorIds.length; i++) {
        const codes = new Uint8Array(m)
        for (let j = 0; j < m; j++) {
          const codeColumn = table.getChild(`code_${j}`)!
          codes[j] = codeColumn.get(i)
        }
        pqCodes.push([codes])
      }

      const partitionData: PartitionData = {
        vector_ids: vectorIds,
        pq_codes: pqCodes,
        size: vectorIds.length,
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
