/**
 * User IVFPQ wrapper that uses user-generated indexes stored in IndexedDB
 * instead of fetching partition data from HTTP.
 * 
 * This wrapper implements a custom IVFPQ-like interface for user indexes
 * while reusing the base model's centroids and PQ codebooks through composition.
 */

import { IVFPQ, SearchConfig, SearchResults, IVFPQMetadata } from '@cytoverse/ivfpq'
import { PQDistance } from '@cytoverse/ivfpq/dist/pq'
import { userIndexService, type UserIndex } from './userIndexService'

export class UserIVFPQ {
  private baseIVFPQ: IVFPQ
  private userIndex: UserIndex | null = null
  private baseModelPath: string
  
  constructor(baseModelPath: string) {
    this.baseModelPath = baseModelPath
    this.baseIVFPQ = new IVFPQ(baseModelPath)
  }
  
  /**
   * Load a user index along with its base model
   */
  async loadUserIndex(userIndexId: string): Promise<void> {
    // Load the user index from IndexedDB
    this.userIndex = await userIndexService.getUserIndex(userIndexId)
    if (!this.userIndex) {
      throw new Error(`User index ${userIndexId} not found`)
    }
    
    // Load the base model (centroids, PQ codebooks, etc.)
    await this.baseIVFPQ.load()
    
    console.log(`User index loaded: ${this.userIndex.name} with ${this.userIndex.cellCount} cells`)
  }
  
  /**
   * Search using the user index
   * This reimplements the search logic but uses user index partitions
   */
  async search(
    queryVector: Float32Array,
    config: Partial<SearchConfig> = {}
  ): Promise<SearchResults> {
    if (!this.userIndex) {
      throw new Error('User index not loaded')
    }
    
    const { n_probe = 1, k = 10 } = config
    
    // Get reference components from base model
    const components = this.baseIVFPQ.getReferenceComponents()
    if (!components) {
      throw new Error('Base model not loaded')
    }
    
    const { centroids, pqDistance, d, n_partitions } = components
    
    // Find nearest partitions using brute force search
    const distances: number[] = []
    for (let i = 0; i < n_partitions; i++) {
      let dist = 0
      for (let j = 0; j < d; j++) {
        const diff = queryVector[j] - centroids[i * d + j]
        dist += diff * diff
      }
      distances.push(dist)
    }
    
    // Get top n_probe partitions
    const sortedIndices = distances
      .map((dist, idx) => ({ dist, idx }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, n_probe)
      .map(item => item.idx)
    
    const nearestPartition = sortedIndices[0]
    
    // Compute residual for the nearest partition
    const centroidOffset = nearestPartition * d
    const residual = new Float32Array(d)
    for (let i = 0; i < d; i++) {
      residual[i] = queryVector[i] - centroids[centroidOffset + i]
    }
    
    // PQ encode the residual
    const pqCode = await pqDistance.encode(residual)
    
    // Search within user index partitions
    const allCandidates: Array<{ id: number; distance: number }> = []
    
    for (const partitionId of sortedIndices) {
      const partition = this.userIndex.partitions[partitionId]
      if (!partition || partition.pqCodeCount === 0) {
        continue
      }
      
      // Compute residual for this partition
      const partitionCentroidOffset = partitionId * d
      const partitionResidual = new Float32Array(d)
      for (let i = 0; i < d; i++) {
        partitionResidual[i] = queryVector[i] - centroids[partitionCentroidOffset + i]
      }
      
      // Compute distances to all vectors in this partition
      const distances = await pqDistance.computeDistances(
        partitionResidual,
        partition.pqCodes,
        partition.pqCodeCount
      )
      
      // Add candidates from this partition
      for (let i = 0; i < partition.pqCodeCount; i++) {
        allCandidates.push({
          id: partition.cellIndices[i],
          distance: distances[i]
        })
      }
    }
    
    // Sort by distance and take top k
    allCandidates.sort((a, b) => a.distance - b.distance)
    const topK = allCandidates.slice(0, k)
    
    return {
      indices: topK.map(c => c.id),
      distances: topK.map(c => c.distance),
      partitionId: nearestPartition,
      pqCode: pqCode
    }
  }
  
  /**
   * Get the user index metadata
   */
  getUserIndexMetadata() {
    if (!this.userIndex) {
      return null
    }
    
    return {
      id: this.userIndex.id,
      name: this.userIndex.name,
      baseModelId: this.userIndex.baseModelId,
      cellCount: this.userIndex.cellCount,
      partitionCount: Object.keys(this.userIndex.partitions).length
    }
  }
  
  /**
   * Get coordinates for visualization
   */
  getCoordinates(): { x: Float32Array; y: Float32Array } | null {
    if (!this.userIndex) {
      return null
    }
    
    return this.userIndex.coordinates
  }
  
  /**
   * Get labels for the user index
   */
  getLabels(): string[] {
    if (!this.userIndex) {
      return []
    }
    
    return this.userIndex.labels
  }
  
  /**
   * Check if the system is ready
   */
  isReady(): boolean {
    return this.userIndex !== null && this.baseIVFPQ.isReady()
  }
}