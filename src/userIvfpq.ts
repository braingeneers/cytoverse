/**
 * User IVFPQ wrapper that uses user-generated indexes stored in IndexedDB
 * instead of fetching partition data from HTTP.
 *
 * This wrapper implements a custom IVFPQ-like interface for user indexes
 * while reusing the base model's centroids and PQ codebooks through composition.
 *
 */

import { IVFPQ, SearchResults } from './ivfpq';
import { userIndexService, type UserIndex } from './userIndexService';

export class UserIVFPQ {
  private basePath: string;
  private n_probe: number;
  private k: number;
  private baseIVFPQ: IVFPQ;
  private userIndex: UserIndex | undefined = undefined;
  private userCentroids: Float32Array | undefined = undefined;
  private userPartitions: Record<number, UserIndex['partitions'][number]> | undefined =
    undefined;

  constructor(basePath: string, n_probe: number = 1, k: number = 10) {
    this.basePath = basePath;
    this.n_probe = n_probe;
    this.k = k;
    this.baseIVFPQ = null as any; // Will be initialized in loadUserIndex
  }

  /**
   * Load a user index along with its base model
   */
  async loadUserIndex(userIndexId: string): Promise<void> {
    await userIndexService.init(); // Ensure IndexedDB is initialized
    this.baseIVFPQ = new IVFPQ(this.basePath, this.n_probe, this.k);

    // Load the user index from IndexedDB
    this.userIndex = await userIndexService.getUserIndex(userIndexId);
    if (!this.userIndex) {
      throw new Error(`User index ${userIndexId} not found`);
    }

    // Load the base model (centroids, PQ codebooks, etc.)
    await this.baseIVFPQ.load();

    // Get reference components from base model
    const components = this.baseIVFPQ.getReferenceComponents();
    if (!components) {
      throw new Error('Base model not loaded');
    }
    const { centroids, d } = components;

    // Extract partition IDs that exist in user index
    const userPartitionIds = Object.keys(this.userIndex.partitions).map((id) =>
      parseInt(id)
    );

    // Create mapping from old partition IDs to new indices
    const partitionIdToNewIndex = new Map<number, number>();
    userPartitionIds.forEach((id, newIndex) => {
      partitionIdToNewIndex.set(id, newIndex);
    });

    // Create userCentroids with only the centroids for user partitions
    const numUserPartitions = userPartitionIds.length;
    this.userCentroids = new Float32Array(numUserPartitions * d);

    // Copy relevant centroids to userCentroids
    for (let i = 0; i < numUserPartitions; i++) {
      const originalPartitionId = userPartitionIds[i];
      const sourceOffset = originalPartitionId * d;
      const destOffset = i * d;

      for (let j = 0; j < d; j++) {
        this.userCentroids[destOffset + j] = centroids[sourceOffset + j];
      }
    }

    // Create adjusted partitions with remapped partition IDs
    this.userPartitions = {};
    for (const [originalId, partition] of Object.entries(this.userIndex.partitions)) {
      const newId = partitionIdToNewIndex.get(parseInt(originalId));
      if (newId !== undefined) {
        this.userPartitions[newId] = partition;
      }
    }

    console.log(
      `User index loaded: ${this.userIndex.name} with ${this.userIndex.cellCount} cells, ${numUserPartitions} partitions`
    );
  }

  /**
   * Search using the user index
   * This reimplements the search logic but uses user index partitions
   */
  async search(queryVector: Float32Array): Promise<SearchResults> {
    if (!this.userIndex || !this.userCentroids || !this.userPartitions) {
      throw new Error('User index not loaded');
    }

    // Get reference components from base model
    const components = this.baseIVFPQ.getReferenceComponents();
    if (!components) {
      throw new Error('Base model not loaded');
    }

    const { pqDistance, d } = components;

    // Step 1: Find nearest partitions (use ONNX if requested and available)
    const partitionIds = await this.baseIVFPQ.findNearestPartitions(
      queryVector,
      this.userCentroids,
      this.n_probe
    );
    const nearestPartition = partitionIds[0]; // Closest partition for artifacts

    // PQ encode the query vector's residual to the partition for a user index
    // Compute residual for nearest partition
    const centroidOffset = nearestPartition * d;
    const nearestResidual = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      nearestResidual[i] = queryVector[i] - this.userCentroids[centroidOffset + i];
    }
    // Encode the residual
    const artifactPqCode = await pqDistance.encode(nearestResidual);

    // Step 2: Search within each partition and collect all distances
    const allCandidates: Array<{ index: number; distance: number }> = [];

    for (const partitionId of partitionIds) {
      const partition = this.userPartitions[partitionId];
      if (!partition) {
        throw new Error(`Partition ${partitionId} not found in user index`);
      }
      if (partition.size === 0) {
        throw new Error(`Partition ${partitionId} is empty`);
      }

      // Compute query residual for this partition
      const centroidOffset = partitionId * d;
      const queryResidual = new Float32Array(d);
      for (let i = 0; i < d; i++) {
        queryResidual[i] = queryVector[i] - this.userCentroids[centroidOffset + i];
      }

      // Use PQ distance to get distances to all vectors in this partition
      const partitionResults = await pqDistance.search(
        queryResidual,
        partition.pqCodes
      );

      // Convert local indices to global indices and add all distances
      for (let i = 0; i < partitionResults.indices.length; i++) {
        const localIdx = partitionResults.indices[i];
        const globalIdx = partition.cellIndices[localIdx];
        allCandidates.push({
          index: globalIdx,
          distance: partitionResults.distances[i],
        });
      }
    }

    // Step 3: Sort all candidates and return top k
    allCandidates.sort((a, b) => a.distance - b.distance);
    const topK = allCandidates.slice(0, this.k);

    const result: SearchResults = {
      indices: topK.map((c) => c.index),
      distances: topK.map((c) => c.distance),
      partitionId: nearestPartition,
      pqCode: artifactPqCode,
    };

    return result;
  }

  /**
   * Get the user index metadata
   */
  getUserIndexMetadata() {
    if (!this.userIndex) {
      return null;
    }

    return {
      id: this.userIndex.id,
      name: this.userIndex.name,
      baseModelId: this.userIndex.baseModelId,
      cellCount: this.userIndex.cellCount,
      partitionCount: Object.keys(this.userIndex.partitions).length,
    };
  }

  /**
   * Get coordinates for visualization
   */
  getCoordinates(): { x: Float32Array; y: Float32Array } | null {
    if (!this.userIndex) {
      return null;
    }

    return this.userIndex.coordinates;
  }

  /**
   * Get labels for the user index
   */
  getLabels(): string[] {
    if (!this.userIndex) {
      return [];
    }

    return this.userIndex.labels;
  }

  /**
   * Check if the system is ready
   */
  isReady(): boolean {
    return this.userIndex !== undefined && this.baseIVFPQ && this.baseIVFPQ.isReady();
  }
}
