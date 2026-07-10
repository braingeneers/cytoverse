/**
 * IVFPQ (Inverted File with Product Quantization) for approximate vector
 * nearest neighbor search over HTTP in the browser
 *
 * - Uses pre-trained IVFPQ models from the Python pipeline to perform ANN search
 * - Combines IVF partitioning with PQ compression
 * - Supports using ONNX models for coarse centroid and distance computation
 * - Supports HTTP access for model artifacts
 * - Supports using a previously searched index as a 'user generated' index
 */

import * as ort from 'onnxruntime-web';
import { PQDistance, PQMetadata } from './pq';

/**
 * IVFPQ metadata structure matching Python output
 */
export interface IVFPQMetadata {
  d: number;
  n_partitions: number;
  pq_m: number;
  pq_k: number;
  version: string;
}

/**
 * Partition data loaded from binary files
 */
interface PartitionData {
  vector_ids: Int32Array;
  pq_codes: Uint8Array;
  size: number;
}

/**
 * Search results
 */
export interface SearchResults {
  indices: number[];
  distances: number[];
  partitionId: number; // Index of nearest centroid
  pqCode: Uint8Array; // PQ encoded residual (m bytes)
}

/**
 * IVFPQ system for ANN search using pre-trained models
 */
export class IVFPQ {
  private basePath: string;
  private n_probe: number;
  private k: number;
  private metadata: IVFPQMetadata | null = null;
  private centroids: Float32Array | null = null;
  private pqDistance: PQDistance | null = null;
  private partitionCache: Map<number, PartitionData> = new Map();
  private forwardSession: ort.InferenceSession | null = null;

  constructor(basePath: string, n_probe: number = 1, k: number = 10) {
    this.basePath = basePath;
    this.n_probe = n_probe;
    this.k = k;
  }

  /**
   * Load the IVFPQ system from artifacts
   */
  async load(useWebGPU: boolean): Promise<void> {
    console.log(`Loading IVFPQ system from ${this.basePath}`);

    // Load IVF metadata
    await this.loadMetadata();

    // Load IVF centroids
    await this.loadCentroids();

    // Load IVF forward ONNX model
    await this.loadForwardModel(useWebGPU);

    // Initialize and load PQ distance calculator
    this.pqDistance = new PQDistance(this.basePath);
    await this.pqDistance.load();

    // Load PQ distance ONNX model
    await this.pqDistance.loadModel(useWebGPU);

    // Load PQ encode model for building user indexes
    await this.pqDistance.loadEncodeModel(useWebGPU);

    console.log(
      `IVFPQ system loaded: ${this.metadata!.n_partitions} partitions, ${
        this.metadata!.d
      }D`
    );
  }

  /**
   * Load IVF metadata from JSON file
   */
  private async loadMetadata(): Promise<void> {
    const data = await this.fetchData('ivf_metadata.json', 'json');
    this.metadata = data as IVFPQMetadata;
  }

  /**
   * Load IVF centroids from binary file
   */
  private async loadCentroids(): Promise<void> {
    const buffer = (await this.fetchData(
      'ivf_centroids.bin',
      'arraybuffer'
    )) as ArrayBuffer;
    const view = new DataView(buffer);

    // Read header
    const n_partitions = view.getUint32(0, true);
    const d = view.getUint32(4, true);

    // Validate dimensions
    if (n_partitions !== this.metadata!.n_partitions || d !== this.metadata!.d) {
      throw new Error('Centroids dimensions mismatch with metadata');
    }

    // Read centroids data
    this.centroids = new Float32Array(buffer, 8);
  }

  /**
   * Load IVF forward ONNX model for finding nearest partitions
   */
  private async loadForwardModel(
    useWebGPU: boolean,
    modelPath?: string
  ): Promise<void> {
    const path = modelPath || `${this.basePath}/ivf_coarse.onnx`;
    let sessionOptions = {};
    try {
      if (useWebGPU) {
        console.log('Configuring centroid search to use WebGPU...');
        sessionOptions = {
          executionProviders: [
            {
              name: 'webgpu',
              deviceType: 'gpu',
              powerPreference: 'high-performance',
            },
            'wasm',
          ],
          graphOptimizationLevel: 'all',
        };
      } else {
        console.log('Configuring centroid search to use WebAssembly...');
        sessionOptions = {
          executionProviders: ['wasm'],
          executionMode: 'parallel',
          graphOptimizationLevel: 'all',
        };
      }
      this.forwardSession = await ort.InferenceSession.create(path, {
        sessionOptions,
      });
      console.log(`Loaded IVF forward model from ${path}`);
    } catch (error) {
      console.error('Failed to load IVF forward model:', error);
      throw error;
    }
  }

  /**
   * Load a partition from binary file
   */
  private async loadPartition(partitionId: number): Promise<PartitionData> {
    // Check cache first
    if (this.partitionCache.has(partitionId)) {
      return this.partitionCache.get(partitionId)!;
    }

    const path = `partitions/partition_${partitionId.toString().padStart(4, '0')}.bin`;
    const buffer = (await this.fetchData(path, 'arraybuffer')) as ArrayBuffer;
    const view = new DataView(buffer);

    // Read header
    const num_vectors = view.getUint32(0, true);
    const m = view.getUint32(4, true);

    // Validate m dimension
    const pqDims = this.pqDistance!.getDimensions()!;
    if (m !== pqDims.m) {
      throw new Error(`Partition m=${m} doesn't match PQ m=${pqDims.m}`);
    }

    // Read interleaved data
    const vector_ids = new Int32Array(num_vectors);
    const pq_codes = new Uint8Array(num_vectors * m);

    let offset = 8;
    for (let i = 0; i < num_vectors; i++) {
      vector_ids[i] = view.getInt32(offset, true);
      offset += 4;

      for (let j = 0; j < m; j++) {
        pq_codes[i * m + j] = view.getUint8(offset);
        offset += 1;
      }
    }

    const partitionData = {
      vector_ids,
      pq_codes,
      size: num_vectors,
    };

    // Cache the partition
    this.partitionCache.set(partitionId, partitionData);

    return partitionData;
  }

  /**
   * Find nearest partitions using ONNX model
   *
   * @param queryVector - Query vector to search with
   * @param nProbe - Number of partitions to probe
   * @returns Array of partition indices sorted by distance
   */
  async findNearestPartitions(
    queryVector: Float32Array,
    centroids: Float32Array,
    nProbe: number
  ): Promise<number[]> {
    if (!this.forwardSession) {
      throw new Error('IVF forward model not loaded. Call load() first.');
    }

    if (!centroids || !this.metadata) {
      throw new Error('IVFPQ not loaded');
    }

    const d = this.metadata.d;

    // Prepare tensors for ONNX model
    const queryTensor = new ort.Tensor('float32', queryVector, [d]);
    const centroidsTensor = new ort.Tensor('float32', centroids, [
      centroids.length / d,
      d,
    ]);
    const kTensor = new ort.Tensor('int32', Int32Array.from([nProbe]), []);

    // Run the ONNX model
    const outputs = await this.forwardSession.run({
      query_vector: queryTensor,
      partition_centroids: centroidsTensor,
      k: kTensor,
    });

    // Extract the indices from the output
    const indices = Array.from(outputs.top_k_indices.data as Int32Array);

    // console.log(`Nearest partitions (ONNX): ${indices}`);

    return indices;
  }

  /**
   * Search for k nearest neighbors with optional artifact retention
   */
  async search(queryVector: Float32Array): Promise<SearchResults> {
    if (
      !this.metadata ||
      !this.centroids ||
      !this.pqDistance ||
      !this.pqDistance.isReady()
    ) {
      throw new Error('IVFPQ system not loaded');
    }

    const d = this.metadata.d;

    if (queryVector.length !== d) {
      throw new Error(
        `Query vector dimension ${queryVector.length} doesn't match index dimension ${d}`
      );
    }

    // Step 1: Find nearest partitions (use ONNX if requested and available)
    const partitionIds = await this.findNearestPartitions(
      queryVector,
      this.centroids,
      this.n_probe
    );
    const nearestPartition = partitionIds[0]; // Closest partition for artifacts

    // Compute residual for nearest partition once (reused for encoding and search)
    const nearestCentroidOffset = nearestPartition * d;
    const nearestResidual = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      nearestResidual[i] = queryVector[i] - this.centroids[nearestCentroidOffset + i];
    }
    // Encode the residual for artifact
    const artifactPqCode = await this.pqDistance.encode(nearestResidual);

    // Step 2: Search within each partition and collect all distances
    const allCandidates: Array<{ index: number; distance: number }> = [];

    for (const partitionId of partitionIds) {
      // Load partition data
      const partition = await this.loadPartition(partitionId);

      if (partition.size === 0) {
        continue;
      }

      // Reuse nearestResidual if this is the nearest partition, otherwise compute
      let queryResidual: Float32Array;
      if (partitionId === nearestPartition) {
        queryResidual = nearestResidual;
      } else {
        const centroidOffset = partitionId * d;
        queryResidual = new Float32Array(d);
        for (let i = 0; i < d; i++) {
          queryResidual[i] = queryVector[i] - this.centroids[centroidOffset + i];
        }
      }

      // PQ codes are already Uint8Array, which is what the ONNX model expects

      // Use PQ distance to get distances to all vectors in this partition
      const partitionResults = await this.pqDistance.search(
        queryResidual,
        partition.pq_codes
      );

      // Convert local indices to global indices and add all distances
      for (let i = 0; i < partitionResults.indices.length; i++) {
        const localIdx = partitionResults.indices[i];
        const globalIdx = partition.vector_ids[localIdx];
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
   * Fetch data from HTTP
   */
  private async fetchData(
    path: string,
    type: 'json' | 'arraybuffer'
  ): Promise<unknown> {
    const fullPath = `${this.basePath}/${path}`;
    // force-cache: serve model assets from the browser cache without a revalidation round-trip
    // (they're immutable per deploy). This is what makes re-runs fast over a network — otherwise
    // each of the hundreds of probed partitions costs a 304 round-trip.
    const response = await fetch(fullPath, { cache: 'force-cache' });
    if (!response.ok) {
      throw new Error(`Failed to fetch ${fullPath}: ${response.statusText}`);
    }
    return type === 'json' ? response.json() : response.arrayBuffer();
  }

  /**
   * Get metadata
   */
  getMetadata(): { ivf: IVFPQMetadata; pq: PQMetadata } | null {
    if (!this.metadata || !this.pqDistance) {
      return null;
    }
    const pqMetadata = this.pqDistance.getMetadata();
    if (!pqMetadata) {
      return null;
    }
    return { ivf: this.metadata, pq: pqMetadata };
  }

  /**
   * Check if system is ready
   */
  isReady(): boolean {
    return !!(
      this.metadata &&
      this.centroids &&
      this.pqDistance &&
      this.pqDistance.isReady()
    );
  }

  /**
   * Get reference components for UserIVFPQ initialization
   */
  getReferenceComponents(): {
    centroids: Float32Array;
    pqDistance: PQDistance;
    d: number;
    n_partitions: number;
  } | null {
    if (!this.metadata || !this.centroids || !this.pqDistance) {
      return null;
    }
    return {
      centroids: this.centroids,
      pqDistance: this.pqDistance,
      d: this.metadata.d,
      n_partitions: this.metadata.n_partitions,
    };
  }
}

/**
 * Create and load an IVFPQ system
 */
export async function createIVFPQ(useWebGPU: boolean, basePath: string): Promise<IVFPQ> {
  const ivfpq = new IVFPQ(basePath);
  await ivfpq.load(useWebGPU);
  return ivfpq;
}
