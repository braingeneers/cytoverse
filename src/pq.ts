/**
 * Product Quantization distance computation for residual vectors.
 *
 * This module focuses on asymmetric distance computation between query residuals
 * and PQ-encoded database vectors using ONNX models.
 */

import * as ort from 'onnxruntime-web';

// ort.env.logLevel = 'verbose';
// ort.env.debug = true;

/**
 * PQ metadata structure
 */
export interface PQMetadata {
  d: number;
  m: number;
  k: number;
  d_sub: number;
  compression_ratio: number;
  codebooks_shape: number[];
  codebooks_size: number;
  training_samples: number;
  version: string;
}

/**
 * Search results from PQ distance computation
 */
export interface PQSearchResults {
  indices: number[];
  distances: number[];
}

/**
 * PQ distance calculator using ONNX models for residual vectors
 */
export class PQDistance {
  private basePath: string;
  private useHttp: boolean;
  private metadata: PQMetadata | null = null;
  private codebooks: Float32Array | null = null;
  private codebooksTensor: ort.Tensor | null = null;
  private codebooksGpuBuffer: GPUBuffer | null = null;
  private distanceSession: ort.InferenceSession | null = null;
  private encodeSession: ort.InferenceSession | null = null;
  private useWebGPU: boolean = false;

  constructor(basePath: string) {
    this.basePath = basePath;
    this.useHttp = basePath.startsWith('http://') || basePath.startsWith('https://');
  }

  /**
   * Load PQ system from artifacts
   */
  async load(): Promise<void> {
    console.log(`Loading PQ system from ${this.basePath}`);

    // Load PQ metadata
    await this.loadMetadata();

    // Load PQ codebooks
    await this.loadCodebooks();

    // Create codebooks tensor
    this.codebooksTensor = new ort.Tensor('float32', this.codebooks!, [
      this.metadata!.m,
      this.metadata!.k,
      this.metadata!.d_sub,
    ]);

    console.log(
      `PQ system loaded: m=${this.metadata!.m}, k=${this.metadata!.k}, d_sub=${
        this.metadata!.d_sub
      }`
    );
  }

  /**
   * Load PQ metadata from JSON file
   */
  private async loadMetadata(): Promise<void> {
    const data = await this.fetchData('pq_metadata.json', 'json');
    this.metadata = data as PQMetadata;
  }

  /**
   * Load PQ codebooks from binary file
   */
  private async loadCodebooks(): Promise<void> {
    const buffer = (await this.fetchData(
      'pq_codebooks.bin',
      'arraybuffer'
    )) as ArrayBuffer;

    // The file contains raw float32 array without header
    this.codebooks = new Float32Array(buffer);

    // Validate the size matches expected dimensions
    const expectedSize = this.metadata!.m * this.metadata!.k * this.metadata!.d_sub;
    if (this.codebooks.length !== expectedSize) {
      throw new Error(
        `Codebooks size mismatch: got ${this.codebooks.length}, expected ${expectedSize} ` +
          `(m=${this.metadata!.m}, k=${this.metadata!.k}, d_sub=${
            this.metadata!.d_sub
          })`
      );
    }
  }

  /**
   * Fetch data from HTTP or file system
   */
  private async fetchData(
    path: string,
    type: 'json' | 'arraybuffer'
  ): Promise<unknown> {
    const fullPath = `${this.basePath}/${path}`;

    if (this.useHttp) {
      const response = await fetch(fullPath);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${fullPath}: ${response.statusText}`);
      }
      return type === 'json' ? response.json() : response.arrayBuffer();
    } else {
      // For Node.js file system access
      const fs = await import('fs');
      const fsPath = await import('path');
      const resolvedPath = fsPath.resolve(fullPath);

      if (type === 'json') {
        const data = await fs.promises.readFile(resolvedPath, 'utf-8');
        return JSON.parse(data);
      } else {
        const buffer = await fs.promises.readFile(resolvedPath);
        return buffer.buffer.slice(
          buffer.byteOffset,
          buffer.byteOffset + buffer.byteLength
        );
      }
    }
  }

  /**
   * Get metadata
   */
  getMetadata(): PQMetadata | null {
    return this.metadata;
  }

  /**
   * Get dimensions for compatibility
   */
  getDimensions(): { m: number; d_sub: number } | null {
    if (!this.metadata) return null;
    return { m: this.metadata.m, d_sub: this.metadata.d_sub };
  }

  /**
   * Load ONNX model for distance computation
   */
  async loadModel(useWebGPU: boolean, modelPath?: string): Promise<void> {
    const path = modelPath || `${this.basePath}/pq_distance.onnx`;
    this.useWebGPU = useWebGPU;

    // Create inference sessions
    let sessionOptions = {};
    try {
      if (useWebGPU) {
        console.log('Configuring pq distance to use WebGPU...');
        sessionOptions = {
          executionProviders: [
            {
              name: 'webgpu',
              deviceType: 'gpu',
              powerPreference: 'high-performance',
            },
          ],
          graphOptimizationLevel: 'all',
          logSeverityLevel: 0,
          logIdSeverityLevel: 3,
          logVerbosityLevel: 0,
        };
      } else {
        console.log('Configuring pq distance to use WebAssembly...');
        sessionOptions = {
          executionProviders: ['wasm'],
          executionMode: 'parallel',
          graphOptimizationLevel: 'all',
        };
      }

      this.distanceSession = await ort.InferenceSession.create(path, {
        sessionOptions,
      });
      console.log(`Loaded PQ distance model from ${path}`);

      // Create GPU tensor for codebooks if using WebGPU
      if (useWebGPU && this.codebooks && this.metadata) {
        await this.createGpuCodebooksTensor();
      }
    } catch (error) {
      console.error('Failed to load PQ distance model:', error);
      throw error;
    }
  }

  /**
   * Create GPU buffer and tensor for codebooks (WebGPU only)
   */
  private async createGpuCodebooksTensor(): Promise<void> {
    if (!this.codebooks || !this.metadata) {
      throw new Error('Codebooks not loaded');
    }

    try {
      // Get WebGPU device from ONNX Runtime
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) {
        console.warn('WebGPU adapter not available, falling back to CPU tensor');
        return;
      }

      const device = await adapter.requestDevice();
      if (!device) {
        console.warn('WebGPU device not available, falling back to CPU tensor');
        return;
      }

      // Create GPU buffer for codebooks
      const bufferSize = this.codebooks.byteLength;
      this.codebooksGpuBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });

      // Copy codebooks data to GPU
      new Float32Array(this.codebooksGpuBuffer.getMappedRange()).set(this.codebooks);
      this.codebooksGpuBuffer.unmap();

      // Create ONNX tensor from GPU buffer
      this.codebooksTensor = ort.Tensor.fromGpuBuffer(this.codebooksGpuBuffer, {
        dataType: 'float32',
        dims: [this.metadata.m, this.metadata.k, this.metadata.d_sub],
      });

      console.log('Created GPU tensor for codebooks');
    } catch (error) {
      console.warn('Failed to create GPU tensor for codebooks, using CPU tensor:', error);
      // Fallback already created in load()
    }
  }

  /**
   * Load ONNX model for PQ encoding
   */
  async loadEncodeModel(useWebGPU: boolean, modelPath?: string): Promise<void> {
    const path = modelPath || `${this.basePath}/pq_encode.onnx`;
    let sessionOptions = {};
    try {
      if (useWebGPU) {
        console.log('Configuring pq encoding to use WebGPU...');
        sessionOptions = {
          executionProviders: [
            {
              name: 'webgpu',
              deviceType: 'gpu',
              powerPreference: 'high-performance',
            },
          ],
          graphOptimizationLevel: 'all',
        };
      } else {
        console.log('Configuring pq encoding to use WebAssembly...');
        sessionOptions = {
          executionProviders: ['wasm'],
          executionMode: 'parallel',
          graphOptimizationLevel: 'all',
        };
      }
      this.encodeSession = await ort.InferenceSession.create(path, {
        sessionOptions,
      });
      console.log(`Loaded PQ encode model from ${path}`);
    } catch (error) {
      console.error('Failed to load PQ encode model:', error);
      throw error;
    }
  }

  /**
   * Encode a residual vector using Product Quantization
   *
   * @param residualVector - Residual vector to encode [d]
   * @returns PQ codes as Uint8Array [m]
   */
  async encode(residualVector: Float32Array): Promise<Uint8Array> {
    if (!this.encodeSession) {
      throw new Error('Encode model not loaded. Call loadEncodeModel() first.');
    }

    if (!this.codebooksTensor || !this.metadata) {
      throw new Error('PQ system not loaded. Call load() first.');
    }

    // Prepare tensor for ONNX model
    const residualTensor = new ort.Tensor('float32', residualVector, [
      1,
      this.metadata.d,
    ]);

    // Run encoding
    const outputs = await this.encodeSession.run(
      {
        vectors: residualTensor,
      },
      {
        // logSeverityLevel: 3,
        // logSeverityLevel: 0,
        // logVerbosityLevel: 0,
      }
    );

    if (!(outputs.codes.data instanceof Uint8Array)) {
      throw new Error('Unexpected output type for PQ codes');
    }

    return outputs.codes.data;
  }

  /**
   * Compute asymmetric distances to all vectors
   *
   * @param queryResidual - Query residual vector [d]
   * @param pqCodes - PQ codes for database vectors [n, m] as uint8 values
   * @returns Indices and distances of all vectors
   */
  async search(
    queryResidual: Float32Array,
    pqCodes: Uint8Array
  ): Promise<PQSearchResults> {
    if (!this.distanceSession) {
      throw new Error('Distance model not loaded. Call loadModel() first.');
    }

    if (!this.codebooksTensor || !this.metadata) {
      throw new Error('PQ system not loaded. Call load() first.');
    }

    const n = pqCodes.length / this.metadata.m;

    // Prepare tensors for ONNX model
    const queryTensor = new ort.Tensor('float32', queryResidual, [this.metadata.d]);
    const codesTensor = new ort.Tensor('uint8', pqCodes, [n, this.metadata.m]);

    // Run distance computation - model returns sorted distances to all vectors
    const outputs = await this.distanceSession.run({
      query_vector: queryTensor,
      reference_codes: codesTensor,
      codebooks: this.codebooksTensor,
    });

    // Extract results - model returns sorted indices and distances
    const indices = Array.from(outputs.indices.data as Int32Array);
    const distances = Array.from(outputs.distances.data as Float32Array);

    return { indices, distances };
  }

  /**
   * Compute asymmetric distances without ONNX (CPU fallback)
   *
   * @param queryResidual - Query residual vector [d]
   * @param pqCodes - PQ codes for database vectors [n, m] as int32 values
   * @returns Indices and distances of all vectors
   */
  searchCPU(queryResidual: Float32Array, pqCodes: Int32Array): PQSearchResults {
    if (!this.metadata || !this.codebooks) {
      throw new Error('PQ system not loaded. Call load() first.');
    }

    const n = pqCodes.length / this.metadata.m;
    const k = this.metadata.k;

    // Precompute lookup tables for each subquantizer
    const lookupTables = new Float32Array(this.metadata.m * k);

    for (let subq = 0; subq < this.metadata.m; subq++) {
      const querySubvectorStart = subq * this.metadata.d_sub;

      for (let centroidIdx = 0; centroidIdx < k; centroidIdx++) {
        let dist = 0;
        const codebookOffset = (subq * k + centroidIdx) * this.metadata.d_sub;

        for (let dim = 0; dim < this.metadata.d_sub; dim++) {
          const diff =
            queryResidual[querySubvectorStart + dim] -
            this.codebooks[codebookOffset + dim];
          dist += diff * diff;
        }

        lookupTables[subq * k + centroidIdx] = dist;
      }
    }

    // Compute distances for all database vectors
    const distances = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      let totalDist = 0;

      for (let subq = 0; subq < this.metadata.m; subq++) {
        const codeIdx = pqCodes[i * this.metadata.m + subq];
        totalDist += lookupTables[subq * k + codeIdx];
      }

      distances[i] = Math.sqrt(totalDist);
    }

    // Sort by distance and return all results
    const indices = Array.from({ length: n }, (_, i) => i);
    const distancesArray = Array.from(distances);

    const sortedIndices = indices
      .map((idx, i) => ({ idx, dist: distancesArray[i] }))
      .sort((a, b) => a.dist - b.dist);

    return {
      indices: sortedIndices.map((x) => x.idx),
      distances: sortedIndices.map((x) => x.dist),
    };
  }

  /**
   * Check if PQ system is ready
   */
  isReady(): boolean {
    return !!(this.metadata && this.codebooks && this.codebooksTensor);
  }

  /**
   * Cleanup GPU resources
   */
  dispose(): void {
    if (this.codebooksGpuBuffer) {
      this.codebooksGpuBuffer.destroy();
      this.codebooksGpuBuffer = null;
      console.log('Destroyed GPU buffer for codebooks');
    }
  }
}
