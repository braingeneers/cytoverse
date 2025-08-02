/**
 * Product Quantization (PQ) implementation for browser-side inference using ONNX models.
 *
 * This module provides TypeScript bindings for Product Quantization ONNX models
 * exported from the Python training pipeline. It focuses on using the trained
 * ONNX models for encoding embeddings and computing asymmetric distances.
 *
 * Based on the paper "Product Quantization for Nearest Neighbor Search" by Jégou et al.
 */

import * as ort from 'onnxruntime-web'

/**
 * Product Quantization metadata structure.
 */
export interface PQMetadata {
  d: number // Input vector dimension
  m: number // Number of subquantizers
  k: number // Number of centroids per subquantizer
  d_sub: number // Dimension of each subspace (d/m)
  compression_ratio: number
  codebooks_shape: number[]
  codebooks_size: number
  training_samples: number
  max_iterations: number
  version: string
}

/**
 * Product Quantization system using ONNX models.
 */
export class ProductQuantizer {
  public readonly metadata: PQMetadata
  
  private codebooks: Float32Array | null = null
  private encodeSession: ort.InferenceSession | null = null
  private decodeSession: ort.InferenceSession | null = null
  private distanceSession: ort.InferenceSession | null = null
  private distanceBaseSession: ort.InferenceSession | null = null

  constructor(metadata: PQMetadata) {
    this.metadata = metadata
    
    if (metadata.d % metadata.m !== 0) {
      throw new Error(`Input dimension ${metadata.d} must be divisible by number of subquantizers ${metadata.m}`)
    }

    console.log(
      `Initialized ProductQuantizer: d=${metadata.d}, m=${metadata.m}, k=${metadata.k}, d_sub=${metadata.d_sub}`
    )
    console.log(`Compression ratio: ${metadata.compression_ratio}x`)
  }

  /**
   * Load codebooks from a Float32Array.
   * Expected format: Float32Array with shape [m, k, d_sub] flattened.
   */
  loadCodebooks(codebooks: Float32Array): void {
    const expectedSize = this.metadata.codebooks_size
    if (codebooks.length !== expectedSize) {
      throw new Error(`Codebook size mismatch: expected ${expectedSize}, got ${codebooks.length}`)
    }
    this.codebooks = codebooks
    console.log(`Loaded codebooks with shape [${this.metadata.m}, ${this.metadata.k}, ${this.metadata.d_sub}]`)
  }

  /**
   * Load ONNX model for encoding vectors to PQ codes.
   */
  async loadEncoder(modelPath: string): Promise<void> {
    try {
      this.encodeSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        logSeverityLevel: 3, // Warning level
      })
      console.log(`Loaded PQ encoder from ${modelPath}`)
    } catch (error) {
      console.error('Failed to load PQ encoder:', error)
      throw error
    }
  }

  /**
   * Load ONNX model for decoding PQ codes to vectors.
   */
  async loadDecoder(modelPath: string): Promise<void> {
    try {
      this.decodeSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        logSeverityLevel: 3,
      })
      console.log(`Loaded PQ decoder from ${modelPath}`)
    } catch (error) {
      console.error('Failed to load PQ decoder:', error)
      throw error
    }
  }

  /**
   * Load ONNX model for distance computation with top-k.
   */
  async loadDistanceModel(modelPath: string): Promise<void> {
    try {
      this.distanceSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        logSeverityLevel: 3,
      })
      console.log(`Loaded PQ distance model from ${modelPath}`)
    } catch (error) {
      console.error('Failed to load PQ distance model:', error)
      throw error
    }
  }

  /**
   * Load ONNX model for distance computation without top-k.
   */
  async loadDistanceBaseModel(modelPath: string): Promise<void> {
    try {
      this.distanceBaseSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        logSeverityLevel: 3,
      })
      console.log(`Loaded PQ distance base model from ${modelPath}`)
    } catch (error) {
      console.error('Failed to load PQ distance base model:', error)
      throw error
    }
  }

  /**
   * Encode vectors using the ONNX encoder model.
   *
   * @param vectors - Input vectors as Float32Array with shape [N, d]
   * @returns PQ codes as Uint8Array with shape [N, m]
   */
  async encode(vectors: Float32Array): Promise<Uint8Array> {
    if (!this.encodeSession) {
      throw new Error('PQ encoder model not loaded. Call loadEncoder() first.')
    }
    if (!this.codebooks) {
      throw new Error('Codebooks not loaded. Call loadCodebooks() first.')
    }

    const batchSize = vectors.length / this.metadata.d

    // Create input tensors
    const embeddingsTensor = new ort.Tensor('float32', vectors, [batchSize, this.metadata.d])
    const codebooksTensor = new ort.Tensor('float32', this.codebooks, [
      this.metadata.m, 
      this.metadata.k, 
      this.metadata.d_sub
    ])

    // Run inference
    const outputs = await this.encodeSession.run({
      embeddings: embeddingsTensor,
      codebooks: codebooksTensor
    })

    // Extract codes (should be int64)
    const codes = outputs.codes.data as BigInt64Array

    // Convert to Uint8Array (assuming codes are in valid range [0, k-1])
    const uint8Codes = new Uint8Array(codes.length)
    for (let i = 0; i < codes.length; i++) {
      uint8Codes[i] = Number(codes[i])
    }

    return uint8Codes
  }

  /**
   * Decode PQ codes using the ONNX decoder model.
   *
   * @param codes - PQ codes as Uint8Array with shape [N, m]
   * @returns Decoded vectors as Float32Array with shape [N, d]
   */
  async decode(codes: Uint8Array): Promise<Float32Array> {
    if (!this.decodeSession) {
      throw new Error('PQ decoder model not loaded. Call loadDecoder() first.')
    }
    if (!this.codebooks) {
      throw new Error('Codebooks not loaded. Call loadCodebooks() first.')
    }

    const batchSize = codes.length / this.metadata.m

    // Convert codes to int64 array for ONNX
    const int64Codes = new Array(codes.length)
    for (let i = 0; i < codes.length; i++) {
      int64Codes[i] = codes[i]
    }

    // Create input tensors
    const codesTensor = new ort.Tensor('int64', int64Codes, [batchSize, this.metadata.m])
    const codebooksTensor = new ort.Tensor('float32', this.codebooks, [
      this.metadata.m, 
      this.metadata.k, 
      this.metadata.d_sub
    ])

    // Run inference
    const outputs = await this.decodeSession.run({
      codes: codesTensor,
      codebooks: codebooksTensor
    })

    return outputs.embeddings.data as Float32Array
  }

  /**
   * Find k nearest neighbors using the ONNX distance model with top-k.
   *
   * @param queryVector - Query vector as Float32Array with shape [d]
   * @param referenceCodes - PQ codes for reference vectors as Uint8Array with shape [N, m]
   * @returns Indices of k nearest neighbors
   */
  async findNearestNeighbors(
    queryVector: Float32Array,
    referenceCodes: Uint8Array
  ): Promise<number[]> {
    if (!this.distanceSession) {
      throw new Error('PQ distance model not loaded. Call loadDistanceModel() first.')
    }
    if (!this.codebooks) {
      throw new Error('Codebooks not loaded. Call loadCodebooks() first.')
    }

    if (queryVector.length !== this.metadata.d) {
      throw new Error(
        `Query vector dimension ${queryVector.length} doesn't match expected ${this.metadata.d}`
      )
    }

    const numReferences = referenceCodes.length / this.metadata.m

    // Convert reference codes to int64 array for ONNX
    const int64RefCodes = new Array(referenceCodes.length)
    for (let i = 0; i < referenceCodes.length; i++) {
      int64RefCodes[i] = referenceCodes[i]
    }

    // Create input tensors
    const queryTensor = new ort.Tensor('float32', queryVector, [this.metadata.d])
    const refCodesTensor = new ort.Tensor('int64', int64RefCodes, [numReferences, this.metadata.m])
    const codebooksTensor = new ort.Tensor('float32', this.codebooks, [
      this.metadata.m, 
      this.metadata.k, 
      this.metadata.d_sub
    ])

    // Run inference
    const outputs = await this.distanceSession.run({
      query: queryTensor,
      reference_codes: refCodesTensor,
      codebooks: codebooksTensor
    })

    // Extract indices
    const indices = outputs.indices.data as BigInt64Array
    const result = new Array(indices.length)
    for (let i = 0; i < indices.length; i++) {
      result[i] = Number(indices[i])
    }

    return result
  }

  /**
   * Compute distances using the ONNX distance base model (no top-k).
   *
   * @param queryVector - Query vector as Float32Array with shape [d]
   * @param referenceCodes - PQ codes for reference vectors as Uint8Array with shape [N, m]
   * @returns Squared distances as Float32Array with shape [N]
   */
  async computeDistances(
    queryVector: Float32Array,
    referenceCodes: Uint8Array
  ): Promise<Float32Array> {
    if (!this.distanceBaseSession) {
      throw new Error('PQ distance base model not loaded. Call loadDistanceBaseModel() first.')
    }
    if (!this.codebooks) {
      throw new Error('Codebooks not loaded. Call loadCodebooks() first.')
    }

    if (queryVector.length !== this.metadata.d) {
      throw new Error(
        `Query vector dimension ${queryVector.length} doesn't match expected ${this.metadata.d}`
      )
    }

    const numReferences = referenceCodes.length / this.metadata.m

    // Convert reference codes to int64 array for ONNX
    const int64RefCodes = new Array(referenceCodes.length)
    for (let i = 0; i < referenceCodes.length; i++) {
      int64RefCodes[i] = referenceCodes[i]
    }

    // Create input tensors
    const queryTensor = new ort.Tensor('float32', queryVector, [this.metadata.d])
    const refCodesTensor = new ort.Tensor('int64', int64RefCodes, [numReferences, this.metadata.m])
    const codebooksTensor = new ort.Tensor('float32', this.codebooks, [
      this.metadata.m, 
      this.metadata.k, 
      this.metadata.d_sub
    ])

    // Run inference
    const outputs = await this.distanceBaseSession.run({
      query: queryTensor,
      reference_codes: refCodesTensor,
      codebooks: codebooksTensor
    })

    return outputs.distances.data as Float32Array
  }

  /**
   * Compute reconstruction error statistics.
   *
   * @param originalVectors - Original vectors as Float32Array with shape [N, d]
   * @param codes - PQ codes as Uint8Array with shape [N, m]
   * @returns Object with MSE and relative error statistics
   */
  async computeReconstructionError(
    originalVectors: Float32Array,
    codes: Uint8Array
  ): Promise<{
    mse: number
    relativeError: number
    compressionRatio: number
  }> {
    const decoded = await this.decode(codes)
    const batchSize = codes.length / this.metadata.m

    let totalSquaredError = 0
    let totalOriginalSquaredNorm = 0

    for (let n = 0; n < batchSize; n++) {
      for (let i = 0; i < this.metadata.d; i++) {
        const idx = n * this.metadata.d + i
        const original = originalVectors[idx]
        const reconstructed = decoded[idx]
        const error = original - reconstructed

        totalSquaredError += error * error
        totalOriginalSquaredNorm += original * original
      }
    }

    const mse = totalSquaredError / (batchSize * this.metadata.d)
    const relativeError = Math.sqrt(totalSquaredError / totalOriginalSquaredNorm)

    return {
      mse,
      relativeError,
      compressionRatio: this.metadata.compression_ratio,
    }
  }
}

/**
 * Load codebooks from a binary file.
 * This assumes the file contains a flattened Float32Array with the codebooks.
 */
export async function loadCodebooksFromFile(path: string): Promise<Float32Array> {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`Failed to load codebooks from ${path}: ${response.statusText}`)
  }

  const buffer = await response.arrayBuffer()
  return new Float32Array(buffer)
}

/**
 * Load PQ model metadata from JSON file.
 */
export async function loadPQMetadata(metadataPath: string): Promise<PQMetadata> {
  const response = await fetch(metadataPath)
  if (!response.ok) {
    throw new Error(`Failed to load metadata from ${metadataPath}: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Load complete PQ model (metadata + codebooks + ONNX models).
 */
export async function loadPQModel(
  basePath: string, 
  options: {
    loadEncoder?: boolean
    loadDecoder?: boolean
    loadDistance?: boolean
    loadDistanceBase?: boolean
  } = {}
): Promise<ProductQuantizer> {
  // Default to loading all models
  const {
    loadEncoder = true,
    loadDecoder = true,
    loadDistance = true,
    loadDistanceBase = true
  } = options

  // Load metadata
  const metadata = await loadPQMetadata(`${basePath}/pq_metadata.json`)

  // Create PQ instance
  const pq = new ProductQuantizer(metadata)

  // Load codebooks
  const codebooks = await loadCodebooksFromFile(`${basePath}/pq_codebooks.bin`)
  pq.loadCodebooks(codebooks)

  // Load ONNX models as requested
  if (loadEncoder) {
    await pq.loadEncoder(`${basePath}/pq_encode.onnx`)
  }

  if (loadDecoder) {
    await pq.loadDecoder(`${basePath}/pq_decode.onnx`)
  }

  if (loadDistance) {
    await pq.loadDistanceModel(`${basePath}/pq_distance.onnx`)
  }

  if (loadDistanceBase) {
    await pq.loadDistanceBaseModel(`${basePath}/pq_distance_base.onnx`)
  }

  console.log(`Loaded PQ model from ${basePath}:`)
  console.log(`  d=${metadata.d}, m=${metadata.m}, k=${metadata.k}, d_sub=${metadata.d_sub}`)

  return pq
}

/**
 * Utility function to validate vector dimensions.
 */
export function validateVectorDimensions(vectors: Float32Array, expectedDim: number): void {
  if (vectors.length % expectedDim !== 0) {
    throw new Error(`Array length ${vectors.length} is not divisible by dimension ${expectedDim}`)
  }
}

/**
 * Utility function to create test data for PQ models.
 */
export function generateTestVectors(numVectors: number, dimension: number, seed?: number): Float32Array {
  // Simple deterministic random number generator if seed is provided
  const rng = seed !== undefined ? (() => {
    let currentSeed = seed
    return () => {
      currentSeed = (currentSeed * 9301 + 49297) % 233280
      return currentSeed / 233280
    }
  })() : Math.random

  const vectors = new Float32Array(numVectors * dimension)
  for (let i = 0; i < vectors.length; i++) {
    vectors[i] = (rng() - 0.5) * 2 // Random values between -1 and 1
  }
  return vectors
}

/**
 * Utility function to create test PQ codes.
 */
export function generateTestCodes(numVectors: number, m: number, k: number, seed?: number): Uint8Array {
  const rng = seed !== undefined ? (() => {
    let currentSeed = seed
    return () => {
      currentSeed = (currentSeed * 9301 + 49297) % 233280
      return currentSeed / 233280
    }
  })() : Math.random

  const codes = new Uint8Array(numVectors * m)
  for (let i = 0; i < codes.length; i++) {
    codes[i] = Math.floor(rng() * k)
  }
  return codes
}
