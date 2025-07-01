/**
 * Product Quantization (PQ) implementation for browser-side inference.
 *
 * This module provides TypeScript implementations for Product Quantization
 * decoding and distance computation, designed to work with ONNX.js models
 * exported from the Python training pipeline.
 *
 * Based on the paper "Product Quantization for Nearest Neighbor Search" by Jégou et al.
 */

import * as ort from 'onnxruntime-web'

/**
 * Product Quantization configuration and runtime.
 */
export class ProductQuantizer {
  public d: number // Input vector dimension
  public m: number // Number of subquantizers
  public k: number // Number of centroids per subquantizer
  private d_sub: number // Dimension of each subspace (d/m)

  private codebooks: Float32Array | null = null // Codebooks: [m, k, d_sub]
  private session: ort.InferenceSession | null = null

  constructor(d: number, m: number, k: number) {
    if (d % m !== 0) {
      throw new Error(`Input dimension ${d} must be divisible by number of subquantizers ${m}`)
    }

    this.d = d
    this.m = m
    this.k = k
    this.d_sub = d / m

    console.log(
      `Initialized ProductQuantizer: d=${this.d}, m=${this.m}, k=${this.k}, d_sub=${this.d_sub}`
    )
    console.log(`Compression ratio: ${(this.d * 32) / (this.m * 8)}x`)
  }

  /**
   * Load ONNX model for encoding vectors to PQ codes.
   */
  async loadEncoder(modelPath: string): Promise<void> {
    try {
      this.session = await ort.InferenceSession.create(modelPath, {
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
   * Load codebooks from a binary file or typed array.
   * Expected format: Float32Array with shape [m, k, d_sub] flattened.
   */
  loadCodebooks(codebooks: Float32Array): void {
    const expectedSize = this.m * this.k * this.d_sub
    if (codebooks.length !== expectedSize) {
      throw new Error(`Codebook size mismatch: expected ${expectedSize}, got ${codebooks.length}`)
    }
    this.codebooks = codebooks
    console.log(`Loaded codebooks with shape [${this.m}, ${this.k}, ${this.d_sub}]`)
  }

  /**
   * Encode vectors using the ONNX model.
   *
   * @param vectors - Input vectors as Float32Array with shape [N, d]
   * @returns PQ codes as Uint8Array with shape [N, m]
   */
  async encode(vectors: Float32Array): Promise<Uint8Array> {
    if (!this.session) {
      throw new Error('PQ encoder model not loaded. Call loadEncoder() first.')
    }

    const batchSize = vectors.length / this.d

    // Create input tensor
    const inputTensor = new ort.Tensor('float32', vectors, [batchSize, this.d])

    // Run inference
    const feeds = { input: inputTensor }
    const outputs = await this.session.run(feeds)

    // Extract codes (should be int64, but we'll convert to uint8)
    const codes = outputs.output.data as BigInt64Array

    // Convert to Uint8Array (assuming codes are in valid range [0, 255])
    const uint8Codes = new Uint8Array(codes.length)
    for (let i = 0; i < codes.length; i++) {
      uint8Codes[i] = Number(codes[i])
    }

    return uint8Codes
  }

  /**
   * Decode PQ codes back to approximate vectors.
   *
   * @param codes - PQ codes as Uint8Array with shape [N, m]
   * @returns Decoded vectors as Float32Array with shape [N, d]
   */
  decode(codes: Uint8Array): Float32Array {
    if (!this.codebooks) {
      throw new Error('Codebooks not loaded. Call loadCodebooks() first.')
    }

    const batchSize = codes.length / this.m
    const decoded = new Float32Array(batchSize * this.d)

    for (let n = 0; n < batchSize; n++) {
      for (let i = 0; i < this.m; i++) {
        const codeIdx = codes[n * this.m + i]

        // Copy centroid for this subspace
        const startIdx = n * this.d + i * this.d_sub
        const centroidStartIdx = (i * this.k + codeIdx) * this.d_sub

        for (let j = 0; j < this.d_sub; j++) {
          decoded[startIdx + j] = this.codebooks[centroidStartIdx + j]
        }
      }
    }

    return decoded
  }

  /**
   * Compute squared Euclidean distance between two vectors.
   */
  static squaredDistance(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vector dimensions must match')
    }

    let sum = 0
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i]
      sum += diff * diff
    }
    return sum
  }

  /**
   * Compute asymmetric distance between a query vector and PQ codes.
   * This is more efficient than decoding and computing full distance.
   *
   * @param queryVector - Query vector as Float32Array with shape [d]
   * @param codes - PQ codes as Uint8Array with shape [N, m]
   * @returns Squared distances as Float32Array with shape [N]
   */
  asymmetricDistance(queryVector: Float32Array, codes: Uint8Array): Float32Array {
    if (!this.codebooks) {
      throw new Error('Codebooks not loaded. Call loadCodebooks() first.')
    }

    if (queryVector.length !== this.d) {
      throw new Error(
        `Query vector dimension ${queryVector.length} doesn't match expected ${this.d}`
      )
    }

    const batchSize = codes.length / this.m

    // Precompute distance tables for each subspace
    const distanceTables = new Float32Array(this.m * this.k)

    for (let i = 0; i < this.m; i++) {
      const querySubStart = i * this.d_sub
      const codebookSubStart = i * this.k * this.d_sub

      for (let j = 0; j < this.k; j++) {
        let dist = 0
        const centroidStart = codebookSubStart + j * this.d_sub

        for (let l = 0; l < this.d_sub; l++) {
          const diff = queryVector[querySubStart + l] - this.codebooks[centroidStart + l]
          dist += diff * diff
        }

        distanceTables[i * this.k + j] = dist
      }
    }

    // Compute distances for each code
    const distances = new Float32Array(batchSize)

    for (let n = 0; n < batchSize; n++) {
      let totalDist = 0

      for (let i = 0; i < this.m; i++) {
        const codeIdx = codes[n * this.m + i]
        totalDist += distanceTables[i * this.k + codeIdx]
      }

      distances[n] = totalDist
    }

    return distances
  }

  /**
   * Find k nearest neighbors using asymmetric distance.
   *
   * @param queryVector - Query vector as Float32Array with shape [d]
   * @param codes - PQ codes as Uint8Array with shape [N, m]
   * @param k - Number of nearest neighbors to return
   * @returns Object with indices and distances of k nearest neighbors
   */
  knnSearch(
    queryVector: Float32Array,
    codes: Uint8Array,
    k: number
  ): { indices: number[]; distances: number[] } {
    const distances = this.asymmetricDistance(queryVector, codes)
    const batchSize = codes.length / this.m

    // Create array of indices
    const indices = Array.from({ length: batchSize }, (_, i) => i)

    // Sort by distance and take top k
    indices.sort((a, b) => distances[a] - distances[b])

    const topK = Math.min(k, batchSize)
    const resultIndices = indices.slice(0, topK)
    const resultDistances = resultIndices.map((i) => distances[i])

    return {
      indices: resultIndices,
      distances: resultDistances,
    }
  }

  /**
   * Compute reconstruction error statistics.
   *
   * @param originalVectors - Original vectors as Float32Array with shape [N, d]
   * @param codes - PQ codes as Uint8Array with shape [N, m]
   * @returns Object with MSE and relative error statistics
   */
  computeReconstructionError(
    originalVectors: Float32Array,
    codes: Uint8Array
  ): {
    mse: number
    relativeError: number
    compressionRatio: number
  } {
    const decoded = this.decode(codes)
    const batchSize = codes.length / this.m

    let totalSquaredError = 0
    let totalOriginalSquaredNorm = 0

    for (let n = 0; n < batchSize; n++) {
      for (let i = 0; i < this.d; i++) {
        const idx = n * this.d + i
        const original = originalVectors[idx]
        const reconstructed = decoded[idx]
        const error = original - reconstructed

        totalSquaredError += error * error
        totalOriginalSquaredNorm += original * original
      }
    }

    const mse = totalSquaredError / (batchSize * this.d)
    const relativeError = Math.sqrt(totalSquaredError / totalOriginalSquaredNorm)
    const compressionRatio = (this.d * 32) / (this.m * 8)

    return {
      mse,
      relativeError,
      compressionRatio,
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
 * Load PQ model configuration from metadata JSON.
 */
export async function loadPQMetadata(metadataPath: string): Promise<{
  d: number
  m: number
  k: number
  d_sub: number
  compression_ratio: number
  codebooks_shape: number[]
  codebooks_size: number
  version: string
}> {
  const response = await fetch(metadataPath)
  if (!response.ok) {
    throw new Error(`Failed to load metadata from ${metadataPath}: ${response.statusText}`)
  }

  return await response.json()
}

/**
 * Load complete PQ model (metadata + codebooks + ONNX encoder).
 */
export async function loadPQModel(basePath: string): Promise<ProductQuantizer> {
  // Load metadata
  const metadata = await loadPQMetadata(`${basePath}/metadata.json`)

  // Create PQ instance
  const pq = new ProductQuantizer(metadata.d, metadata.m, metadata.k)

  // Load codebooks
  const codebooks = await loadCodebooksFromFile(`${basePath}/codebooks.bin`)
  pq.loadCodebooks(codebooks)

  // Load ONNX encoder
  await pq.loadEncoder(`${basePath}/pq_encoder.onnx`)

  console.log(`Loaded PQ model v${metadata.version}:`)
  console.log(`  Dimension: ${metadata.d}`)
  console.log(`  Subquantizers: ${metadata.m}`)
  console.log(`  Centroids: ${metadata.k}`)
  console.log(`  Compression: ${metadata.compression_ratio.toFixed(1)}x`)

  return pq
}

/**
 * Utility function to reshape a flat array into [N, d] format.
 */
export function reshapeVectors(flatArray: Float32Array, d: number): Float32Array {
  if (flatArray.length % d !== 0) {
    throw new Error(`Array length ${flatArray.length} is not divisible by dimension ${d}`)
  }
  return flatArray // Already flat, just return as-is with shape understanding
}
