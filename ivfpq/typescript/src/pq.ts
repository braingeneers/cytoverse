/**
 * Product Quantization (PQ) implementation for browser-side inference using ONNX models with residual vectors.
 *
 * This module provides TypeScript bindings for Product Quantization ONNX models
 * exported from the Python training pipeline. It focuses on using the trained
 * ONNX models for encoding residual embeddings and computing asymmetric distances
 * on residual vectors for improved accuracy.
 *
 * Based on the paper "Product Quantization for Nearest Neighbor Search" by Jégou et al.
 * with residual vector enhancements.
 */

import * as ort from 'onnxruntime-web'

/**
 * Product Quantization metadata structure for residual vectors.
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
  version: string // Should be "residual-1.0" for new implementation
}

/**
 * Search results interface
 */
export interface SearchResults {
  indices: number[]
  distances: number[]
}

/**
 * Product Quantization system using ONNX models for residual vectors.
 */
export class ProductQuantizer {
  public readonly metadata: PQMetadata
  
  private codebooks: Float32Array | null = null
  private encodeSession: ort.InferenceSession | null = null
  private distanceSession: ort.InferenceSession | null = null

  constructor(metadata: PQMetadata) {
    this.metadata = metadata
    
    if (metadata.d % metadata.m !== 0) {
      throw new Error(`Input dimension ${metadata.d} must be divisible by number of subquantizers ${metadata.m}`)
    }

    console.log(
      `Initialized ProductQuantizer for residual vectors: d=${metadata.d}, m=${metadata.m}, k=${metadata.k}, d_sub=${metadata.d_sub}`
    )
    console.log(`Compression ratio: ${metadata.compression_ratio}x, version: ${metadata.version}`)
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
   * Load ONNX model for encoding residual vectors to PQ codes.
   */
  async loadEncoder(modelPath: string): Promise<void> {
    try {
      this.encodeSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        logSeverityLevel: 3, // Warning level
      })
      console.log(`Loaded PQ encoder for residual vectors from ${modelPath}`)
    } catch (error) {
      console.error('Failed to load PQ encoder:', error)
      throw error
    }
  }


  /**
   * Load ONNX model for distance computation with top-k on residual vectors.
   */
  async loadDistanceModel(modelPath: string): Promise<void> {
    try {
      this.distanceSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        logSeverityLevel: 3,
      })
      console.log(`Loaded PQ distance model for residual vectors from ${modelPath}`)
    } catch (error) {
      console.error('Failed to load PQ distance model:', error)
      throw error
    }
  }

  /**
   * Encode residual vectors using the ONNX encoder model.
   *
   * @param residualVectors - Input residual vectors as Float32Array with shape [N, d]
   * @returns PQ codes as Uint8Array with shape [N, m]
   */
  async encode(residualVectors: Float32Array): Promise<Uint8Array> {
    if (!this.encodeSession) {
      throw new Error('PQ encoder model not loaded. Call loadEncoder() first.')
    }
    if (!this.codebooks) {
      throw new Error('Codebooks not loaded. Call loadCodebooks() first.')
    }

    const batchSize = residualVectors.length / this.metadata.d

    // Create input tensors
    const embeddingsTensor = new ort.Tensor('float32', residualVectors, [batchSize, this.metadata.d])
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
   * Find k nearest neighbors using the ONNX distance model with residual vectors.
   * This method is optimized for residual-based IVFPQ search.
   *
   * @param queryResidual - Query residual vector as Float32Array with shape [d]
   * @param referenceCodes - PQ codes for reference residual vectors as Uint8Array with shape [N, m]
   * @param k - Number of neighbors to return (optional, uses model default if not specified)
   * @returns Search results with indices and distances
   */
  async findNearestNeighborsResidual(
    queryResidual: Float32Array,
    referenceCodes: Uint8Array,
    k?: number
  ): Promise<SearchResults> {
    if (!this.distanceSession) {
      throw new Error('PQ distance model not loaded. Call loadDistanceModel() first.')
    }
    if (!this.codebooks) {
      throw new Error('Codebooks not loaded. Call loadCodebooks() first.')
    }

    if (queryResidual.length !== this.metadata.d) {
      throw new Error(
        `Query residual dimension ${queryResidual.length} doesn't match expected ${this.metadata.d}`
      )
    }

    const numReferences = referenceCodes.length / this.metadata.m

    // Convert reference codes to int64 array for ONNX
    const int64RefCodes = new Array(referenceCodes.length)
    for (let i = 0; i < referenceCodes.length; i++) {
      int64RefCodes[i] = referenceCodes[i]
    }

    // Create input tensors
    const queryTensor = new ort.Tensor('float32', queryResidual, [this.metadata.d])
    const refCodesTensor = new ort.Tensor('int64', int64RefCodes, [numReferences, this.metadata.m])
    const codebooksTensor = new ort.Tensor('float32', this.codebooks, [
      this.metadata.m, 
      this.metadata.k, 
      this.metadata.d_sub
    ])

    // Run inference - the ONNX model handles top-k internally
    const outputs = await this.distanceSession.run({
      query_residual: queryTensor,
      reference_codes: refCodesTensor,
      codebooks: codebooksTensor
    })

    // Extract results
    const indices = outputs.indices.data as BigInt64Array
    const distances = outputs.distances.data as Float32Array

    const resultIndices = new Array(indices.length)
    for (let i = 0; i < indices.length; i++) {
      resultIndices[i] = Number(indices[i])
    }

    const resultDistances = Array.from(distances)

    // If k is specified and different from model output, truncate results
    if (k !== undefined && k < resultIndices.length) {
      return {
        indices: resultIndices.slice(0, k),
        distances: resultDistances.slice(0, k)
      }
    }

    return {
      indices: resultIndices,
      distances: resultDistances
    }
  }

  /**
   * Legacy method for compatibility - redirects to residual-based search
   * @deprecated Use findNearestNeighborsResidual instead
   */
  async findNearestNeighbors(
    queryVector: Float32Array,
    referenceCodes: Uint8Array
  ): Promise<number[]> {
    console.warn('findNearestNeighbors is deprecated. Use findNearestNeighborsResidual for residual-based search.')
    
    const results = await this.findNearestNeighborsResidual(queryVector, referenceCodes)
    return results.indices
  }


  /**
   * Check if this PQ instance supports residual vectors
   */
  isResidualVersion(): boolean {
    return this.metadata.version === 'residual-1.0'
  }

  /**
   * Get codebooks as Float32Array (for debugging/inspection)
   */
  getCodebooks(): Float32Array | null {
    return this.codebooks
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

  const metadata = await response.json()
  
  // Validate that this is a residual-based PQ model
  if (metadata.version && metadata.version !== 'residual-1.0') {
    console.warn(`Loading PQ model with version ${metadata.version}, expected residual-1.0`)
  }

  return metadata
}

/**
 * Load complete PQ model (metadata + codebooks + ONNX models) for residual vectors.
 */
export async function loadPQModel(
  basePath: string, 
  options: {
    loadEncoder?: boolean
    loadDistance?: boolean
  } = {}
): Promise<ProductQuantizer> {
  // Default to loading all models
  const {
    loadEncoder = true,
    loadDistance = true
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

  if (loadDistance) {
    await pq.loadDistanceModel(`${basePath}/pq_distance.onnx`)
  }

  console.log(`Loaded PQ model for residual vectors from ${basePath}:`)
  console.log(`  d=${metadata.d}, m=${metadata.m}, k=${metadata.k}, d_sub=${metadata.d_sub}`)
  console.log(`  version=${metadata.version}, compression=${metadata.compression_ratio}x`)

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
 * Utility function to create test residual vectors for PQ models.
 */
export function generateTestResidualVectors(numVectors: number, dimension: number, seed?: number): Float32Array {
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
    // Generate residual-like vectors (zero-centered with moderate variance)
    vectors[i] = (rng() - 0.5) * 0.8 // Random values between -0.4 and 0.4
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

/**
 * Create an integrated IVFPQ search system
 */
export async function createIVFPQSystem(basePath: string): Promise<{
  ivf: any, // InvertedFileIndex
  pq: ProductQuantizer
}> {
  // Load PQ model
  const pq = await loadPQModel(basePath)
  
  // Return both components for integrated search
  // Note: IVF should be imported from the ivf module
  return { ivf: null, pq } // IVF will be loaded separately
}