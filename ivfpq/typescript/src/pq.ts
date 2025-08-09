/**
 * Product Quantization distance computation for residual vectors.
 * 
 * This module focuses on asymmetric distance computation between query residuals
 * and PQ-encoded database vectors using ONNX models.
 */

import * as ort from 'onnxruntime-web'

/**
 * Search results from PQ distance computation
 */
export interface PQSearchResults {
  indices: number[]
  distances: number[]
}

/**
 * PQ distance calculator using ONNX models for residual vectors
 */
export class PQDistance {
  private m: number
  private k: number
  private d_sub: number
  private distanceSession: ort.InferenceSession | null = null
  
  constructor(m: number, k: number, d_sub: number) {
    this.m = m
    this.k = k
    this.d_sub = d_sub
  }
  
  /**
   * Load ONNX model for distance computation
   */
  async loadModel(modelPath: string): Promise<void> {
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
   * Compute asymmetric distances and find k nearest neighbors
   * 
   * @param queryResidual - Query residual vector [d]
   * @param pqCodes - PQ codes for database vectors [n, m]
   * @param codebooks - PQ codebooks [m, k, d_sub]
   * @param k - Number of nearest neighbors to return
   * @returns Indices and distances of k nearest neighbors
   */
  async search(
    queryResidual: Float32Array,
    pqCodes: Uint8Array,
    codebooks: Float32Array,
    k: number
  ): Promise<PQSearchResults> {
    if (!this.distanceSession) {
      throw new Error('Distance model not loaded. Call loadModel() first.')
    }
    
    const n = pqCodes.length / this.m
    
    // Prepare tensors for ONNX model
    const queryTensor = new ort.Tensor('float32', queryResidual, [1, queryResidual.length])
    const codesTensor = new ort.Tensor('uint8', pqCodes, [n, this.m])
    const codebooksTensor = new ort.Tensor('float32', codebooks, [this.m, this.k, this.d_sub])
    const kTensor = new ort.Tensor('int64', [BigInt(k)], [1])
    
    // Run distance computation
    const outputs = await this.distanceSession.run({
      query_residual: queryTensor,
      pq_codes: codesTensor,
      codebooks: codebooksTensor,
      k: kTensor
    })
    
    // Extract results
    const indices = Array.from(outputs.indices.data as BigInt64Array).map(Number)
    const distances = Array.from(outputs.distances.data as Float32Array)
    
    return { indices, distances }
  }
  
  /**
   * Compute asymmetric distances without ONNX (CPU fallback)
   * 
   * @param queryResidual - Query residual vector [d]
   * @param pqCodes - PQ codes for database vectors [n, m]
   * @param codebooks - PQ codebooks [m, k, d_sub]
   * @param k - Number of nearest neighbors to return
   * @returns Indices and distances of k nearest neighbors
   */
  searchCPU(
    queryResidual: Float32Array,
    pqCodes: Uint8Array,
    codebooks: Float32Array,
    k: number
  ): PQSearchResults {
    const n = pqCodes.length / this.m
    
    // Precompute lookup tables for each subquantizer
    const lookupTables = new Float32Array(this.m * this.k)
    
    for (let subq = 0; subq < this.m; subq++) {
      const querySubvectorStart = subq * this.d_sub
      
      for (let centroidIdx = 0; centroidIdx < this.k; centroidIdx++) {
        let dist = 0
        const codebookOffset = (subq * this.k + centroidIdx) * this.d_sub
        
        for (let dim = 0; dim < this.d_sub; dim++) {
          const diff = queryResidual[querySubvectorStart + dim] - codebooks[codebookOffset + dim]
          dist += diff * diff
        }
        
        lookupTables[subq * this.k + centroidIdx] = dist
      }
    }
    
    // Compute distances for all database vectors
    const distances = new Float32Array(n)
    
    for (let i = 0; i < n; i++) {
      let totalDist = 0
      
      for (let subq = 0; subq < this.m; subq++) {
        const codeIdx = pqCodes[i * this.m + subq]
        totalDist += lookupTables[subq * this.k + codeIdx]
      }
      
      distances[i] = Math.sqrt(totalDist)
    }
    
    // Find top-k indices
    const indices = Array.from({length: n}, (_, i) => i)
    indices.sort((a, b) => distances[a] - distances[b])
    const topKIndices = indices.slice(0, Math.min(k, n))
    const topKDistances = topKIndices.map(i => distances[i])
    
    return {
      indices: topKIndices,
      distances: topKDistances
    }
  }
}