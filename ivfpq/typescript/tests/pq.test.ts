/**
 * Unit tests for Product Quantization distance computation.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import * as ort from 'onnxruntime-web'
import { PQDistance } from '../src/pq'

// Mock onnxruntime-web
vi.mock('onnxruntime-web', () => ({
  InferenceSession: {
    create: vi.fn()
  },
  Tensor: vi.fn((type: string, data: any, dims: number[]) => ({
    type,
    data,
    dims
  }))
}))

describe('PQDistance', () => {
  let mockSession: any
  
  beforeEach(() => {
    mockSession = {
      run: vi.fn()
    }
    vi.clearAllMocks()
    ;(ort.InferenceSession.create as any).mockResolvedValue(mockSession)
  })
  
  describe('Constructor', () => {
    it('should initialize with correct parameters', () => {
      const pqDistance = new PQDistance(16, 256, 8)
      expect(pqDistance).toBeDefined()
    })
  })
  
  describe('Model Loading', () => {
    it('should load ONNX model successfully', async () => {
      const pqDistance = new PQDistance(16, 256, 8)
      await pqDistance.loadModel('http://example.com/pq_distance.onnx')
      
      expect(ort.InferenceSession.create).toHaveBeenCalledWith(
        'http://example.com/pq_distance.onnx',
        expect.objectContaining({
          executionProviders: ['wasm'],
          logSeverityLevel: 3
        })
      )
    })
    
    it('should throw error if model loading fails', async () => {
      ;(ort.InferenceSession.create as any).mockRejectedValue(new Error('Failed to load'))
      
      const pqDistance = new PQDistance(16, 256, 8)
      await expect(pqDistance.loadModel('http://example.com/pq_distance.onnx'))
        .rejects.toThrow('Failed to load')
    })
  })
  
  describe('Search with ONNX', () => {
    it('should perform search using ONNX model', async () => {
      const pqDistance = new PQDistance(2, 4, 2)
      await pqDistance.loadModel('http://example.com/pq_distance.onnx')
      
      // Mock ONNX output
      mockSession.run.mockResolvedValue({
        indices: { data: new BigInt64Array([0n, 2n, 1n]) },
        distances: { data: new Float32Array([0.1, 0.3, 0.5]) }
      })
      
      const queryResidual = new Float32Array([0.1, 0.2, 0.3, 0.4])
      const pqCodes = new Uint8Array([
        0, 1,  // Vector 0
        1, 0,  // Vector 1
        2, 2,  // Vector 2
      ])
      const codebooks = new Float32Array(2 * 4 * 2) // m * k * d_sub
      
      const results = await pqDistance.search(queryResidual, pqCodes, codebooks, 3)
      
      expect(results.indices).toEqual([0, 2, 1])
      expect(results.distances[0]).toBeCloseTo(0.1, 5)
      expect(results.distances[1]).toBeCloseTo(0.3, 5)
      expect(results.distances[2]).toBeCloseTo(0.5, 5)
      
      // Verify ONNX was called with correct tensors
      expect(mockSession.run).toHaveBeenCalledWith({
        query_residual: expect.objectContaining({
          type: 'float32',
          dims: [1, 4]
        }),
        pq_codes: expect.objectContaining({
          type: 'uint8',
          dims: [3, 2]
        }),
        codebooks: expect.objectContaining({
          type: 'float32',
          dims: [2, 4, 2]
        }),
        k: expect.objectContaining({
          type: 'int64',
          dims: [1]
        })
      })
    })
    
    it('should throw error if model not loaded', async () => {
      const pqDistance = new PQDistance(2, 4, 2)
      
      const queryResidual = new Float32Array([0.1, 0.2, 0.3, 0.4])
      const pqCodes = new Uint8Array([0, 1])
      const codebooks = new Float32Array(2 * 4 * 2)
      
      await expect(pqDistance.search(queryResidual, pqCodes, codebooks, 1))
        .rejects.toThrow('Distance model not loaded')
    })
  })
  
  describe('CPU Search Fallback', () => {
    it('should compute asymmetric distances on CPU', () => {
      const pqDistance = new PQDistance(2, 4, 2)
      
      // Create test data
      const queryResidual = new Float32Array([1, 0, 0, 1]) // d=4, m=2, d_sub=2
      
      // PQ codes for 3 vectors
      const pqCodes = new Uint8Array([
        0, 0,  // Vector 0: uses centroids [0,0] from both subquantizers
        1, 1,  // Vector 1: uses centroids [1,1]
        0, 1,  // Vector 2: uses centroids [0,1]
      ])
      
      // Codebooks: [m=2, k=4, d_sub=2]
      // Subquantizer 0 centroids (for first 2 dims of query)
      // Centroid 0: [1, 0]
      // Centroid 1: [0, 1]
      // Centroid 2: [-1, 0]
      // Centroid 3: [0, -1]
      // Subquantizer 1 centroids (for last 2 dims of query)
      // Similar pattern
      const codebooks = new Float32Array([
        // Subquantizer 0
        1, 0,   // Centroid 0
        0, 1,   // Centroid 1
        -1, 0,  // Centroid 2
        0, -1,  // Centroid 3
        // Subquantizer 1
        0, 1,   // Centroid 0
        1, 0,   // Centroid 1
        0, -1,  // Centroid 2
        -1, 0,  // Centroid 3
      ])
      
      const results = pqDistance.searchCPU(queryResidual, pqCodes, codebooks, 2)
      
      expect(results.indices).toHaveLength(2)
      expect(results.distances).toHaveLength(2)
      
      // Verify that distances are computed correctly
      // Vector 0 uses centroids [1,0] and [0,1]
      // Query residual is [1,0,0,1]
      // Distance = sqrt((1-1)^2 + (0-0)^2 + (0-0)^2 + (1-1)^2) = 0
      expect(results.distances[0]).toBeCloseTo(0, 5)
      
      // Results should be sorted by distance
      expect(results.distances[0]).toBeLessThanOrEqual(results.distances[1])
    })
    
    it('should handle k larger than number of vectors', () => {
      const pqDistance = new PQDistance(2, 4, 2)
      
      const queryResidual = new Float32Array([0.1, 0.2, 0.3, 0.4])
      const pqCodes = new Uint8Array([0, 1, 1, 0]) // 2 vectors
      const codebooks = new Float32Array(2 * 4 * 2)
      
      const results = pqDistance.searchCPU(queryResidual, pqCodes, codebooks, 10)
      
      expect(results.indices).toHaveLength(2) // Only 2 vectors available
      expect(results.distances).toHaveLength(2)
    })
    
    it('should return empty results for empty pqCodes', () => {
      const pqDistance = new PQDistance(2, 4, 2)
      
      const queryResidual = new Float32Array([0.1, 0.2, 0.3, 0.4])
      const pqCodes = new Uint8Array(0)
      const codebooks = new Float32Array(2 * 4 * 2)
      
      const results = pqDistance.searchCPU(queryResidual, pqCodes, codebooks, 5)
      
      expect(results.indices).toHaveLength(0)
      expect(results.distances).toHaveLength(0)
    })
    
    it('should compute correct distances with real-world-like data', () => {
      const pqDistance = new PQDistance(4, 16, 2)
      
      // 8-dimensional query residual
      const queryResidual = new Float32Array([
        0.5, 0.5,  // Subquantizer 0
        -0.5, 0.5, // Subquantizer 1
        0.5, -0.5, // Subquantizer 2
        -0.5, -0.5 // Subquantizer 3
      ])
      
      // PQ codes for 5 vectors
      const pqCodes = new Uint8Array([
        0, 0, 0, 0,  // Vector 0
        15, 15, 15, 15,  // Vector 1
        0, 5, 10, 15,  // Vector 2
        7, 7, 7, 7,  // Vector 3
        1, 2, 3, 4,  // Vector 4
      ])
      
      // Create codebooks with known patterns
      const codebooks = new Float32Array(4 * 16 * 2)
      for (let subq = 0; subq < 4; subq++) {
        for (let k = 0; k < 16; k++) {
          const offset = (subq * 16 + k) * 2
          // Create diverse centroids
          codebooks[offset] = Math.cos(k * Math.PI / 8)
          codebooks[offset + 1] = Math.sin(k * Math.PI / 8)
        }
      }
      
      const results = pqDistance.searchCPU(queryResidual, pqCodes, codebooks, 3)
      
      expect(results.indices).toHaveLength(3)
      expect(results.distances).toHaveLength(3)
      
      // Verify results are sorted
      for (let i = 1; i < results.distances.length; i++) {
        expect(results.distances[i]).toBeGreaterThanOrEqual(results.distances[i - 1])
      }
    })
  })
})