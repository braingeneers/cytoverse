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

// Mock fs for file operations
vi.mock('fs', () => ({
  promises: {
    readFile: vi.fn()
  }
}))

vi.mock('path', () => ({
  resolve: vi.fn((path: string) => path)
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
    it('should initialize with base path', () => {
      const pqDistance = new PQDistance('/path/to/model')
      expect(pqDistance).toBeDefined()
    })
    
    it('should detect HTTP paths', () => {
      const pqDistance = new PQDistance('http://example.com/model')
      expect(pqDistance).toBeDefined()
    })
  })
  
  describe('System Loading', () => {
    it('should load complete PQ system successfully', async () => {
      const mockMetadata = {
        d: 8,
        m: 2,
        k: 4,
        d_sub: 4,
        compression_ratio: 2,
        codebooks_shape: [2, 4, 4],
        codebooks_size: 32,
        training_samples: 1000,
        max_iterations: 100,
        version: '1.0'
      }
      
      const mockCodebooks = new Float32Array(32)
      
      // Mock fetch for HTTP
      global.fetch = vi.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMetadata)
        })
        .mockResolvedValueOnce({
          ok: true,
          arrayBuffer: () => Promise.resolve(mockCodebooks.buffer)
        })
      
      const pqDistance = new PQDistance('http://example.com')
      await pqDistance.load()
      
      expect(pqDistance.isReady()).toBe(true)
      expect(pqDistance.getMetadata()).toEqual(mockMetadata)
    })
  })
  
  describe('Search with ONNX', () => {
    it('should perform search using ONNX model', async () => {
      const mockMetadata = { 
        d: 4, m: 2, k: 4, d_sub: 2, compression_ratio: 2, 
        codebooks_shape: [2, 4, 2], codebooks_size: 16, 
        training_samples: 1000, max_iterations: 100, version: '1.0' 
      }
      const mockCodebooks = new Float32Array(16)
      
      global.fetch = vi.fn()
        .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockMetadata) })
        .mockResolvedValueOnce({ ok: true, arrayBuffer: () => Promise.resolve(mockCodebooks.buffer) })
      
      const pqDistance = new PQDistance('http://example.com')
      await pqDistance.load()
      await pqDistance.loadModel()
      
      // Mock ONNX output
      mockSession.run.mockResolvedValue({
        indices: { data: new Int32Array([0, 2, 1]) },
        distances: { data: new Float32Array([0.1, 0.3, 0.5]) }
      })
      
      const queryResidual = new Float32Array([0.1, 0.2, 0.3, 0.4])
      const pqCodesUint8 = new Uint8Array([
        0, 1,  // Vector 0
        1, 0,  // Vector 1
        2, 2,  // Vector 2
      ])
      const pqCodes = new Int32Array(pqCodesUint8)
      
      const results = await pqDistance.search(queryResidual, pqCodes)
      
      expect(results.indices).toEqual([0, 2, 1])
      expect(results.distances[0]).toBeCloseTo(0.1, 5)
      expect(results.distances[1]).toBeCloseTo(0.3, 5)
      expect(results.distances[2]).toBeCloseTo(0.5, 5)
    })
    
    it('should throw error if system not loaded', async () => {
      const pqDistance = new PQDistance('http://example.com')
      
      const queryResidual = new Float32Array([0.1, 0.2, 0.3, 0.4])
      const pqCodesUint8 = new Uint8Array([0, 1])
      const pqCodes = new Int32Array(pqCodesUint8)
      
      await expect(pqDistance.search(queryResidual, pqCodes))
        .rejects.toThrow('Distance model not loaded. Call loadModel() first.')
    })
  })
  
  describe('PQ Encoding with ONNX', () => {
    it('should encode residual vectors using ONNX model', async () => {
      const mockMetadata = { 
        d: 4, m: 2, k: 4, d_sub: 2, compression_ratio: 2, 
        codebooks_shape: [2, 4, 2], codebooks_size: 16, 
        training_samples: 1000, max_iterations: 100, version: '1.0' 
      }
      const mockCodebooks = new Float32Array(16)
      
      global.fetch = vi.fn()
        .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockMetadata) })
        .mockResolvedValueOnce({ ok: true, arrayBuffer: () => Promise.resolve(mockCodebooks.buffer) })
      
      const pqDistance = new PQDistance('http://example.com')
      await pqDistance.load()
      await pqDistance.loadEncodeModel()
      
      // Mock ONNX encode output
      mockSession.run.mockResolvedValue({
        codes: { data: new Uint8Array([2, 3]) }  // m=2, so 2 codes
      })
      
      const residualVector = new Float32Array([0.1, 0.2, 0.3, 0.4])
      const codes = await pqDistance.encode(residualVector)
      
      expect(codes).toBeInstanceOf(Uint8Array)
      expect(codes.length).toBe(2)  // m=2
      expect(codes[0]).toBe(2)
      expect(codes[1]).toBe(3)
      
      // Verify correct tensors were passed to ONNX
      expect(mockSession.run).toHaveBeenCalledWith({
        vectors: {
          type: 'float32',
          dims: [1, 4],
          data: residualVector
        }
      }, {})
    })
    
    it('should throw error if encode model not loaded', async () => {
      const mockMetadata = { 
        d: 4, m: 2, k: 4, d_sub: 2, compression_ratio: 2, 
        codebooks_shape: [2, 4, 2], codebooks_size: 16, 
        training_samples: 1000, max_iterations: 100, version: '1.0' 
      }
      const mockCodebooks = new Float32Array(16)
      
      global.fetch = vi.fn()
        .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockMetadata) })
        .mockResolvedValueOnce({ ok: true, arrayBuffer: () => Promise.resolve(mockCodebooks.buffer) })
      
      const pqDistance = new PQDistance('http://example.com')
      await pqDistance.load()
      
      const residualVector = new Float32Array([0.1, 0.2, 0.3, 0.4])
      
      await expect(pqDistance.encode(residualVector))
        .rejects.toThrow('Encode model not loaded. Call loadEncodeModel() first.')
    })
    
    it('should throw error if PQ system not loaded for encoding', async () => {
      const pqDistance = new PQDistance('http://example.com')
      await pqDistance.loadEncodeModel()
      
      const residualVector = new Float32Array([0.1, 0.2, 0.3, 0.4])
      
      await expect(pqDistance.encode(residualVector))
        .rejects.toThrow('PQ system not loaded. Call load() first.')
    })
  })
  
  describe('CPU Search Fallback', () => {
    it('should compute asymmetric distances on CPU', async () => {
      const mockMetadata = { 
        d: 4, m: 2, k: 4, d_sub: 2, compression_ratio: 2, 
        codebooks_shape: [2, 4, 2], codebooks_size: 32, 
        training_samples: 1000, max_iterations: 100, version: '1.0' 
      }
      
      // Create known codebooks
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
      
      global.fetch = vi.fn()
        .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockMetadata) })
        .mockResolvedValueOnce({ ok: true, arrayBuffer: () => Promise.resolve(codebooks.buffer) })
      
      const pqDistance = new PQDistance('http://example.com')
      await pqDistance.load()
      
      const queryResidual = new Float32Array([1, 0, 0, 1])
      const pqCodesUint8 = new Uint8Array([
        0, 0,  // Vector 0: uses centroids [0,0] from both subquantizers
        1, 1,  // Vector 1: uses centroids [1,1]
        0, 1,  // Vector 2: uses centroids [0,1]
      ])
      const pqCodes = new Int32Array(pqCodesUint8)
      
      const results = pqDistance.searchCPU(queryResidual, pqCodes)
      
      expect(results.indices).toHaveLength(3)
      expect(results.distances).toHaveLength(3)
      
      // Vector 0 should have distance 0 (exact match)
      expect(results.distances[0]).toBeCloseTo(0, 5)
      
      // Results should be sorted by distance
      expect(results.distances[0]).toBeLessThanOrEqual(results.distances[1])
    })
    
    it('should throw error if CPU search called without system loaded', () => {
      const pqDistance = new PQDistance('http://example.com')
      
      const queryResidual = new Float32Array([0.1, 0.2, 0.3, 0.4])
      const pqCodesUint8 = new Uint8Array([0, 1])
      const pqCodes = new Int32Array(pqCodesUint8)
      
      expect(() => pqDistance.searchCPU(queryResidual, pqCodes))
        .toThrow('PQ system not loaded')
    })
  })
})