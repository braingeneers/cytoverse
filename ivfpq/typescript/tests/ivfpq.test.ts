/**
 * Unit tests for IVFPQ (Inverted File with Product Quantization) system.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import * as ort from 'onnxruntime-web'
import { IVFPQ, createIVFPQ } from '../src/ivfpq'

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

// Mock fetch for loading model artifacts
global.fetch = vi.fn()

describe('IVFPQ System', () => {
  let mockFetch: any
  let mockSession: any
  
  beforeEach(() => {
    mockFetch = global.fetch as any
    mockSession = {
      run: vi.fn()
    }
    
    // Reset all mocks
    vi.clearAllMocks()
    
    // Setup ONNX session mock
    ;(ort.InferenceSession.create as any).mockResolvedValue(mockSession)
  })
  
  describe('IVFPQ Loading', () => {
    it('should load complete IVFPQ system from HTTP path', async () => {
      // Mock IVF metadata
      const mockIVFMetadata = {
        d: 128,
        n_partitions: 4,
        pq_m: 16,
        pq_k: 256,
        total_vectors: 1000,
        max_iterations: 100,
        inertia: 1234.5,
        partition_sizes: { '0': 250, '1': 250, '2': 250, '3': 250 },
        centroids_shape: [4, 128],
        version: 'residual-1.1'
      }
      
      // Mock PQ metadata
      const mockPQMetadata = {
        d: 128,
        m: 16,
        k: 256,
        d_sub: 8,
        compression_ratio: 32,
        codebooks_shape: [16, 256, 8],
        codebooks_size: 16 * 256 * 8,
        training_samples: 10000,
        max_iterations: 100,
        version: 'residual-1.0'
      }
      
      // Mock IVF centroids binary
      const centroidsBuffer = new ArrayBuffer(8 + 4 * 128 * 4)
      const centroidsView = new DataView(centroidsBuffer)
      centroidsView.setUint32(0, 4, true) // n_partitions
      centroidsView.setUint32(4, 128, true) // d
      
      // Mock PQ codebooks binary (raw float32 array without header)
      const codebooksBuffer = new ArrayBuffer(16 * 256 * 8 * 4)
      
      // Setup fetch responses
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata)
          })
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata)
          })
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer)
          })
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer)
          })
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true })
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true })
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`))
      })
      
      const ivfpq = new IVFPQ('http://example.com/models')
      await ivfpq.load()
      
      expect(ivfpq.isReady()).toBe(true)
      
      const metadata = ivfpq.getMetadata()
      expect(metadata).not.toBeNull()
      expect(metadata!.ivf.n_partitions).toBe(4)
      expect(metadata!.ivf.d).toBe(128)
      expect(metadata!.pq.m).toBe(16)
      expect(metadata!.pq.k).toBe(256)
    })
    
    it('should throw error if metadata fetch fails', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'))
      
      const ivfpq = new IVFPQ('http://example.com/models')
      await expect(ivfpq.load()).rejects.toThrow('Network error')
    })
  })
  
  describe('IVFPQ Search', () => {
    it('should perform search with loaded system', async () => {
      // Setup complete IVFPQ system
      const mockIVFMetadata = {
        d: 4,
        n_partitions: 2,
        pq_m: 2,
        pq_k: 4,
        total_vectors: 10,
        max_iterations: 10,
        inertia: 10.0,
        partition_sizes: { '0': 5, '1': 5 },
        centroids_shape: [2, 4],
        version: 'residual-1.1'
      }
      
      const mockPQMetadata = {
        d: 4,
        m: 2,
        k: 4,
        d_sub: 2,
        compression_ratio: 4,
        codebooks_shape: [2, 4, 2],
        codebooks_size: 2 * 4 * 2,
        training_samples: 100,
        max_iterations: 10,
        version: 'residual-1.0'
      }
      
      // Create binary data for centroids
      const centroidsBuffer = new ArrayBuffer(8 + 2 * 4 * 4)
      const centroidsView = new DataView(centroidsBuffer)
      centroidsView.setUint32(0, 2, true) // n_partitions
      centroidsView.setUint32(4, 4, true) // d
      const centroidsData = new Float32Array(centroidsBuffer, 8)
      centroidsData.set([0, 0, 0, 0, 1, 1, 1, 1]) // Two centroids
      
      // Create binary data for codebooks (raw float32 array without header)
      const codebooksBuffer = new ArrayBuffer(2 * 4 * 2 * 4)
      
      // Create partition data
      const partitionBuffer = new ArrayBuffer(8 + 3 * (4 + 2))
      const partitionView = new DataView(partitionBuffer)
      partitionView.setUint32(0, 3, true) // num_vectors
      partitionView.setUint32(4, 2, true) // m
      let offset = 8
      // Vector 0
      partitionView.setInt32(offset, 0, true); offset += 4
      partitionView.setUint8(offset, 0); offset += 1
      partitionView.setUint8(offset, 1); offset += 1
      // Vector 1
      partitionView.setInt32(offset, 1, true); offset += 4
      partitionView.setUint8(offset, 1); offset += 1
      partitionView.setUint8(offset, 0); offset += 1
      // Vector 2
      partitionView.setInt32(offset, 2, true); offset += 4
      partitionView.setUint8(offset, 2); offset += 1
      partitionView.setUint8(offset, 2); offset += 1
      
      // Setup fetch mock
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata)
          })
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata)
          })
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer)
          })
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer)
          })
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true })
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true })
        }
        if (url.includes('/partitions/partition_0000.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(partitionBuffer)
          })
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`))
      })
      
      // Mock ONNX session run for both encode and distance operations
      mockSession.run.mockImplementation(() => {
        // Check if this is an encode operation (has 'vectors' input) or distance operation
        // For simplicity, we'll return appropriate mock data for each type
        // First call will be encode, subsequent calls will be distance
        if (mockSession.run.mock.calls.length <= 1) {
          // This is the encode operation
          return Promise.resolve({
            codes: { data: new Uint8Array([0, 1]) }  // m=2, so 2 codes
          })
        } else {
          // This is the distance operation
          return Promise.resolve({
            indices: { data: new BigInt64Array([0n, 1n]) },
            distances: { data: new Float32Array([0.1, 0.2]) }
          })
        }
      })
      
      const ivfpq = new IVFPQ('http://example.com/models')
      await ivfpq.load()
      
      const queryVector = new Float32Array([0.1, 0.1, 0.1, 0.1])
      const results = await ivfpq.search(queryVector, { n_probe: 1, k: 2 })
      
      expect(results.indices).toHaveLength(2)
      expect(results.distances).toHaveLength(2)
      expect(results.indices[0]).toBe(0)
      expect(results.indices[1]).toBe(1)
    })
    
    it('should handle empty partitions gracefully', async () => {
      // Setup basic mocks
      const mockIVFMetadata = {
        d: 4,
        n_partitions: 2,
        pq_m: 2,
        pq_k: 4,
        total_vectors: 0,
        max_iterations: 10,
        inertia: 0,
        partition_sizes: { '0': 0, '1': 0 },
        centroids_shape: [2, 4],
        version: 'residual-1.1'
      }
      
      const mockPQMetadata = {
        d: 4,
        m: 2,
        k: 4,
        d_sub: 2,
        compression_ratio: 4,
        codebooks_shape: [2, 4, 2],
        codebooks_size: 2 * 4 * 2,
        training_samples: 100,
        max_iterations: 10,
        version: 'residual-1.0'
      }
      
      // Create empty partition
      const emptyPartitionBuffer = new ArrayBuffer(8)
      const emptyPartitionView = new DataView(emptyPartitionBuffer)
      emptyPartitionView.setUint32(0, 0, true) // num_vectors = 0
      emptyPartitionView.setUint32(4, 2, true) // m
      
      // Create binary data for centroids
      const centroidsBuffer = new ArrayBuffer(8 + 2 * 4 * 4)
      const centroidsView = new DataView(centroidsBuffer)
      centroidsView.setUint32(0, 2, true)
      centroidsView.setUint32(4, 4, true)
      
      // Create binary data for codebooks (raw float32 array without header)
      const codebooksBuffer = new ArrayBuffer(2 * 4 * 2 * 4)
      
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata)
          })
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata)
          })
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer)
          })
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer)
          })
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true })
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true })
        }
        if (url.includes('/partitions/')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(emptyPartitionBuffer)
          })
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`))
      })
      
      // Mock ONNX session run for both encode and distance operations
      mockSession.run.mockImplementation(() => {
        // First call will be encode, subsequent calls will be distance
        if (mockSession.run.mock.calls.length <= 1) {
          // This is the encode operation
          return Promise.resolve({
            codes: { data: new Uint8Array([0, 1]) }  // m=2, so 2 codes
          })
        } else {
          // This is the distance operation - return empty results for empty partitions
          return Promise.resolve({
            indices: { data: new BigInt64Array([]) },
            distances: { data: new Float32Array([]) }
          })
        }
      })
      
      const ivfpq = new IVFPQ('http://example.com/models')
      await ivfpq.load()
      
      const queryVector = new Float32Array([0.1, 0.1, 0.1, 0.1])
      const results = await ivfpq.search(queryVector, { n_probe: 1, k: 5 })
      
      expect(results.indices).toHaveLength(0)
      expect(results.distances).toHaveLength(0)
    })
  })
  
  describe('Factory Functions', () => {
    it('should create IVFPQ using factory function', async () => {
      // Mock minimal required responses
      const mockIVFMetadata = {
        d: 4,
        n_partitions: 1,
        pq_m: 2,
        pq_k: 4,
        total_vectors: 10,
        max_iterations: 10,
        inertia: 1.0,
        partition_sizes: { '0': 10 },
        centroids_shape: [1, 4],
        version: 'residual-1.1'
      }
      
      const mockPQMetadata = {
        d: 4,
        m: 2,
        k: 4,
        d_sub: 2,
        compression_ratio: 4,
        codebooks_shape: [2, 4, 2],
        codebooks_size: 2 * 4 * 2,
        training_samples: 100,
        max_iterations: 10,
        version: 'residual-1.0'
      }
      
      const centroidsBuffer = new ArrayBuffer(8 + 1 * 4 * 4)
      const centroidsView = new DataView(centroidsBuffer)
      centroidsView.setUint32(0, 1, true)
      centroidsView.setUint32(4, 4, true)
      
      const codebooksBuffer = new ArrayBuffer(2 * 4 * 2 * 4)
      
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata)
          })
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata)
          })
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer)
          })
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer)
          })
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true })
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true })
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`))
      })
      
      const ivfpq = await createIVFPQ('http://example.com/models')
      
      expect(ivfpq).toBeInstanceOf(IVFPQ)
      expect(ivfpq.isReady()).toBe(true)
    })
  })
})