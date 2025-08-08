/**
 * Unit tests for integrated IVFPQ (Inverted File with Product Quantization) system.
 * 
 * Tests the complete IVFPQ pipeline including IVF partition search, PQ distance
 * computation with residual vectors, and integrated ANN search functionality.
 */

import { describe, it, expect, beforeAll, vi, beforeEach, afterEach } from 'vitest'
import * as ort from 'onnxruntime-web'
import { IVFPQ, createIVFPQ, loadIVFPQForBrowser, validateIVFPQDatabase } from '../src/ivfpq'

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
  
  afterEach(() => {
    vi.clearAllMocks()
  })
  
  describe('IVFPQ Loading', () => {
    it('should load complete IVFPQ system from base path', async () => {
      // Mock IVF centroids binary file
      const mockCentroidsBuffer = new ArrayBuffer(8 + 4 * 4 * 128) // header + 4 partitions × 128 dims
      const centroidsView = new DataView(mockCentroidsBuffer)
      centroidsView.setUint32(0, 4, true) // n_partitions
      centroidsView.setUint32(4, 128, true) // dimension
      
      // Mock metadata files
      const mockIVFMetadata = { version: 'residual-1.0' }
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
      
      // Mock PQ codebooks
      const mockCodebooksBuffer = new ArrayBuffer(mockPQMetadata.codebooks_size * 4)
      
      // Setup fetch responses
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(mockCentroidsBuffer)
          })
        }
        if (url.endsWith('/metadata.json')) {
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
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(mockCodebooksBuffer)
          })
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true })
        }
        if (url.endsWith('/ivf_search.onnx')) {
          return Promise.resolve({ ok: true })
        }
        return Promise.resolve({ ok: false })
      })
      
      const ivfpq = new IVFPQ('/models')
      await ivfpq.load()
      
      expect(ivfpq.isReady()).toBe(true)
      
      const metadata = ivfpq.getMetadata()
      expect(metadata).toBeDefined()
      expect(metadata?.ivf.n_partitions).toBe(4)
      expect(metadata?.ivf.d).toBe(128)
      expect(metadata?.pq.m).toBe(16)
      expect(metadata?.pq.k).toBe(256)
    })
    
    it('should handle missing IVF ONNX model gracefully', async () => {
      // Mock responses without IVF ONNX model
      const mockCentroidsBuffer = new ArrayBuffer(8 + 4 * 4 * 128)
      const centroidsView = new DataView(mockCentroidsBuffer)
      centroidsView.setUint32(0, 4, true)
      centroidsView.setUint32(4, 128, true)
      
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
      
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_search.onnx')) {
          return Promise.resolve({ ok: false }) // Model not found
        }
        if (url.endsWith('/centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(mockCentroidsBuffer)
          })
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata)
          })
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(new ArrayBuffer(mockPQMetadata.codebooks_size * 4))
          })
        }
        return Promise.resolve({ ok: true })
      })
      
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      
      const ivfpq = new IVFPQ('/models')
      await ivfpq.load()
      
      expect(consoleWarnSpy).toHaveBeenCalledWith(expect.stringContaining('IVF ONNX model not found'))
      expect(ivfpq.isReady()).toBe(true)
      
      consoleWarnSpy.mockRestore()
    })
    
    it('should throw error when required files are missing', async () => {
      mockFetch.mockResolvedValue({ ok: false, statusText: 'Not Found' })
      
      const ivfpq = new IVFPQ('/models')
      await expect(ivfpq.load()).rejects.toThrow('Failed to load centroids')
    })
  })
  
  describe('IVFPQ Search', () => {
    let ivfpq: IVFPQ
    
    beforeEach(async () => {
      // Setup complete mock environment for search tests
      const mockCentroidsBuffer = new ArrayBuffer(8 + 2 * 4 * 128) // 2 partitions
      const centroidsView = new DataView(mockCentroidsBuffer)
      centroidsView.setUint32(0, 2, true) // n_partitions
      centroidsView.setUint32(4, 128, true) // dimension
      
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
      
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(mockCentroidsBuffer)
          })
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata)
          })
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(new ArrayBuffer(mockPQMetadata.codebooks_size * 4))
          })
        }
        if (url.includes('/partition_')) {
          // Mock partition data
          const mockPartitionBuffer = new ArrayBuffer(12 + 100 * 4 + 100 * 16) // header + 100 vectors
          const partitionView = new DataView(mockPartitionBuffer)
          partitionView.setUint32(0, 100, true) // size
          partitionView.setUint32(4, 16, true) // m
          partitionView.setUint32(8, 100 * 4 + 100 * 16, true) // data_size
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(mockPartitionBuffer)
          })
        }
        return Promise.resolve({ ok: true })
      })
      
      ivfpq = new IVFPQ('/models')
      await ivfpq.load()
    })
    
    it('should perform search with ONNX models', async () => {
      // Mock ONNX session responses
      mockSession.run.mockImplementation((inputs: any) => {
        if (inputs.query) {
          // IVF search response
          return Promise.resolve({
            partition_ids: { data: new BigInt64Array([0n, 1n]) },
            query_residuals: { data: new Float32Array(2 * 128) }
          })
        }
        if (inputs.query_residual) {
          // PQ distance response
          const indices = new BigInt64Array([0n, 1n, 2n, 3n, 4n])
          const distances = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5])
          return Promise.resolve({
            indices: { data: indices },
            distances: { data: distances }
          })
        }
        return Promise.resolve({})
      })
      
      const queryVector = new Float32Array(128)
      queryVector.fill(0.5)
      
      const results = await ivfpq.search(queryVector, { n_probe: 2, k: 5 })
      
      expect(results).toHaveLength(1)
      expect(results[0].indices).toHaveLength(5)
      expect(results[0].distances).toHaveLength(5)
      expect(results[0].distances[0]).toBeLessThanOrEqual(results[0].distances[1])
    })
    
    it('should handle batch queries', async () => {
      mockSession.run.mockImplementation((inputs: any) => {
        if (inputs.query) {
          return Promise.resolve({
            partition_ids: { data: new BigInt64Array([0n]) },
            query_residuals: { data: new Float32Array(128) }
          })
        }
        if (inputs.query_residual) {
          return Promise.resolve({
            indices: { data: new BigInt64Array([0n, 1n]) },
            distances: { data: new Float32Array([0.1, 0.2]) }
          })
        }
        return Promise.resolve({})
      })
      
      const batchSize = 3
      const queryVectors = new Float32Array(batchSize * 128)
      
      const results = await ivfpq.search(queryVectors, { n_probe: 1, k: 2 })
      
      expect(results).toHaveLength(batchSize)
      results.forEach(result => {
        expect(result.indices).toHaveLength(2)
        expect(result.distances).toHaveLength(2)
      })
    })
    
    it('should throw error for invalid query dimensions', async () => {
      const invalidQuery = new Float32Array(100) // Wrong dimension
      
      await expect(ivfpq.search(invalidQuery, { k: 5 }))
        .rejects.toThrow('not divisible by dimension')
    })
    
    it('should handle empty partitions gracefully', async () => {
      // Override partition loading to return empty partition
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/partition_')) {
          const emptyBuffer = new ArrayBuffer(12) // Just header, no data
          const view = new DataView(emptyBuffer)
          view.setUint32(0, 0, true) // size = 0
          view.setUint32(4, 16, true) // m
          view.setUint32(8, 0, true) // data_size = 0
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(emptyBuffer)
          })
        }
        // Default response for other files
        return Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(new ArrayBuffer(100)) })
      })
      
      mockSession.run.mockImplementation(() => {
        return Promise.resolve({
          partition_ids: { data: new BigInt64Array([0n]) },
          query_residuals: { data: new Float32Array(128) }
        })
      })
      
      const queryVector = new Float32Array(128)
      const results = await ivfpq.search(queryVector, { n_probe: 1, k: 5 })
      
      expect(results[0].indices).toHaveLength(0)
      expect(results[0].distances).toHaveLength(0)
    })
  })
  
  describe('Factory Functions', () => {
    it('should create IVFPQ with factory function', async () => {
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
      
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/centroids.bin')) {
          const buffer = new ArrayBuffer(8 + 4 * 128)
          const view = new DataView(buffer)
          view.setUint32(0, 1, true)
          view.setUint32(4, 128, true)
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(buffer)
          })
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata)
          })
        }
        return Promise.resolve({
          ok: true,
          arrayBuffer: () => Promise.resolve(new ArrayBuffer(100))
        })
      })
      
      const ivfpq = await createIVFPQ('/models')
      expect(ivfpq.isReady()).toBe(true)
    })
    
    it('should load IVFPQ for browser with HTTP URLs', async () => {
      mockFetch.mockImplementation(() => {
        return Promise.resolve({
          ok: true,
          arrayBuffer: () => Promise.resolve(new ArrayBuffer(100)),
          json: () => Promise.resolve({})
        })
      })
      
      const ivfpq = await loadIVFPQForBrowser('https://example.com/models')
      expect(ivfpq).toBeDefined()
    })
    
    it('should reject non-HTTP URLs in browser context', async () => {
      await expect(loadIVFPQForBrowser('/local/path'))
        .rejects.toThrow('Browser context requires HTTP/HTTPS URLs')
    })
  })
  
  describe('Database Validation', () => {
    it('should validate complete IVFPQ database', async () => {
      mockFetch.mockImplementation((url: string) => {
        const requiredFiles = [
          'centroids.bin',
          'pq_metadata.json',
          'pq_codebooks.bin',
          'pq_distance.onnx'
        ]
        
        const filename = url.split('/').pop()
        if (requiredFiles.includes(filename || '')) {
          if (filename === 'pq_metadata.json') {
            return Promise.resolve({
              ok: true,
              json: () => Promise.resolve({ version: 'residual-1.0' })
            })
          }
          return Promise.resolve({ ok: true })
        }
        return Promise.resolve({ ok: false })
      })
      
      const isValid = await validateIVFPQDatabase('/models')
      expect(isValid).toBe(true)
    })
    
    it('should detect missing required files', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('pq_distance.onnx')) {
          return Promise.resolve({ ok: false }) // Missing file
        }
        if (url.endsWith('pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ version: 'residual-1.0' })
          })
        }
        return Promise.resolve({ ok: true })
      })
      
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      
      const isValid = await validateIVFPQDatabase('/models')
      expect(isValid).toBe(false)
      expect(consoleErrorSpy).toHaveBeenCalledWith(expect.stringContaining('Missing required file'))
      
      consoleErrorSpy.mockRestore()
    })
    
    it('should warn about incompatible PQ version', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ version: 'legacy-1.0' })
          })
        }
        return Promise.resolve({ ok: true })
      })
      
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      
      const isValid = await validateIVFPQDatabase('/models')
      expect(isValid).toBe(true)
      expect(consoleWarnSpy).toHaveBeenCalledWith(expect.stringContaining('not compatible with residual vectors'))
      
      consoleWarnSpy.mockRestore()
    })
  })
  
  describe('Error Handling', () => {
    it('should throw error when searching before loading', async () => {
      const ivfpq = new IVFPQ('/models')
      const queryVector = new Float32Array(128)
      
      await expect(ivfpq.search(queryVector))
        .rejects.toThrow('IVFPQ system not loaded')
    })
    
    it('should handle network errors gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'))
      
      const ivfpq = new IVFPQ('/models')
      await expect(ivfpq.load()).rejects.toThrow('Network error')
    })
    
    it('should handle corrupted metadata', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.reject(new Error('Invalid JSON'))
          })
        }
        return Promise.resolve({ ok: true })
      })
      
      const ivfpq = new IVFPQ('/models')
      await expect(ivfpq.load()).rejects.toThrow()
    })
  })
})