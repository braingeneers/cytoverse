/**
 * Unit tests for IVFPQ (Inverted File with Product Quantization) system.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as ort from 'onnxruntime-web';
import { IVFPQ, createIVFPQ } from '../../src/ivfpq';

// Mock onnxruntime-web
vi.mock('onnxruntime-web', () => ({
  InferenceSession: {
    create: vi.fn(),
  },
  Tensor: vi.fn((type: string, data: any, dims: number[]) => ({
    type,
    data,
    dims,
  })),
}));

// Mock fetch for loading model artifacts
global.fetch = vi.fn();

describe.each([
  { useWebGPU: false },
  { useWebGPU: true },
])('IVFPQ System (useWebGPU: $useWebGPU)', ({ useWebGPU }) => {
  let mockFetch: any;
  let mockSession: any;

  beforeEach(() => {
    mockFetch = global.fetch as any;
    mockSession = {
      run: vi.fn(),
    };

    // Reset all mocks
    vi.clearAllMocks();

    // Setup ONNX session mock
    (ort.InferenceSession.create as any).mockResolvedValue(mockSession);
  });

  describe('IVFPQ Loading', () => {
    it('should load complete IVFPQ system from HTTP path', async () => {
      // Mock IVF metadata
      const mockIVFMetadata = {
        d: 128,
        n_partitions: 4,
        pq_m: 16,
        pq_k: 256,
        version: 'residual-1.1',
      };

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
        version: 'residual-1.0',
      };

      // Mock IVF centroids binary
      const centroidsBuffer = new ArrayBuffer(8 + 4 * 128 * 4);
      const centroidsView = new DataView(centroidsBuffer);
      centroidsView.setUint32(0, 4, true); // n_partitions
      centroidsView.setUint32(4, 128, true); // d

      // Mock PQ codebooks binary (raw float32 array without header)
      const codebooksBuffer = new ArrayBuffer(16 * 256 * 8 * 4);

      // Setup fetch responses
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata),
          });
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata),
          });
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer),
          });
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer),
          });
        }
        if (url.endsWith('/ivf_coarse.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true });
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`));
      });

      const ivfpq = new IVFPQ('http://example.com/models');
      await ivfpq.load(useWebGPU);

      expect(ivfpq.isReady()).toBe(true);

      const metadata = ivfpq.getMetadata();
      expect(metadata).not.toBeNull();
      expect(metadata!.ivf.n_partitions).toBe(4);
      expect(metadata!.ivf.d).toBe(128);
      expect(metadata!.pq.m).toBe(16);
      expect(metadata!.pq.k).toBe(256);
    });

    it('should throw error if metadata fetch fails', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const ivfpq = new IVFPQ('http://example.com/models');
      await expect(ivfpq.load(false)).rejects.toThrow('Network error');
    });
  });

  describe('IVFPQ Search', () => {
    it('should perform search with loaded system', async () => {
      // Setup complete IVFPQ system
      const mockIVFMetadata = {
        d: 4,
        n_partitions: 2,
        pq_m: 2,
        pq_k: 4,
        version: 'residual-1.1',
      };

      const mockPQMetadata = {
        d: 4,
        m: 2,
        k: 4,
        d_sub: 2,
        compression_ratio: 4,
        codebooks_shape: [2, 4, 2],
        codebooks_size: 2 * 4 * 2,
        training_samples: 100,
        version: 'residual-1.0',
      };

      // Create binary data for centroids
      const centroidsBuffer = new ArrayBuffer(8 + 2 * 4 * 4);
      const centroidsView = new DataView(centroidsBuffer);
      centroidsView.setUint32(0, 2, true); // n_partitions
      centroidsView.setUint32(4, 4, true); // d
      const centroidsData = new Float32Array(centroidsBuffer, 8);
      centroidsData.set([0, 0, 0, 0, 1, 1, 1, 1]); // Two centroids

      // Create binary data for codebooks (raw float32 array without header)
      const codebooksBuffer = new ArrayBuffer(2 * 4 * 2 * 4);

      // Create partition data
      const partitionBuffer = new ArrayBuffer(8 + 3 * (4 + 2));
      const partitionView = new DataView(partitionBuffer);
      partitionView.setUint32(0, 3, true); // num_vectors
      partitionView.setUint32(4, 2, true); // m
      let offset = 8;
      // Vector 0
      partitionView.setInt32(offset, 0, true);
      offset += 4;
      partitionView.setUint8(offset, 0);
      offset += 1;
      partitionView.setUint8(offset, 1);
      offset += 1;
      // Vector 1
      partitionView.setInt32(offset, 1, true);
      offset += 4;
      partitionView.setUint8(offset, 1);
      offset += 1;
      partitionView.setUint8(offset, 0);
      offset += 1;
      // Vector 2
      partitionView.setInt32(offset, 2, true);
      offset += 4;
      partitionView.setUint8(offset, 2);
      offset += 1;
      partitionView.setUint8(offset, 2);
      offset += 1;

      // Setup fetch mock
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata),
          });
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata),
          });
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer),
          });
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer),
          });
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/partitions/')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(partitionBuffer),
          });
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`));
      });

      // Mock ONNX session run for IVF forward, encode and distance operations
      mockSession.run.mockImplementation((inputs: any) => {
        // Check which operation based on inputs
        if (inputs.query_vector && inputs.partition_centroids) {
          // This is the IVF forward operation - return only one partition since n_probe=1
          return Promise.resolve({
            top_k_indices: { data: new Int32Array([0]) },
          });
        } else if (inputs.vectors) {
          // This is the encode operation
          return Promise.resolve({
            codes: { data: new Uint8Array([0, 1]) }, // m=2, so 2 codes
          });
        } else {
          // This is the distance operation - return 2 results since k=2
          return Promise.resolve({
            indices: { data: new Int32Array([0, 1]) },
            distances: { data: new Float32Array([0.1, 0.2]) },
          });
        }
      });

      const ivfpq = new IVFPQ('http://example.com/models', 1, 2);
      await ivfpq.load(useWebGPU);

      const queryVector = new Float32Array([0.1, 0.1, 0.1, 0.1]);
      const results = await ivfpq.search(queryVector);

      expect(results.indices).toHaveLength(2);
      expect(results.distances).toHaveLength(2);
      expect(results.indices[0]).toBe(0);
      expect(results.indices[1]).toBe(1);
    });

    it('should handle empty partitions gracefully', async () => {
      // Setup basic mocks
      const mockIVFMetadata = {
        d: 4,
        n_partitions: 2,
        pq_m: 2,
        pq_k: 4,
        version: 'residual-1.1',
      };

      const mockPQMetadata = {
        d: 4,
        m: 2,
        k: 4,
        d_sub: 2,
        compression_ratio: 4,
        codebooks_shape: [2, 4, 2],
        codebooks_size: 2 * 4 * 2,
        training_samples: 100,
        version: 'residual-1.0',
      };

      // Create empty partition
      const emptyPartitionBuffer = new ArrayBuffer(8);
      const emptyPartitionView = new DataView(emptyPartitionBuffer);
      emptyPartitionView.setUint32(0, 0, true); // num_vectors = 0
      emptyPartitionView.setUint32(4, 2, true); // m

      // Create binary data for centroids
      const centroidsBuffer = new ArrayBuffer(8 + 2 * 4 * 4);
      const centroidsView = new DataView(centroidsBuffer);
      centroidsView.setUint32(0, 2, true);
      centroidsView.setUint32(4, 4, true);

      // Create binary data for codebooks (raw float32 array without header)
      const codebooksBuffer = new ArrayBuffer(2 * 4 * 2 * 4);

      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata),
          });
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata),
          });
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer),
          });
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer),
          });
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/partitions/')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(emptyPartitionBuffer),
          });
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`));
      });

      // Mock ONNX session run for IVF forward, encode and distance operations
      mockSession.run.mockImplementation((inputs: any) => {
        // Check which operation based on inputs
        if (inputs.query_vector && inputs.partition_centroids) {
          // This is the IVF forward operation
          return Promise.resolve({
            top_k_indices: { data: new Int32Array([0]) },
          });
        } else if (inputs.vectors) {
          // This is the encode operation
          return Promise.resolve({
            codes: { data: new Uint8Array([0, 1]) }, // m=2, so 2 codes
          });
        } else {
          // This is the distance operation - return empty results for empty partitions
          return Promise.resolve({
            indices: { data: new Int16Array([]) },
            distances: { data: new Float32Array([]) },
          });
        }
      });

      const ivfpq = new IVFPQ('http://example.com/models', 1, 5);
      await ivfpq.load(useWebGPU);

      const queryVector = new Float32Array([0.1, 0.1, 0.1, 0.1]);
      const results = await ivfpq.search(queryVector);

      expect(results.indices).toHaveLength(0);
      expect(results.distances).toHaveLength(0);
    });

    it('should use ONNX model for finding nearest partitions when useONNX is true', async () => {
      // Setup complete IVFPQ system
      const mockIVFMetadata = {
        d: 4,
        n_partitions: 3,
        pq_m: 2,
        pq_k: 4,
        version: 'residual-1.1',
      };

      const mockPQMetadata = {
        d: 4,
        m: 2,
        k: 4,
        d_sub: 2,
        compression_ratio: 4,
        codebooks_shape: [2, 4, 2],
        codebooks_size: 2 * 4 * 2,
        training_samples: 100,
        version: 'residual-1.0',
      };

      // Create binary data for centroids
      const centroidsBuffer = new ArrayBuffer(8 + 3 * 4 * 4);
      const centroidsView = new DataView(centroidsBuffer);
      centroidsView.setUint32(0, 3, true); // n_partitions
      centroidsView.setUint32(4, 4, true); // d
      const centroidsData = new Float32Array(centroidsBuffer, 8);
      // Three centroids: [0,0,0,0], [1,1,1,1], [2,2,2,2]
      centroidsData.set([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);

      // Create binary data for codebooks
      const codebooksBuffer = new ArrayBuffer(2 * 4 * 2 * 4);

      // Create partition data
      const partitionBuffer = new ArrayBuffer(8 + 2 * (4 + 2));
      const partitionView = new DataView(partitionBuffer);
      partitionView.setUint32(0, 2, true); // num_vectors
      partitionView.setUint32(4, 2, true); // m
      let offset = 8;
      // Vector 0
      partitionView.setInt32(offset, 10, true);
      offset += 4;
      partitionView.setUint8(offset, 0);
      offset += 1;
      partitionView.setUint8(offset, 1);
      offset += 1;
      // Vector 1
      partitionView.setInt32(offset, 11, true);
      offset += 4;
      partitionView.setUint8(offset, 1);
      offset += 1;
      partitionView.setUint8(offset, 0);
      offset += 1;

      // Setup fetch mock
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata),
          });
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata),
          });
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer),
          });
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer),
          });
        }
        if (url.endsWith('/ivf_coarse.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/partitions/')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(partitionBuffer),
          });
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`));
      });

      // Track which session is being called
      let forwardSessionCallCount = 0;
      let distanceSessionCallCount = 0;
      let encodeSessionCallCount = 0;

      // Create separate mock sessions for each model
      const mockForwardSession = {
        run: vi.fn().mockImplementation(() => {
          forwardSessionCallCount++;
          // Return the nearest partition indices for the IVF forward model
          // For a query closest to [1,1,1,1], partitions should be [1, 0, 2] in order
          return Promise.resolve({
            top_k_indices: { data: new Int32Array([1, 0]) }, // Top 2 partitions
          });
        }),
      };

      const mockDistanceSession = {
        run: vi.fn().mockImplementation(() => {
          distanceSessionCallCount++;
          return Promise.resolve({
            indices: { data: new Int32Array([0, 1]) },
            distances: { data: new Float32Array([0.2, 0.3]) },
          });
        }),
      };

      const mockEncodeSession = {
        run: vi.fn().mockImplementation(() => {
          encodeSessionCallCount++;
          return Promise.resolve({
            codes: { data: new Uint8Array([0, 1]) }, // m=2, so 2 codes
          });
        }),
      };

      // Mock ONNX session creation to return different sessions for different models
      (ort.InferenceSession.create as any).mockImplementation((path: string) => {
        if (path.includes('ivf_coarse.onnx')) {
          return Promise.resolve(mockForwardSession);
        } else if (path.includes('pq_distance.onnx')) {
          return Promise.resolve(mockDistanceSession);
        } else if (path.includes('pq_encode.onnx')) {
          return Promise.resolve(mockEncodeSession);
        }
        return Promise.resolve(mockSession);
      });

      const ivfpq = new IVFPQ('http://example.com/models', 2, 3);
      await ivfpq.load(useWebGPU);

      // Test with ONNX model (always used)
      const queryVector = new Float32Array([0.9, 0.9, 0.9, 0.9]); // Close to [1,1,1,1]
      const results = await ivfpq.search(queryVector);

      // Verify that the IVF forward ONNX model was called
      expect(mockForwardSession.run).toHaveBeenCalled();
      expect(forwardSessionCallCount).toBe(1);

      // Verify results
      expect(results.indices).toHaveLength(3);
      expect(results.distances).toHaveLength(3);
      expect(results.partitionId).toBe(1); // First partition returned by ONNX model
      expect(results.pqCode).toBeInstanceOf(Uint8Array);
      expect(results.pqCode).toHaveLength(2); // m=2
    });

    it('should always use ONNX implementation for partition finding', async () => {
      // Setup minimal IVFPQ system
      const mockIVFMetadata = {
        d: 4,
        n_partitions: 2,
        pq_m: 2,
        pq_k: 4,
        version: 'residual-1.1',
      };

      const mockPQMetadata = {
        d: 4,
        m: 2,
        k: 4,
        d_sub: 2,
        compression_ratio: 4,
        codebooks_shape: [2, 4, 2],
        codebooks_size: 2 * 4 * 2,
        training_samples: 100,
        version: 'residual-1.0',
      };

      // Create binary data for centroids
      const centroidsBuffer = new ArrayBuffer(8 + 2 * 4 * 4);
      const centroidsView = new DataView(centroidsBuffer);
      centroidsView.setUint32(0, 2, true); // n_partitions
      centroidsView.setUint32(4, 4, true); // d
      const centroidsData = new Float32Array(centroidsBuffer, 8);
      centroidsData.set([0, 0, 0, 0, 1, 1, 1, 1]); // Two centroids

      // Create binary data for codebooks
      const codebooksBuffer = new ArrayBuffer(2 * 4 * 2 * 4);

      // Create partition data
      const partitionBuffer = new ArrayBuffer(8 + 1 * (4 + 2));
      const partitionView = new DataView(partitionBuffer);
      partitionView.setUint32(0, 1, true); // num_vectors
      partitionView.setUint32(4, 2, true); // m
      let offset = 8;
      partitionView.setInt32(offset, 5, true);
      offset += 4;
      partitionView.setUint8(offset, 0);
      offset += 1;
      partitionView.setUint8(offset, 1);
      offset += 1;

      // Setup fetch mock
      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata),
          });
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata),
          });
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer),
          });
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer),
          });
        }
        if (url.endsWith('/ivf_coarse.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/partitions/')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(partitionBuffer),
          });
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`));
      });

      // Create a mock for the forward ONNX session
      const mockForwardSession = {
        run: vi.fn().mockResolvedValue({
          top_k_indices: { data: new Int32Array([0]) },
        }),
      };

      // Mock ONNX session creation
      (ort.InferenceSession.create as any).mockImplementation((path: string) => {
        if (path.includes('ivf_coarse.onnx')) {
          return Promise.resolve(mockForwardSession);
        }
        // Return the default mock session for other models
        mockSession.run.mockImplementation((inputs: any) => {
          if (inputs.vectors) {
            // Encode operation
            return Promise.resolve({
              codes: { data: new Uint8Array([0, 1]) },
            });
          } else {
            // Distance operation
            return Promise.resolve({
              indices: { data: new Int32Array([0]) },
              distances: { data: new Float32Array([0.1]) },
            });
          }
        });
        return Promise.resolve(mockSession);
      });

      const ivfpq = new IVFPQ('http://example.com/models', 1, 1);
      await ivfpq.load(useWebGPU);

      // Test that ONNX model is always used
      const queryVector = new Float32Array([0.1, 0.1, 0.1, 0.1]); // Close to [0,0,0,0]
      const results = await ivfpq.search(queryVector);

      // Verify that the IVF forward ONNX model was called (since we don't have CPU fallback anymore)
      expect(mockForwardSession.run).toHaveBeenCalled();

      // Verify results
      expect(results.indices).toHaveLength(1);
      expect(results.distances).toHaveLength(1);
      expect(results.partitionId).toBe(0); // Should find partition 0 as nearest to [0.1,0.1,0.1,0.1]
    });
  });

  describe('Factory Functions', () => {
    it('should create IVFPQ using factory function', async () => {
      // Mock minimal required responses
      const mockIVFMetadata = {
        d: 4,
        n_partitions: 1,
        pq_m: 2,
        pq_k: 4,
        version: 'residual-1.1',
      };

      const mockPQMetadata = {
        d: 4,
        m: 2,
        k: 4,
        d_sub: 2,
        compression_ratio: 4,
        codebooks_shape: [2, 4, 2],
        codebooks_size: 2 * 4 * 2,
        training_samples: 100,
        version: 'residual-1.0',
      };

      const centroidsBuffer = new ArrayBuffer(8 + 1 * 4 * 4);
      const centroidsView = new DataView(centroidsBuffer);
      centroidsView.setUint32(0, 1, true);
      centroidsView.setUint32(4, 4, true);

      const codebooksBuffer = new ArrayBuffer(2 * 4 * 2 * 4);

      mockFetch.mockImplementation((url: string) => {
        if (url.endsWith('/ivf_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockIVFMetadata),
          });
        }
        if (url.endsWith('/pq_metadata.json')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPQMetadata),
          });
        }
        if (url.endsWith('/ivf_centroids.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(centroidsBuffer),
          });
        }
        if (url.endsWith('/pq_codebooks.bin')) {
          return Promise.resolve({
            ok: true,
            arrayBuffer: () => Promise.resolve(codebooksBuffer),
          });
        }
        if (url.endsWith('/ivf_coarse.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_distance.onnx')) {
          return Promise.resolve({ ok: true });
        }
        if (url.endsWith('/pq_encode.onnx')) {
          return Promise.resolve({ ok: true });
        }
        return Promise.reject(new Error(`Unexpected URL: ${url}`));
      });

      const ivfpq = await createIVFPQ(useWebGPU, 'http://example.com/models');

      expect(ivfpq).toBeInstanceOf(IVFPQ);
      expect(ivfpq.isReady()).toBe(true);
    });
  });
});
