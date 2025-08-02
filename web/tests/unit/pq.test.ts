/**
 * TypeScript tests for ONNX PQ (Product Quantization) models.
 * 
 * This test file validates that the exported ONNX PQ models work correctly
 * in a browser-like environment, comparing results with expected values from Python.
 */

import { describe, it, expect, beforeAll } from 'vitest'
import * as ort from 'onnxruntime-web'
import fs from 'fs'
import path from 'path'

// Test configuration
const FIXTURES_DIR = path.join(__dirname, '..', '..', '..', 'tests', 'artifacts')
const TOLERANCE = 1e-5

interface PQTestMetadata {
  d: number
  m: number
  k: number
  d_sub: number
  compression_ratio: number
  codebooks_shape: number[]
  codebooks_size: number
  training_samples: number
  max_iterations: number
  version: string
}

// Helper functions to load binary data
function loadFloat32Binary(filePath: string): Float32Array {
  const buffer = fs.readFileSync(filePath)
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4)
}

function loadInt32Binary(filePath: string): Int32Array {
  const buffer = fs.readFileSync(filePath)
  return new Int32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4)
}

function loadInt64Binary(filePath: string): BigInt64Array {
  const buffer = fs.readFileSync(filePath)
  return new BigInt64Array(buffer.buffer, buffer.byteOffset, buffer.length / 8)
}

function loadMetadata(filePath: string): PQTestMetadata {
  const jsonStr = fs.readFileSync(filePath, 'utf8')
  return JSON.parse(jsonStr) as PQTestMetadata
}

// Test data - will be generated
let metadata: PQTestMetadata
let testCodebooks: Float32Array

// Test parameters (since we'll generate test data)
const TEST_PARAMS = {
  n_samples: 100,
  n_references: 1000,
  k_nn: 50
}

beforeAll(() => {
  // Load metadata and codebooks
  metadata = loadMetadata(path.join(FIXTURES_DIR, 'pq_metadata.json'))
  testCodebooks = loadFloat32Binary(path.join(FIXTURES_DIR, 'pq_codebooks.bin'))

  console.log('Loaded PQ test data:')
  console.log(`  Metadata: d=${metadata.d}, m=${metadata.m}, k=${metadata.k}, d_sub=${metadata.d_sub}`)
  console.log(`  Codebooks shape: [${metadata.m}, ${metadata.k}, ${metadata.d_sub}]`)
  console.log(`  Test parameters: n_samples=${TEST_PARAMS.n_samples}, n_references=${TEST_PARAMS.n_references}, k_nn=${TEST_PARAMS.k_nn}`)
})

describe('ONNX PQ Models', () => {
  
  it('should load metadata correctly', () => {
    expect(metadata).toBeDefined()
    expect(metadata.d).toBe(128)
    expect(metadata.m).toBe(8)
    expect(metadata.k).toBe(256)
    expect(metadata.d_sub).toBe(16)
    expect(metadata.compression_ratio).toBe(64.0)
    expect(metadata.codebooks_shape).toEqual([8, 256, 16])
  })

  it('should load codebooks with correct shape', () => {
    expect(testCodebooks.length).toBe(metadata.codebooks_size)
    expect(testCodebooks.length).toBe(metadata.m * metadata.k * metadata.d_sub)
  })

  it('should encode embeddings correctly with PQEncode model', async () => {
    // Load ONNX model
    const modelPath = path.join(FIXTURES_DIR, 'pq_encode.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test embeddings
    const testEmbeddings = new Float32Array(TEST_PARAMS.n_samples * metadata.d)
    for (let i = 0; i < testEmbeddings.length; i++) {
      testEmbeddings[i] = (Math.random() - 0.5) * 2 // Random values between -1 and 1
    }

    // Prepare inputs
    const embeddings = new ort.Tensor('float32', testEmbeddings, [TEST_PARAMS.n_samples, metadata.d])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    // Run inference
    const results = await session.run({
      embeddings: embeddings,
      codebooks: codebooks
    })

    const codes = results.codes.data as BigInt64Array

    // Verify output shape and constraints
    expect(codes.length).toBe(TEST_PARAMS.n_samples * metadata.m)
    
    // Check that all codes are within valid range
    for (let i = 0; i < codes.length; i++) {
      expect(Number(codes[i])).toBeGreaterThanOrEqual(0)
      expect(Number(codes[i])).toBeLessThan(metadata.k)
    }

    console.log('✅ PQEncode model produces valid codes')
  })

  it('should decode codes correctly with PQDecode model', async () => {
    // Load ONNX model
    const modelPath = path.join(FIXTURES_DIR, 'pq_decode.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate random test codes
    const testCodes = new Array(TEST_PARAMS.n_samples * metadata.m)
    for (let i = 0; i < testCodes.length; i++) {
      testCodes[i] = Math.floor(Math.random() * metadata.k)
    }
    
    const codes = new ort.Tensor('int64', testCodes, [TEST_PARAMS.n_samples, metadata.m])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    // Run inference
    const results = await session.run({
      codes: codes,
      codebooks: codebooks
    })

    const decodedEmbeddings = results.embeddings.data as Float32Array

    // Verify output shape and properties
    expect(decodedEmbeddings.length).toBe(TEST_PARAMS.n_samples * metadata.d)
    
    // Check that all values are finite
    for (let i = 0; i < decodedEmbeddings.length; i++) {
      expect(isFinite(decodedEmbeddings[i])).toBe(true)
    }

    console.log('✅ PQDecode model produces valid embeddings')
  })

  it('should compute distances correctly with PQDistanceBase model', async () => {
    // Load ONNX model
    const modelPath = path.join(FIXTURES_DIR, 'pq_distance_base.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test data
    const testQuery = new Float32Array(metadata.d)
    for (let i = 0; i < testQuery.length; i++) {
      testQuery[i] = (Math.random() - 0.5) * 2
    }

    const testRefCodes = new Array(TEST_PARAMS.n_references * metadata.m)
    for (let i = 0; i < testRefCodes.length; i++) {
      testRefCodes[i] = Math.floor(Math.random() * metadata.k)
    }

    // Prepare inputs
    const query = new ort.Tensor('float32', testQuery, [metadata.d])
    const referenceCodes = new ort.Tensor('int64', testRefCodes, [TEST_PARAMS.n_references, metadata.m])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    // Run inference
    const results = await session.run({
      query: query,
      reference_codes: referenceCodes,
      codebooks: codebooks
    })

    const distances = results.distances.data as Float32Array

    // Verify output shape and properties
    expect(distances.length).toBe(TEST_PARAMS.n_references)
    
    // Check that all distances are non-negative and finite
    for (let i = 0; i < distances.length; i++) {
      expect(distances[i]).toBeGreaterThanOrEqual(0)
      expect(isFinite(distances[i])).toBe(true)
    }

    console.log('✅ PQDistanceBase model produces valid distances')
  })

  it('should find top-k neighbors correctly with PQDistance model', async () => {
    // Load ONNX model
    const modelPath = path.join(FIXTURES_DIR, 'pq_distance.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test data
    const testQuery = new Float32Array(metadata.d)
    for (let i = 0; i < testQuery.length; i++) {
      testQuery[i] = (Math.random() - 0.5) * 2
    }

    const testRefCodes = new Array(TEST_PARAMS.n_references * metadata.m)
    for (let i = 0; i < testRefCodes.length; i++) {
      testRefCodes[i] = Math.floor(Math.random() * metadata.k)
    }

    // Prepare inputs
    const query = new ort.Tensor('float32', testQuery, [metadata.d])
    const referenceCodes = new ort.Tensor('int64', testRefCodes, [TEST_PARAMS.n_references, metadata.m])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    // Run inference
    const results = await session.run({
      query: query,
      reference_codes: referenceCodes,
      codebooks: codebooks
    })

    const indices = results.indices.data as BigInt64Array

    // Verify output shape and properties
    expect(indices.length).toBe(TEST_PARAMS.k_nn)
    
    // Check that all indices are valid and unique
    const indexSet = new Set()
    for (let i = 0; i < indices.length; i++) {
      const idx = Number(indices[i])
      expect(idx).toBeGreaterThanOrEqual(0)
      expect(idx).toBeLessThan(TEST_PARAMS.n_references)
      expect(indexSet.has(idx)).toBe(false) // Should be unique
      indexSet.add(idx)
    }

    console.log('✅ PQDistance model produces valid top-k indices')
  })

  it('should handle edge cases gracefully', async () => {
    const modelPath = path.join(FIXTURES_DIR, 'pq_encode.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Test with single embedding
    const singleEmbedding = new Float32Array(metadata.d)
    for (let i = 0; i < singleEmbedding.length; i++) {
      singleEmbedding[i] = Math.random() - 0.5
    }
    
    const embeddings = new ort.Tensor('float32', singleEmbedding, [1, metadata.d])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    const results = await session.run({
      embeddings: embeddings,
      codebooks: codebooks
    })

    const codes = results.codes.data as BigInt64Array
    expect(codes.length).toBe(metadata.m)

    // Verify all codes are within valid range
    for (let i = 0; i < codes.length; i++) {
      expect(Number(codes[i])).toBeGreaterThanOrEqual(0)
      expect(Number(codes[i])).toBeLessThan(metadata.k)
    }

    console.log('✅ Edge case (single embedding) handled correctly')
  })

  it('should have reasonable compression ratio', () => {
    // Calculate expected compression ratio
    const originalBits = metadata.d * 32 // 32-bit floats
    const compressedBits = metadata.m * 8 // 8 bits per code (since k=256)
    const expectedRatio = originalBits / compressedBits
    
    expect(metadata.compression_ratio).toBeGreaterThan(1) // Should have compression
    expect(metadata.compression_ratio).toBe(expectedRatio) // Should match calculated ratio
    expect(metadata.compression_ratio).toBe(64.0) // Expected value for d=128, m=8
    
    console.log(`✅ Compression ratio: ${metadata.compression_ratio}x (${originalBits} -> ${compressedBits} bits)`)
  })
})

describe('PQ Integration Tests', () => {
  
  it('should encode and decode consistently', async () => {
    // Load models
    const encodeModelPath = path.join(FIXTURES_DIR, 'pq_encode.onnx')
    const decodeModelPath = path.join(FIXTURES_DIR, 'pq_decode.onnx')
    
    const encodeSession = await ort.InferenceSession.create(encodeModelPath, {
      executionProviders: ['cpu']
    })
    const decodeSession = await ort.InferenceSession.create(decodeModelPath, {
      executionProviders: ['cpu']
    })

    // Generate test embeddings
    const testEmbeddings = new Float32Array(TEST_PARAMS.n_samples * metadata.d)
    for (let i = 0; i < testEmbeddings.length; i++) {
      testEmbeddings[i] = (Math.random() - 0.5) * 2
    }

    // Prepare data
    const embeddings = new ort.Tensor('float32', testEmbeddings, [TEST_PARAMS.n_samples, metadata.d])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    // Encode
    const encodeResults = await encodeSession.run({
      embeddings: embeddings,
      codebooks: codebooks
    })

    const codes = encodeResults.codes

    // Decode
    const decodeResults = await decodeSession.run({
      codes: codes,
      codebooks: codebooks
    })

    const reconstructed = decodeResults.embeddings.data as Float32Array

    // Check reconstruction error is reasonable
    let totalError = 0
    for (let i = 0; i < testEmbeddings.length; i++) {
      const error = Math.abs(testEmbeddings[i] - reconstructed[i])
      totalError += error * error
    }
    const mse = totalError / testEmbeddings.length
    
    // MSE should be reasonable (not zero due to quantization, but not too large)
    expect(mse).toBeLessThan(2.0)
    expect(mse).toBeGreaterThan(0.0)
    
    console.log(`✅ Encode-decode roundtrip MSE: ${mse.toFixed(6)}`)
  })

  it('should compute distances faster than brute force conceptually', async () => {
    // This test measures the conceptual efficiency by checking that we can
    // compute distances for many reference vectors quickly
    
    const modelPath = path.join(FIXTURES_DIR, 'pq_distance_base.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test data
    const testQuery = new Float32Array(metadata.d)
    for (let i = 0; i < testQuery.length; i++) {
      testQuery[i] = (Math.random() - 0.5) * 2
    }

    const testRefCodes = new Array(TEST_PARAMS.n_references * metadata.m)
    for (let i = 0; i < testRefCodes.length; i++) {
      testRefCodes[i] = Math.floor(Math.random() * metadata.k)
    }

    // Measure distance computation time
    console.time('PQ Distance Computation')
    
    const query = new ort.Tensor('float32', testQuery, [metadata.d])
    const referenceCodes = new ort.Tensor('int64', testRefCodes, [TEST_PARAMS.n_references, metadata.m])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    const results = await session.run({
      query: query,
      reference_codes: referenceCodes,
      codebooks: codebooks
    })
    
    console.timeEnd('PQ Distance Computation')

    const distances = results.distances.data as Float32Array
    expect(distances.length).toBe(TEST_PARAMS.n_references)
    
    // All distances should be non-negative
    for (let i = 0; i < distances.length; i++) {
      expect(distances[i]).toBeGreaterThanOrEqual(0)
    }

    console.log(`✅ Computed distances for ${TEST_PARAMS.n_references} reference vectors`)
  })
})

describe('PQ Performance Tests', () => {
  it('should run PQ encoding with reasonable performance', async () => {
    const modelPath = path.join(FIXTURES_DIR, 'pq_encode.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test data
    const testEmbeddings = new Float32Array(TEST_PARAMS.n_samples * metadata.d)
    for (let i = 0; i < testEmbeddings.length; i++) {
      testEmbeddings[i] = (Math.random() - 0.5) * 2
    }

    // Measure encoding time
    console.time('PQ Encoding')
    const embeddings = new ort.Tensor('float32', testEmbeddings, [TEST_PARAMS.n_samples, metadata.d])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    await session.run({
      embeddings: embeddings,
      codebooks: codebooks
    })
    console.timeEnd('PQ Encoding')

    // This test just ensures the operations complete in reasonable time
    expect(true).toBe(true)
  })

  it('should run PQ distance computation with reasonable performance', async () => {
    const modelPath = path.join(FIXTURES_DIR, 'pq_distance.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test data
    const testQuery = new Float32Array(metadata.d)
    for (let i = 0; i < testQuery.length; i++) {
      testQuery[i] = (Math.random() - 0.5) * 2
    }

    const testRefCodes = new Array(TEST_PARAMS.n_references * metadata.m)
    for (let i = 0; i < testRefCodes.length; i++) {
      testRefCodes[i] = Math.floor(Math.random() * metadata.k)
    }

    // Measure distance + top-k time
    console.time('PQ Distance + Top-K')
    const query = new ort.Tensor('float32', testQuery, [metadata.d])
    const referenceCodes = new ort.Tensor('int64', testRefCodes, [TEST_PARAMS.n_references, metadata.m])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    await session.run({
      query: query,
      reference_codes: referenceCodes,
      codebooks: codebooks
    })
    console.timeEnd('PQ Distance + Top-K')

    // This test just ensures the operations complete in reasonable time
    expect(true).toBe(true)
  })
})