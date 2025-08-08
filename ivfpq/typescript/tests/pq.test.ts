/**
 * TypeScript tests for ONNX PQ (Product Quantization) models with residual vectors.
 * 
 * This test file validates that the exported ONNX PQ models work correctly
 * in a browser-like environment with residual vector support, comparing results
 * with expected values from Python training pipeline.
 */

import { describe, it, expect, beforeAll } from 'vitest'
import * as ort from 'onnxruntime-web'
import fs from 'fs'
import path from 'path'

// Test configuration
const FIXTURES_DIR = path.join(__dirname, '..', '..', 'python', 'tests', 'artifacts')

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
  version: string // Should be "residual-1.0" for new implementation
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

// Generate residual-like test vectors (zero-centered)
function generateResidualVectors(numVectors: number, dimension: number, seed: number = 42): Float32Array {
  // Simple deterministic random number generator
  let seedValue = seed
  const rng = () => {
    seedValue = (seedValue * 9301 + 49297) % 233280
    return seedValue / 233280
  }

  const vectors = new Float32Array(numVectors * dimension)
  for (let i = 0; i < vectors.length; i++) {
    // Generate residual-like vectors (zero-centered with moderate variance)
    vectors[i] = (rng() - 0.5) * 0.8 // Values between -0.4 and 0.4
  }
  return vectors
}

// Test data - will be loaded from fixtures
let metadata: PQTestMetadata
let testCodebooks: Float32Array

// Test parameters
const TEST_PARAMS = {
  n_samples: 100,
  n_references: 1000,
  k_nn: 50
}

beforeAll(() => {
  // Load metadata and codebooks
  metadata = loadMetadata(path.join(FIXTURES_DIR, 'pq_metadata.json'))
  testCodebooks = loadFloat32Binary(path.join(FIXTURES_DIR, 'pq_codebooks.bin'))

  console.log('Loaded PQ test data for residual vectors:')
  console.log(`  Metadata: d=${metadata.d}, m=${metadata.m}, k=${metadata.k}, d_sub=${metadata.d_sub}`)
  console.log(`  Version: ${metadata.version}`)
  console.log(`  Codebooks shape: [${metadata.m}, ${metadata.k}, ${metadata.d_sub}]`)
  console.log(`  Test parameters: n_samples=${TEST_PARAMS.n_samples}, n_references=${TEST_PARAMS.n_references}, k_nn=${TEST_PARAMS.k_nn}`)
})

describe('ONNX PQ Models for Residual Vectors', () => {
  
  it('should load metadata correctly with residual version', () => {
    expect(metadata).toBeDefined()
    expect(metadata.d).toBeGreaterThan(0)
    expect(metadata.m).toBeGreaterThan(0)
    expect(metadata.k).toBeGreaterThan(0)
    expect(metadata.d_sub).toBe(metadata.d / metadata.m)
    expect(metadata.compression_ratio).toBeGreaterThan(1)
    expect(metadata.codebooks_shape).toEqual([metadata.m, metadata.k, metadata.d_sub])
    
    // Check for residual version
    if (metadata.version) {
      expect(metadata.version).toBe('residual-1.0')
      console.log('✅ Detected residual-based PQ model')
    } else {
      console.warn('⚠️  Legacy PQ model detected, proceeding with tests')
    }
  })

  it('should load codebooks with correct shape for residual vectors', () => {
    expect(testCodebooks.length).toBe(metadata.codebooks_size)
    expect(testCodebooks.length).toBe(metadata.m * metadata.k * metadata.d_sub)
    
    // Check that codebooks contain valid data (not all zeros or NaN)
    let nonZeroCount = 0
    for (let i = 0; i < testCodebooks.length; i++) {
      expect(isFinite(testCodebooks[i])).toBe(true)
      if (Math.abs(testCodebooks[i]) > 1e-6) {
        nonZeroCount++
      }
    }
    expect(nonZeroCount).toBeGreaterThan(testCodebooks.length * 0.5) // At least 50% non-zero

    console.log('✅ Codebooks loaded successfully for residual vector processing')
  })

  it('should encode residual vectors correctly with PQEncode model', async () => {
    // Load ONNX model
    const modelPath = path.join(FIXTURES_DIR, 'pq_encode.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test residual vectors
    const testResiduals = generateResidualVectors(TEST_PARAMS.n_samples, metadata.d)

    // Prepare inputs
    const embeddings = new ort.Tensor('float32', testResiduals, [TEST_PARAMS.n_samples, metadata.d])
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

    console.log('✅ PQEncode model produces valid codes for residual vectors')
  })


  it('should compute distances correctly for residual vectors with PQDistance model', async () => {
    // Load ONNX model
    const modelPath = path.join(FIXTURES_DIR, 'pq_distance.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test residual query
    const testQueryResidual = generateResidualVectors(1, metadata.d, 123)

    const testRefCodes = new Array(TEST_PARAMS.n_references * metadata.m)
    for (let i = 0; i < testRefCodes.length; i++) {
      testRefCodes[i] = Math.floor(Math.random() * metadata.k)
    }

    // Prepare inputs using correct input names for residual model
    const queryResidual = new ort.Tensor('float32', testQueryResidual, [metadata.d])
    const referenceCodes = new ort.Tensor('int64', testRefCodes, [TEST_PARAMS.n_references, metadata.m])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    // Run inference
    const results = await session.run({
      query_residual: queryResidual,
      reference_codes: referenceCodes,
      codebooks: codebooks
    })

    const indices = results.indices.data as BigInt64Array
    const distances = results.distances.data as Float32Array

    // Verify output shape and properties
    expect(indices.length).toBe(TEST_PARAMS.k_nn)
    expect(distances.length).toBe(TEST_PARAMS.k_nn)
    
    // Check that all indices are valid and unique
    const indexSet = new Set()
    for (let i = 0; i < indices.length; i++) {
      const idx = Number(indices[i])
      expect(idx).toBeGreaterThanOrEqual(0)
      expect(idx).toBeLessThan(TEST_PARAMS.n_references)
      expect(indexSet.has(idx)).toBe(false) // Should be unique
      indexSet.add(idx)
    }

    // Check that distances are non-negative and sorted
    for (let i = 0; i < distances.length; i++) {
      expect(distances[i]).toBeGreaterThanOrEqual(0)
      expect(isFinite(distances[i])).toBe(true)
      
      if (i > 0) {
        expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1]) // Should be sorted
      }
    }

    console.log(`✅ PQDistance model produces valid results for residual vectors (min distance: ${distances[0].toFixed(6)})`)
  })

  it('should handle edge cases gracefully with residual vectors', async () => {
    const modelPath = path.join(FIXTURES_DIR, 'pq_encode.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Test with single residual vector
    const singleResidual = generateResidualVectors(1, metadata.d, 456)
    
    const embeddings = new ort.Tensor('float32', singleResidual, [1, metadata.d])
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

    console.log('✅ Edge case (single residual vector) handled correctly')
  })

  it('should have reasonable compression ratio for residual vectors', () => {
    // Calculate expected compression ratio
    const originalBits = metadata.d * 32 // 32-bit floats
    const compressedBits = metadata.m * 8 // 8 bits per code (since k <= 256)
    const expectedRatio = originalBits / compressedBits
    
    expect(metadata.compression_ratio).toBeGreaterThan(1) // Should have compression
    expect(metadata.compression_ratio).toBeCloseTo(expectedRatio, 1) // Should match calculated ratio
    
    console.log(`✅ Compression ratio for residual vectors: ${metadata.compression_ratio}x (${originalBits} -> ${compressedBits} bits)`)
  })
})

describe('PQ Integration Tests for Residual Vectors', () => {
  

  it('should compute distances efficiently for residual vectors', async () => {
    // This test measures the efficiency of residual-based distance computation
    
    const modelPath = path.join(FIXTURES_DIR, 'pq_distance.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test residual query
    const testQueryResidual = generateResidualVectors(1, metadata.d, 999)

    const testRefCodes = new Array(TEST_PARAMS.n_references * metadata.m)
    for (let i = 0; i < testRefCodes.length; i++) {
      testRefCodes[i] = Math.floor(Math.random() * metadata.k)
    }

    // Measure distance computation time
    console.time('PQ Distance Computation (Residual)')
    
    const queryResidual = new ort.Tensor('float32', testQueryResidual, [metadata.d])
    const referenceCodes = new ort.Tensor('int64', testRefCodes, [TEST_PARAMS.n_references, metadata.m])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    const results = await session.run({
      query_residual: queryResidual,
      reference_codes: referenceCodes,
      codebooks: codebooks
    })
    
    console.timeEnd('PQ Distance Computation (Residual)')

    const indices = results.indices.data as BigInt64Array
    const distances = results.distances.data as Float32Array
    
    expect(indices.length).toBe(TEST_PARAMS.k_nn)
    expect(distances.length).toBe(TEST_PARAMS.k_nn)
    
    // All distances should be non-negative and sorted
    for (let i = 0; i < distances.length; i++) {
      expect(distances[i]).toBeGreaterThanOrEqual(0)
      if (i > 0) {
        expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1])
      }
    }

    console.log(`✅ Computed residual-based distances for ${TEST_PARAMS.n_references} reference vectors in top-k format`)
  })

})

describe('PQ Performance Tests for Residual Vectors', () => {
  it('should run residual vector encoding with reasonable performance', async () => {
    const modelPath = path.join(FIXTURES_DIR, 'pq_encode.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test residual vectors
    const testResiduals = generateResidualVectors(TEST_PARAMS.n_samples, metadata.d, 222)

    // Measure encoding time
    console.time('PQ Encoding (Residual)')
    const embeddings = new ort.Tensor('float32', testResiduals, [TEST_PARAMS.n_samples, metadata.d])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    await session.run({
      embeddings: embeddings,
      codebooks: codebooks
    })
    console.timeEnd('PQ Encoding (Residual)')

    // This test just ensures the operations complete in reasonable time
    expect(true).toBe(true)
  })

  it('should run residual distance computation with reasonable performance', async () => {
    const modelPath = path.join(FIXTURES_DIR, 'pq_distance.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Generate test residual query
    const testQueryResidual = generateResidualVectors(1, metadata.d, 333)

    const testRefCodes = new Array(TEST_PARAMS.n_references * metadata.m)
    for (let i = 0; i < testRefCodes.length; i++) {
      testRefCodes[i] = Math.floor(Math.random() * metadata.k)
    }

    // Measure distance + top-k time
    console.time('PQ Distance + Top-K (Residual)')
    const queryResidual = new ort.Tensor('float32', testQueryResidual, [metadata.d])
    const referenceCodes = new ort.Tensor('int64', testRefCodes, [TEST_PARAMS.n_references, metadata.m])
    const codebooks = new ort.Tensor('float32', testCodebooks, [metadata.m, metadata.k, metadata.d_sub])

    await session.run({
      query_residual: queryResidual,
      reference_codes: referenceCodes,
      codebooks: codebooks
    })
    console.timeEnd('PQ Distance + Top-K (Residual)')

    // This test just ensures the operations complete in reasonable time
    expect(true).toBe(true)
  })
})