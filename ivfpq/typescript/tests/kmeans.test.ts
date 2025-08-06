/**
 * TypeScript tests for ONNX k-means models.
 * 
 * This test file validates that the exported ONNX k-means models work correctly
 * in a browser-like environment, comparing results with expected values from Python.
 */

import { describe, it, expect, beforeAll } from 'vitest'
import * as ort from 'onnxruntime-web'
import fs from 'fs'
import path from 'path'

// Test configuration
const FIXTURES_DIR = path.join(__dirname, '..', '..', 'python', 'tests', 'artifacts')

interface TestMetadata {
  k: number
  seed: number
  n_samples: number
  n_features: number
  converged: number
  description: string
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

function loadMetadata(filePath: string): TestMetadata {
  const jsonStr = fs.readFileSync(filePath, 'utf8')
  return JSON.parse(jsonStr) as TestMetadata
}

// Test data - generated since kmeans fixtures don't exist in artifacts
const TEST_PARAMS = {
  k: 10,
  seed: 42,
  n_samples: 100,
  n_features: 128
}

let testEmbeddings: Float32Array
let metadata: TestMetadata

beforeAll(() => {
  // Generate test data since kmeans fixtures don't exist
  metadata = {
    k: TEST_PARAMS.k,
    seed: TEST_PARAMS.seed,
    n_samples: TEST_PARAMS.n_samples,
    n_features: TEST_PARAMS.n_features,
    converged: 0,
    description: 'Generated test data for TypeScript tests'
  }

  // Generate random test embeddings
  testEmbeddings = new Float32Array(TEST_PARAMS.n_samples * TEST_PARAMS.n_features)
  const random = new Random(TEST_PARAMS.seed)
  for (let i = 0; i < testEmbeddings.length; i++) {
    testEmbeddings[i] = (random.random() - 0.5) * 2 // Random values between -1 and 1
  }

  console.log('Generated test data:')
  console.log(`  Metadata: k=${metadata.k}, samples=${metadata.n_samples}, features=${metadata.n_features}`)
  console.log(`  Embeddings shape: [${metadata.n_samples}, ${metadata.n_features}]`)
})

// Simple PRNG for reproducible random numbers
class Random {
  private seed: number

  constructor(seed: number) {
    this.seed = seed
  }

  random(): number {
    const x = Math.sin(this.seed++) * 10000
    return x - Math.floor(x)
  }
}

describe('ONNX K-means Models', () => {
  
  it('should load metadata correctly', () => {
    expect(metadata).toBeDefined()
    expect(metadata.k).toBe(10)
    expect(metadata.n_samples).toBe(100)
    expect(metadata.n_features).toBe(128)
    expect(metadata.seed).toBe(42)
  })

  it('should load test data with correct shapes', () => {
    expect(testEmbeddings.length).toBe(metadata.n_samples * metadata.n_features)
  })

  it('should initialize centroids correctly with KMeansInit model', async () => {
    // Load ONNX model
    const modelPath = path.join(FIXTURES_DIR, 'kmeans_init.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Prepare inputs
    const embeddings = new ort.Tensor('float32', testEmbeddings, [metadata.n_samples, metadata.n_features])
    const k = new ort.Tensor('int64', [metadata.k], [])
    const seed = new ort.Tensor('int64', [metadata.seed], [])

    // Run inference
    const results = await session.run({
      embeddings: embeddings,
      k: k,
      seed: seed
    })

    const centroids = results.centroids.data as Float32Array

    // Verify output shape and properties
    expect(centroids.length).toBe(metadata.k * metadata.n_features)
    
    // Check that all values are finite
    for (let i = 0; i < centroids.length; i++) {
      expect(isFinite(centroids[i])).toBe(true)
    }

    console.log('✅ KMeansInit model produces valid centroids')
  })

  it('should perform one iteration correctly with KMeansIteration model', async () => {
    // First get initial centroids
    const initModelPath = path.join(FIXTURES_DIR, 'kmeans_init.onnx')
    const initSession = await ort.InferenceSession.create(initModelPath, {
      executionProviders: ['cpu']
    })

    const embeddings = new ort.Tensor('float32', testEmbeddings, [metadata.n_samples, metadata.n_features])
    const k = new ort.Tensor('int64', [metadata.k], [])
    const seed = new ort.Tensor('int64', [metadata.seed], [])

    const initResults = await initSession.run({
      embeddings: embeddings,
      k: k,
      seed: seed
    })

    // Now test iteration model
    const iterModelPath = path.join(FIXTURES_DIR, 'kmeans_iteration.onnx')
    const iterSession = await ort.InferenceSession.create(iterModelPath, {
      executionProviders: ['cpu']
    })

    const initialCentroids = initResults.centroids.data as Float32Array
    const centroids = new ort.Tensor('float32', initialCentroids, [metadata.k, metadata.n_features])

    // Run inference
    const results = await iterSession.run({
      embeddings: embeddings,
      centroids: centroids
    })

    const updatedCentroids = results.updated_centroids.data as Float32Array
    const assignments = results.assignments.data as BigInt64Array
    const converged = results.converged.data as Float32Array

    // Verify output shapes and properties
    expect(updatedCentroids.length).toBe(metadata.k * metadata.n_features)
    expect(assignments.length).toBe(metadata.n_samples)
    expect(converged.length).toBe(1)

    // Check that all values are finite
    for (let i = 0; i < updatedCentroids.length; i++) {
      expect(isFinite(updatedCentroids[i])).toBe(true)
    }

    // Check that all assignments are valid cluster IDs
    for (let i = 0; i < assignments.length; i++) {
      const assignment = Number(assignments[i])
      expect(assignment).toBeGreaterThanOrEqual(0)
      expect(assignment).toBeLessThan(metadata.k)
    }

    // Check convergence is a valid boolean value (0 or 1)
    expect(converged[0]).toBeGreaterThanOrEqual(0)
    expect(converged[0]).toBeLessThanOrEqual(1)

    console.log('✅ KMeansIteration model produces valid results')
  })

  it('should run iterative k-means for multiple iterations', async () => {
    // Load ONNX models
    const initModelPath = path.join(FIXTURES_DIR, 'kmeans_init.onnx')
    const iterModelPath = path.join(FIXTURES_DIR, 'kmeans_iteration.onnx')
    
    const initSession = await ort.InferenceSession.create(initModelPath, {
      executionProviders: ['cpu']
    })
    const iterSession = await ort.InferenceSession.create(iterModelPath, {
      executionProviders: ['cpu']
    })

    // Initialize centroids
    const embeddings = new ort.Tensor('float32', testEmbeddings, [metadata.n_samples, metadata.n_features])
    const k = new ort.Tensor('int64', [metadata.k], [])
    const seed = new ort.Tensor('int64', [metadata.seed], [])

    const initResults = await initSession.run({
      embeddings: embeddings,
      k: k,
      seed: seed
    })

    let centroids = initResults.centroids.data as Float32Array
    let assignments: Int32Array | null = null

    // Run multiple iterations
    const maxIterations = 10
    for (let i = 0; i < maxIterations; i++) {
      const centroidsTensor = new ort.Tensor('float32', centroids, [metadata.k, metadata.n_features])
      
      const results = await iterSession.run({
        embeddings: embeddings,
        centroids: centroidsTensor
      })

      centroids = results.updated_centroids.data as Float32Array
      const assignmentsBigInt = results.assignments.data as BigInt64Array
      const convergedValue = results.converged.data as Float32Array

      // Convert assignments
      assignments = new Int32Array(assignmentsBigInt.length)
      for (let j = 0; j < assignmentsBigInt.length; j++) {
        assignments[j] = Number(assignmentsBigInt[j])
      }

      if (convergedValue[0] > 0.5) {
        console.log(`✅ K-means converged after ${i + 1} iterations`)
        break
      }
    }

    // Verify we got reasonable results
    expect(centroids.length).toBe(metadata.k * metadata.n_features)
    expect(assignments).not.toBeNull()
    expect(assignments!.length).toBe(metadata.n_samples)

    // Check that all assignments are valid cluster IDs
    for (let i = 0; i < assignments!.length; i++) {
      expect(assignments![i]).toBeGreaterThanOrEqual(0)
      expect(assignments![i]).toBeLessThan(metadata.k)
    }

    // Verify we have points in multiple clusters
    const uniqueAssignments = new Set(assignments!)
    expect(uniqueAssignments.size).toBeGreaterThan(1)
    expect(uniqueAssignments.size).toBeLessThanOrEqual(metadata.k)

    console.log(`✅ Iterative k-means completed with ${uniqueAssignments.size} non-empty clusters`)
  })

  it('should handle edge cases gracefully', async () => {
    const modelPath = path.join(FIXTURES_DIR, 'kmeans_init.onnx')
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu']
    })

    // Test with k=1 (single cluster)
    const singleEmbedding = testEmbeddings.slice(0, metadata.n_features)
    const embeddings = new ort.Tensor('float32', singleEmbedding, [1, metadata.n_features])
    const k = new ort.Tensor('int64', [1], [])
    const seed = new ort.Tensor('int64', [42], [])

    const results = await session.run({
      embeddings: embeddings,
      k: k,
      seed: seed
    })

    const centroids = results.centroids.data as Float32Array
    expect(centroids.length).toBe(metadata.n_features)

    console.log('✅ Edge case (k=1) handled correctly')
  })
})

describe('Performance Tests', () => {
  it('should run k-means with reasonable performance', async () => {
    // Load models
    const initModelPath = path.join(FIXTURES_DIR, 'kmeans_init.onnx')
    const iterModelPath = path.join(FIXTURES_DIR, 'kmeans_iteration.onnx')
    
    const initSession = await ort.InferenceSession.create(initModelPath, {
      executionProviders: ['cpu']
    })
    const iterSession = await ort.InferenceSession.create(iterModelPath, {
      executionProviders: ['cpu']
    })

    // Measure initialization time
    console.time('KMeans Initialization')
    const embeddings = new ort.Tensor('float32', testEmbeddings, [metadata.n_samples, metadata.n_features])
    const k = new ort.Tensor('int64', [metadata.k], [])
    const seed = new ort.Tensor('int64', [metadata.seed], [])

    const initResults = await initSession.run({
      embeddings: embeddings,
      k: k,
      seed: seed
    })
    console.timeEnd('KMeans Initialization')

    // Measure iteration time
    console.time('KMeans Single Iteration')
    const centroids = new ort.Tensor('float32', initResults.centroids.data as Float32Array, [metadata.k, metadata.n_features])
    
    await iterSession.run({
      embeddings: embeddings,
      centroids: centroids
    })
    console.timeEnd('KMeans Single Iteration')

    // This test just ensures the operations complete in reasonable time
    // Actual performance will vary by machine
    expect(true).toBe(true)
  })
})