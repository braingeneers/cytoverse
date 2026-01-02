/**
 * Unit tests for validating ONNX models can be loaded and executed.
 *
 * Tests all model.onnx files under public/models/<model-name>/embedding/
 * to ensure they can be loaded in the browser and run inference.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as ort from 'onnxruntime-web';
import { readFileSync, readdirSync, existsSync } from 'fs';
import { join } from 'path';
import { mod } from '@tensorflow/tfjs';

// Configure ONNX Runtime to use CPU (WebGPU not available in Node test environment)
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

interface ModelInfo {
  name: string;
  path: string;
  configPath: string;
}

/**
 * Discover all available models in public/models directory.
 */
function discoverModels(): ModelInfo[] {
  const modelsDir = join(process.cwd(), 'public/models');
  if (!existsSync(modelsDir)) {
    console.warn('Models directory not found. Tried:', modelsDir);
    return [];
  }
  console.log('Found models directory:', modelsDir);

  const models: ModelInfo[] = [];
  const modelDirs = readdirSync(modelsDir, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name);

  for (const modelName of modelDirs) {
    const modelPath = join(modelsDir, modelName, 'embedding', 'model.onnx');
    const configPath = join(modelsDir, modelName, 'embedding', 'config.json');

    if (existsSync(modelPath)) {
      models.push({
        name: modelName,
        path: modelPath,
        configPath: configPath,
      });
    }
  }

  return models;
}

/**
 * Load model configuration from config.json.
 */
function loadModelConfig(configPath: string): any {
  if (!existsSync(configPath)) {
    return null;
  }
  const configJson = readFileSync(configPath, 'utf-8');
  return JSON.parse(configJson);
}

// Discover models at module load time (before tests run)
const discoveredModels = discoverModels();

describe('ONNX Model Validation', () => {
  it('should find at least one model', () => {
    expect(discoveredModels.length).toBeGreaterThan(0);
  });

  describe.each(
    // Generate test cases for each discovered model
    discoveredModels.length > 0
      ? discoveredModels
      : [{ name: 'placeholder', path: '', configPath: '' }]
  )('Model: $name', ({ name, path, configPath }) => {
    let session: ort.InferenceSession;
    let config: any;

    beforeAll(async () => {
      if (name === 'placeholder') {
        // Skip placeholder test
        return;
      }

      // Load model configuration
      config = loadModelConfig(configPath);

      // Load ONNX model
      const modelBuffer = readFileSync(path);
      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all',
      });
    });

    it('should have valid config.json', () => {
      if (name === 'placeholder') return;

      // Config is optional for some models
      if (!config) {
        console.log(`  Skipping config validation for ${name} (no config.json)`);
        return;
      }

      expect(config).toHaveProperty('model_type');

      // Check for model-specific configs
      if (config.model_type === 'geneformer') {
        expect(config).toHaveProperty('version');
        expect(config).toHaveProperty('model_input_size');
        expect(config).toHaveProperty('use_cls_token');
        expect(config).toHaveProperty('embedding_dim');
      }
    });

    it('should load ONNX model successfully', () => {
      if (name === 'placeholder') return;

      expect(session).toBeDefined();
    });

    it('should have correct input/output shapes', () => {
      if (name === 'placeholder') return;

      const inputs = session.inputNames;
      const outputs = session.outputNames;

      // All models should have exactly 1 input and 1 output
      expect(inputs).toHaveLength(1);
      expect(outputs).toHaveLength(1);

      // Input should be named "input"
      expect(inputs[0]).toBe('input');

      // Output should be named "output"
      expect(outputs[0]).toBe('output');
    });

    it('should run inference with random input', async () => {
      if (name === 'placeholder') return;

      // Determine input size from genes.txt
      const genesPath = configPath.replace('config.json', 'genes.txt');
      const genesList = readFileSync(genesPath, 'utf-8')
        .split('\n')
        .filter((line) => line.trim());
      const inputSize = genesList.length;

      // Create random input tensor (simulating raw gene counts)
      const batchSize = 2;
      const inputData = new Float32Array(batchSize * inputSize);

      // Fill with random count-like values (0-100 range)
      for (let i = 0; i < inputData.length; i++) {
        inputData[i] = Math.random() * 100;
      }

      const inputTensor = new ort.Tensor('float32', inputData, [batchSize, inputSize]);

      // Run inference
      const feeds = { [session.inputNames[0]]: inputTensor };
      const results = await session.run(feeds);

      // Validate output
      const outputTensor = results[session.outputNames[0]];
      expect(outputTensor).toBeDefined();
      expect(outputTensor.dims[0]).toBe(batchSize);

      // Output should be embeddings with dimension from config
      if (config && config.embedding_dim) {
        expect(outputTensor.dims[1]).toBe(config.embedding_dim);
      } else {
        // Generic check - output should be 2D with reasonable embedding size
        expect(outputTensor.dims).toHaveLength(2);
        expect(outputTensor.dims[1]).toBeGreaterThan(0);
        expect(outputTensor.dims[1]).toBeLessThan(1024);
      }

      // Embeddings should be finite numbers
      const outputData = outputTensor.data as Float32Array;
      for (let i = 0; i < outputData.length; i++) {
        expect(Number.isFinite(outputData[i])).toBe(true);
      }
    });

    it('should produce consistent output for same input', async () => {
      if (name === 'placeholder') return;

      // Determine input size from genes.txt
      const genesPath = configPath.replace('config.json', 'genes.txt');
      const genesList = readFileSync(genesPath, 'utf-8')
        .split('\n')
        .filter((line) => line.trim());
      const inputSize = genesList.length;

      // Create fixed input
      const batchSize = 1;
      const inputData = new Float32Array(inputSize);
      for (let i = 0; i < inputSize; i++) {
        inputData[i] = Math.sin(i) * 50 + 50; // Deterministic values
      }

      const inputTensor = new ort.Tensor('float32', inputData, [batchSize, inputSize]);
      const feeds = { [session.inputNames[0]]: inputTensor };

      // Run inference twice
      const results1 = await session.run(feeds);
      const results2 = await session.run(feeds);

      // Outputs should be identical
      const output1 = results1[session.outputNames[0]].data as Float32Array;
      const output2 = results2[session.outputNames[0]].data as Float32Array;

      expect(output1.length).toBe(output2.length);

      for (let i = 0; i < output1.length; i++) {
        expect(output1[i]).toBeCloseTo(output2[i], 5);
      }
    });

    it('should handle batch inference correctly', async () => {
      if (name === 'placeholder') return;

      // Determine input size from genes.txt
      const genesPath = configPath.replace('config.json', 'genes.txt');
      const genesList = readFileSync(genesPath, 'utf-8')
        .split('\n')
        .filter((line) => line.trim());
      const inputSize = genesList.length;

      // Test with batch size of 2 (quick test)
      const batchSize = 2;
      const inputData = new Float32Array(batchSize * inputSize);

      // Fill with unique values per sample
      for (let b = 0; b < batchSize; b++) {
        for (let i = 0; i < inputSize; i++) {
          inputData[b * inputSize + i] = Math.sin(i + b * 1000) * 50 + 50;
        }
      }

      const inputTensor = new ort.Tensor('float32', inputData, [batchSize, inputSize]);
      const feeds = { [session.inputNames[0]]: inputTensor };

      const results = await session.run(feeds);
      const outputTensor = results[session.outputNames[0]];

      // Batch dimension should match
      expect(outputTensor.dims[0]).toBe(batchSize);
    }, 10000); // 10 second timeout for batch inference
  });
});
