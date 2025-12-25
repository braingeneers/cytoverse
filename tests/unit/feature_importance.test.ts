/**
 * Test occlusion-based feature importance on real cells from h5ad dataset.
 *
 * This test validates the attribution by:
 * 1. Loading 10 cells from the allen-celltypes dataset (one from each layer)
 * 2. Using the full preprocessing model (model.onnx)
 * 3. Computing top 10 genes for each cell's embedding
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as h5wasm from 'h5wasm';
import { InferenceSession } from 'onnxruntime-web';
import { FeatureImportanceCalculator } from '../../src/feature_importance';
import { parse } from 'csv-parse/sync';

// Match worker.ts interfaces
interface H5DataSet {
  type: string;
  value: unknown;
  shape: number[];
  keys(): string[];
  slice(ranges: number[][]): unknown;
}

interface H5Group {
  type: string;
  keys(): string[];
  get(path: string): H5DataSet | H5Group;
}

interface H5File {
  keys(): string[];
  get(path: string): H5DataSet | H5Group;
  close(): void;
}

interface RawCountsData {
  isSparse: boolean;
  data: H5DataSet;
  indices: H5DataSet | null;
  indptr: H5DataSet | null;
}

/**
 * Extract cell names from h5ad file (from worker.ts)
 */
function getCellNames(annData: H5File): string[] {
  if (!annData.keys().includes('obs')) {
    throw new Error('Unable to find cell names: Missing "obs" group in h5ad file');
  }

  const obs = annData.get('obs');

  if (obs.type === 'Dataset') {
    try {
      const obsValue = (obs as H5DataSet).value as Array<[string]>;
      return obsValue.map((e) => e[0]);
    } catch {
      throw new Error('Unable to extract cell names from obs Dataset');
    }
  }

  if (obs.type === 'Group') {
    const obsGroup = obs as H5Group;
    const obsKeys = obsGroup.keys();
    const indexKeys = ['index', '_index', 'barcodes', '_barcodes', 'cell_id', 'cell_name'];

    for (const key of indexKeys) {
      if (obsKeys.includes(key)) {
        try {
          return (obsGroup.get(key) as H5DataSet).value as string[];
        } catch {
          console.warn(`Failed to read cell names from obs/${key}`);
        }
      }
    }

    throw new Error(
      `Unable to find cell names in obs group. Available keys: ${obsKeys.join(', ')}`
    );
  }

  throw new Error('Unable to find cell names: obs is neither Dataset nor Group');
}

/**
 * Load labels from CSV file
 */
interface LabelRecord {
  cell_id: string;
  category_label: string;
  confidence: string;
}

function loadLabelsFromCSV(csvPath: string): Map<string, string> {
  const csvContent = fs.readFileSync(csvPath, 'utf-8');
  const records = parse(csvContent, {
    columns: true,
    skip_empty_lines: true,
  }) as LabelRecord[];

  // Create a map of cell_id -> category_label
  const labelMap = new Map<string, string>();
  for (const record of records) {
    labelMap.set(record.cell_id, record.category_label);
  }

  return labelMap;
}

/**
 * Validate gene symbols (from worker.ts)
 */
function validateGeneSymbols(geneArray: string[]): boolean {
  const knownGenes = ['TP53', 'BRCA1'];
  return knownGenes.some((gene) => geneArray.includes(gene));
}

/**
 * Extract gene names from h5ad file (from worker.ts)
 */
function getSampleGenes(annData: H5File): string[] {
  if (!annData.keys().includes('var')) {
    throw new Error('Unable to find gene names: Missing "var" group in h5ad file');
  }

  const varData = annData.get('var');

  if (varData.type === 'Dataset') {
    try {
      const varValue = (varData as H5DataSet).value as Array<[string]>;
      return varValue.map((e) => e[0]);
    } catch {
      throw new Error('Unable to extract gene names from var Dataset');
    }
  }

  if (varData.type === 'Group') {
    const varGroup = varData as H5Group;
    const varKeys = varGroup.keys();
    const geneKeys = [
      'symbol',
      'gene_symbol',
      'gene_symbols',
      'gene_name',
      'gene_names',
      'feature_name',
      'features',
      'index',
      '_index',
      'gene_id',
      'gene_ids',
    ];

    for (const key of geneKeys) {
      if (varKeys.includes(key)) {
        try {
          const genes = (varGroup.get(key) as H5DataSet).value as string[];
          if (validateGeneSymbols(genes)) {
            return genes;
          }
        } catch {
          console.warn(`Failed to read gene names from var/${key}`);
        }
      }
    }

    if (varKeys.length > 0) {
      try {
        const firstKey = varKeys[0];
        const potentialGenes = (varGroup.get(firstKey) as H5DataSet).value as string[];
        if (Array.isArray(potentialGenes) && validateGeneSymbols(potentialGenes)) {
          console.warn('Falling back to using var index as gene names');
          return potentialGenes;
        }
      } catch {
        // Ignore
      }
    }

    throw new Error(
      `Unable to find gene names in var group. Available keys: ${varKeys.join(', ')}`
    );
  }

  throw new Error('Unable to find gene names: var is neither Dataset nor Group');
}

/**
 * Extract raw counts from h5ad file (from worker.ts)
 */
function getRawCounts(annData: H5File): RawCountsData {
  const topLevelKeys = annData.keys();

  // Check layers for raw counts
  if (topLevelKeys.includes('layers')) {
    const layers = annData.get('layers');
    if (layers.type === 'Group') {
      const layersGroup = layers as H5Group;
      const layerKeys = layersGroup.keys();
      const rawCountKeys = ['counts', 'raw_counts', 'raw', 'count', 'spliced', 'unspliced'];

      for (const key of rawCountKeys) {
        if (layerKeys.includes(key)) {
          const countsData = layersGroup.get(key);

          if (countsData.type === 'Dataset') {
            return {
              isSparse: false,
              data: countsData as H5DataSet,
              indices: null,
              indptr: null,
            };
          }

          if (countsData.type === 'Group') {
            const sparseGroup = countsData as H5Group;
            const sparseKeys = sparseGroup.keys();

            if (
              sparseKeys.includes('data') &&
              sparseKeys.includes('indices') &&
              sparseKeys.includes('indptr')
            ) {
              return {
                isSparse: true,
                data: sparseGroup.get('data') as H5DataSet,
                indices: sparseGroup.get('indices') as H5DataSet,
                indptr: sparseGroup.get('indptr') as H5DataSet,
              };
            }
          }
        }
      }
    }
  }

  // Check raw attribute
  if (topLevelKeys.includes('raw')) {
    const raw = annData.get('raw');
    if (raw.type === 'Group') {
      const rawGroup = raw as H5Group;
      if (rawGroup.keys().includes('X')) {
        const rawX = rawGroup.get('X');

        if (rawX.type === 'Dataset') {
          return {
            isSparse: false,
            data: rawX as H5DataSet,
            indices: null,
            indptr: null,
          };
        }

        if (rawX.type === 'Group') {
          const sparseGroup = rawX as H5Group;
          const sparseKeys = sparseGroup.keys();

          if (
            sparseKeys.includes('data') &&
            sparseKeys.includes('indices') &&
            sparseKeys.includes('indptr')
          ) {
            return {
              isSparse: true,
              data: sparseGroup.get('data') as H5DataSet,
              indices: sparseGroup.get('indices') as H5DataSet,
              indptr: sparseGroup.get('indptr') as H5DataSet,
            };
          }
        }
      }
    }
  }

  // Use main X matrix as last resort
  if (topLevelKeys.includes('X')) {
    const X = annData.get('X');

    if (X.type === 'Dataset') {
      return {
        isSparse: false,
        data: X as H5DataSet,
        indices: null,
        indptr: null,
      };
    }

    if (X.type === 'Group') {
      const sparseGroup = X as H5Group;
      const sparseKeys = sparseGroup.keys();

      if (
        sparseKeys.includes('data') &&
        sparseKeys.includes('indices') &&
        sparseKeys.includes('indptr')
      ) {
        return {
          isSparse: true,
          data: sparseGroup.get('data') as H5DataSet,
          indices: sparseGroup.get('indices') as H5DataSet,
          indptr: sparseGroup.get('indptr') as H5DataSet,
        };
      }
    }
  }

  const availableKeys = topLevelKeys.join(', ');
  throw new Error(
    `Unable to find raw counts data. Available keys: ${availableKeys}`
  );
}

/**
 * Precompute inflation indices for gene mapping (from worker.ts)
 */
function precomputeInflationIndices(
  currentModelGenes: string[],
  sampleGenes: string[]
): number[] {
  const inflationIndices: number[] = [];
  for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
    inflationIndices.push(currentModelGenes.indexOf(sampleGenes[geneIndex]));
  }
  return inflationIndices;
}

/**
 * Extract and inflate expression for a single cell (from worker.ts)
 */
function extractCellExpression(
  cellIndex: number,
  data: H5DataSet,
  indices: H5DataSet | null,
  indptr: H5DataSet | null,
  isSparse: boolean,
  sampleGenes: string[],
  inflationIndices: number[],
  modelGenesLength: number
): Float32Array {
  const expression = new Float32Array(modelGenesLength);

  if (isSparse) {
    const [start, end] = indptr!.slice([[cellIndex, cellIndex + 2]]) as number[];
    const values = data.slice([[start, end]]) as number[];
    const valueIndices = indices!.slice([[start, end]]) as number[];

    for (let j = 0; j < valueIndices.length; j++) {
      const sampleIndex = inflationIndices[valueIndices[j]];
      if (sampleIndex !== -1) {
        expression[sampleIndex] = Number(values[j]);
      }
    }
  } else {
    let sampleExpression: number[] | null = null;
    if (data.shape.length === 1) {
      sampleExpression = data.slice([
        [cellIndex * sampleGenes.length, (cellIndex + 1) * sampleGenes.length],
      ]) as number[];
    } else if (data.shape.length === 2) {
      sampleExpression = data.slice([
        [cellIndex, cellIndex + 1],
        [0, sampleGenes.length],
      ]) as number[];
    } else {
      throw new Error('Unsupported data shape');
    }
    for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
      const sampleIndex = inflationIndices[geneIndex];
      if (sampleIndex !== -1) {
        expression[sampleIndex] = sampleExpression[geneIndex];
      }
    }
  }

  return expression;
}

describe('Occlusion-based Feature Importance', () => {
  let calculator: FeatureImportanceCalculator;
  let modelGenes: string[];

  beforeAll(async () => {
    // Load ONNX model
    const modelPath = path.join(process.cwd(), 'public/models/scimilarity/embedding/model.onnx');
    const genesPath = path.join(process.cwd(), 'public/models/scimilarity/embedding/genes.txt');

    if (!fs.existsSync(modelPath)) {
      console.warn(`Skipping tests: Model not found at ${modelPath}`);
      return;
    }

    const session = await InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
    });

    const genesText = fs.readFileSync(genesPath, 'utf-8');
    modelGenes = genesText.trim().split('\n');

    calculator = new FeatureImportanceCalculator(session, modelGenes);
    console.log(`Loaded model with ${modelGenes.length} genes`);
  });

  it('should load model and calculator', () => {
    if (!calculator) return;
    expect(calculator).toBeDefined();
    expect(modelGenes.length).toBeGreaterThan(10000);
  });

  it('should identify top 10 genes for cells from each category label', async () => {
    if (!calculator) {
      console.warn('Skipping: calculator not initialized');
      return;
    }

    // Load labels from CSV
    const csvPath = path.join(process.cwd(), 'tests/fixtures/allen-celltypes+human-cortex+m1-500_labels.csv');
    const labelMap = loadLabelsFromCSV(csvPath);
    console.log(`Loaded ${labelMap.size} cell labels from CSV`);

    // Load h5ad file
    const h5adPath = path.join(process.cwd(), 'tests/fixtures/allen-celltypes+human-cortex+m1-500.h5ad');

    if (!fs.existsSync(h5adPath)) {
      console.warn(`Skipping: H5AD not found at ${h5adPath}`);
      return;
    }

    console.log(`\nLoading ${h5adPath}...`);

    // Initialize h5wasm and load h5ad file
    const Module = await h5wasm.ready;
    const buffer = fs.readFileSync(h5adPath);

    // Write buffer to virtual filesystem
    Module.FS.writeFile('test.h5ad', new Uint8Array(buffer));

    // Open file from virtual filesystem
    const f = new h5wasm.File('test.h5ad', 'r');
    const annData = f as unknown as H5File;

    // Extract metadata
    const cellNames = getCellNames(annData);
    const sampleGenes = getSampleGenes(annData);
    const rawCounts = getRawCounts(annData);

    console.log(`Found ${cellNames.length} cells and ${sampleGenes.length} genes`);

    // Find unique category labels and select one cell from each
    const categoryToCellMap = new Map<string, { cellId: string; cellIndex: number }>();

    for (let i = 0; i < cellNames.length; i++) {
      const cellId = cellNames[i];
      const categoryLabel = labelMap.get(cellId);

      if (categoryLabel && !categoryToCellMap.has(categoryLabel)) {
        categoryToCellMap.set(categoryLabel, { cellId, cellIndex: i });
      }
    }

    const selectedCells = Array.from(categoryToCellMap.entries()).map(([label, info]) => ({
      categoryLabel: label,
      cellId: info.cellId,
      cellIndex: info.cellIndex,
    }));

    console.log(`Found ${selectedCells.length} unique category labels`);
    console.log('Category labels:', selectedCells.map(c => c.categoryLabel));

    // Precompute inflation indices
    const inflationIndices = precomputeInflationIndices(modelGenes, sampleGenes);
    const mappedGenes = inflationIndices.filter((x) => x !== -1).length;
    console.log(`Mapped ${mappedGenes} / ${sampleGenes.length} sample genes to model`);

    // Process each selected cell
    console.log('\n=== Computing Feature Importance ===\n');

    for (let i = 0; i < selectedCells.length; i++) {
      const cell = selectedCells[i];
      const cellIdx = cell.cellIndex;
      const cellName = cell.cellId;
      const categoryLabel = cell.categoryLabel;

      console.log(`\n[${i + 1}/${selectedCells.length}] Category: ${categoryLabel}`);
      console.log(`Cell: ${cellName} (Index: ${cellIdx})`);
      console.log('─'.repeat(80));

      // Extract and inflate expression
      const expression = extractCellExpression(
        cellIdx,
        rawCounts.data,
        rawCounts.indices,
        rawCounts.indptr,
        rawCounts.isSparse,
        sampleGenes,
        inflationIndices,
        modelGenes.length
      );

      // Count expressed genes
      const expressedCount = expression.filter((x) => x > 0).length;
      console.log(`Expressed genes: ${expressedCount} / ${modelGenes.length}`);

      // Calculate feature importance
      const topGenes = await calculator.calculateImportance(expression, 10, {
        filterUnexpressed: true,
        minExpression: 0.1,
        perturbationMethod: 'zero',
      });

      // Validate results
      expect(topGenes).toBeDefined();
      expect(topGenes.length).toBe(10);
      expect(topGenes[0].importance).toBeGreaterThan(0);

      // Check that importance is sorted descending
      for (let j = 0; j < topGenes.length - 1; j++) {
        expect(topGenes[j].importance).toBeGreaterThanOrEqual(topGenes[j + 1].importance);
      }

      // Display results
      console.log('\nTop 10 genes attributed to embedding:');
      console.log('Rank  Gene            Importance   Expression');
      console.log('----  --------------  -----------  -----------');
      topGenes.forEach((result, index) => {
        const rank = String(index + 1).padEnd(4);
        const gene = result.gene.padEnd(14);
        const importance = result.importance.toFixed(6).padStart(11);
        const expr = result.expression.toFixed(2).padStart(11);
        console.log(`${rank}  ${gene}  ${importance}  ${expr}`);
      });
    }

    annData.close();
    console.log('\n=== Test Complete ===');
  }, 300000); // 5 minute timeout for this long test
});
