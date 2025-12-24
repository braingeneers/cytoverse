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
 * Get cell type annotations from h5ad file
 * Returns array where index corresponds to cell index and value is the cell type string
 */
function getCellTypeAnnotations(annData: H5File): string[] {
  const obs = annData.get('obs') as H5Group;
  const cellTypeData = obs.get('CellType');

  // CellType is stored as categorical data (codes + categories)
  if (cellTypeData.type === 'Group') {
    const catGroup = cellTypeData as H5Group;
    const codesData = catGroup.get('codes') as H5DataSet;
    const categoriesData = catGroup.get('categories') as H5DataSet;

    const codes = codesData.value;
    const categories = categoriesData.value;

    console.log('DEBUG: codes type:', typeof codes, 'length:', Array.isArray(codes) ? codes.length : 'N/A');
    console.log('DEBUG: categories type:', typeof categories);
    console.log('DEBUG: categories:', categories);
    console.log('DEBUG: first 10 codes:', Array.isArray(codes) ? codes.slice(0, 10) : codes);

    // Handle different possible formats
    let codeArray: number[];
    let categoryArray: string[];

    // Convert codes to array if needed
    if (Array.isArray(codes)) {
      codeArray = codes;
    } else if (
      codes instanceof Float32Array ||
      codes instanceof Int32Array ||
      codes instanceof Uint32Array ||
      codes instanceof Int8Array ||
      codes instanceof Uint8Array ||
      codes instanceof Int16Array ||
      codes instanceof Uint16Array
    ) {
      codeArray = Array.from(codes);
    } else {
      const constructorName = codes && typeof codes === 'object' && 'constructor' in codes
        ? (codes as { constructor: { name: string } }).constructor.name
        : 'unknown';
      throw new Error(`Unexpected codes type: ${typeof codes}, constructor: ${constructorName}`);
    }

    // Convert categories to string array if needed
    if (Array.isArray(categories)) {
      // Categories might be an array of strings or an array of byte arrays
      categoryArray = categories.map(c => {
        if (typeof c === 'string') return c;
        if (c instanceof Uint8Array) return new TextDecoder().decode(c);
        return String(c);
      });
    } else {
      throw new Error(`Unexpected categories type: ${typeof categories}`);
    }

    console.log('DEBUG: categoryArray:', categoryArray);
    console.log('DEBUG: codeArray length:', codeArray.length);

    // Map codes to category strings - this gives us array[cellIndex] = cellType
    const result = codeArray.map(code => categoryArray[code]);
    console.log('DEBUG: first 5 mapped cell types:', result.slice(0, 5));
    return result;
  }

  throw new Error('CellType not found or not in expected categorical format');
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

  it('should identify top 10 genes for cells from each cell type', async () => {
    if (!calculator) {
      console.warn('Skipping: calculator not initialized');
      return;
    }

    const os = await import('os');
    const homeDir = os.homedir();
    const h5adPath = path.join(homeDir, 'data/cytoverse/allen-celltypes+human-cortex+m1-500.h5ad');

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

    // Debug: Check what's in obs
    const obs = annData.get('obs');
    if (obs && obs.type === 'Group') {
      const obsGroup = obs as H5Group;
      console.log('Available obs fields:', obsGroup.keys());
    }

    const cellTypeAnnotations = getCellTypeAnnotations(annData);
    const rawCounts = getRawCounts(annData);

    console.log(`Found ${cellNames.length} cells and ${sampleGenes.length} genes`);
    console.log(`Cell type annotations length: ${cellTypeAnnotations.length}`);
    console.log('First 5 cell types:', cellTypeAnnotations.slice(0, 5));

    // Get unique cell types
    const uniqueCellTypes = [...new Set(cellTypeAnnotations)];
    console.log(`Found ${uniqueCellTypes.length} unique cell types`);

    // Select one cell from each cell type
    const selectedIndices: number[] = [];
    const cellTypeLabels: string[] = [];

    // For each unique cell type, find the first cell with that type
    for (const cellType of uniqueCellTypes) {
      // Find first cell index with this cell type
      const idx = cellTypeAnnotations.findIndex(ct => ct === cellType);
      if (idx !== -1) {
        selectedIndices.push(idx);
        cellTypeLabels.push(cellType);
      }
    }

    console.log(`Selected ${selectedIndices.length} cells (one per cell type)`);
    console.log('Cell types:', cellTypeLabels);

    // Precompute inflation indices
    const inflationIndices = precomputeInflationIndices(modelGenes, sampleGenes);
    const mappedGenes = inflationIndices.filter((x) => x !== -1).length;
    console.log(`Mapped ${mappedGenes} / ${sampleGenes.length} sample genes to model`);

    // Process each selected cell
    console.log('\n=== Computing Feature Importance ===\n');

    for (let i = 0; i < selectedIndices.length; i++) {
      const cellIdx = selectedIndices[i];
      const cellName = cellNames[cellIdx];
      const cellType = cellTypeLabels[i];

      console.log(`\n[${i + 1}/${selectedIndices.length}] Cell Type: ${cellType}`);
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
