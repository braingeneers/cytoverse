/**
 * Feature importance extraction for ONNX embedding models in the browser.
 *
 * This module implements occlusion-based attribution to identify
 * the top N genes that most influence cell embeddings.
 *
 * Key features:
 * - Sparse computation (only tests expressed genes)
 * - Memory-efficient batch processing
 * - Compatible with existing ONNX inference pipeline
 */

import * as ort from 'onnxruntime-web';

/**
 * Configuration for feature importance calculation.
 */
export interface FeatureImportanceConfig {
  /** Minimum expression threshold for filtering genes (default: 0.1) */
  minExpression?: number;

  /** Number of genes to process in each batch (default: 50) */
  batchSize?: number;

  /** Only test expressed genes (default: true) */
  filterUnexpressed?: boolean;

  /** Method for perturbation: 'zero' | 'mean' (default: 'zero') */
  perturbationMethod?: 'zero' | 'mean';

  /** Use Web Workers for parallel processing (default: false) */
  useWebWorker?: boolean;
}

/**
 * Result for a single gene's importance.
 */
export interface GeneImportance {
  /** Gene name */
  gene: string;

  /** Importance score (L2 distance in embedding space) */
  importance: number;

  /** Raw expression value */
  expression: number;

  /** Gene index in the model */
  index: number;
}

/**
 * Calculator for feature importance using occlusion method.
 */
export class FeatureImportanceCalculator {
  private session: ort.InferenceSession;
  private genes: string[];
  private nGenes: number;

  /**
   * Create a new feature importance calculator.
   *
   * @param session - ONNX Runtime inference session
   * @param genes - Array of gene names in model order
   */
  constructor(session: ort.InferenceSession, genes: string[]) {
    this.session = session;
    this.genes = genes;
    this.nGenes = genes.length;
  }

  /**
   * Get embedding from ONNX model.
   *
   * @param expression - Raw gene expression counts (Float32Array)
   * @returns Embedding vector (128d Float32Array)
   */
  private async getEmbedding(expression: Float32Array): Promise<Float32Array> {
    // Ensure 2D shape (1, n_genes)
    const inputTensor = new ort.Tensor('float32', expression, [1, this.nGenes]);

    const results = await this.session.run({ input: inputTensor });
    const outputTensor = results.output;

    if (!outputTensor) {
      throw new Error('Model did not return output');
    }

    // Return as Float32Array
    return new Float32Array(outputTensor.data as Float32Array);
  }

  /**
   * Calculate L2 distance between two embeddings.
   */
  private l2Distance(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Calculate feature importance using occlusion method.
   *
   * This method perturbs each gene individually and measures the
   * change in the embedding space.
   *
   * @param expression - Raw gene expression counts (Float32Array)
   * @param topN - Number of top genes to return
   * @param config - Configuration options
   * @returns Promise resolving to array of top N gene importances
   */
  async calculateImportance(
    expression: Float32Array,
    topN: number = 5,
    config: FeatureImportanceConfig = {}
  ): Promise<GeneImportance[]> {
    const {
      minExpression = 0.1,
      batchSize = 50,
      filterUnexpressed = true,
      perturbationMethod = 'zero',
    } = config;

    console.log(`Calculating feature importance for ${this.nGenes} genes...`);

    // Get baseline embedding
    const baselineEmbedding = await this.getEmbedding(expression);

    // Determine which genes to test
    let genesToTest: number[];
    if (filterUnexpressed) {
      genesToTest = [];
      for (let i = 0; i < this.nGenes; i++) {
        if (expression[i] > minExpression) {
          genesToTest.push(i);
        }
      }
      console.log(
        `Testing ${genesToTest.length} expressed genes (>${minExpression}) out of ${this.nGenes} total`
      );
    } else {
      genesToTest = Array.from({ length: this.nGenes }, (_, i) => i);
    }

    // Calculate mean expression for 'mean' perturbation method
    let meanExpression = 0;
    if (perturbationMethod === 'mean') {
      let sum = 0;
      for (const idx of genesToTest) {
        sum += expression[idx];
      }
      meanExpression = sum / genesToTest.length;
    }

    // Calculate importance for each gene
    const importances = new Float32Array(this.nGenes);

    // Process in batches to avoid blocking UI
    for (let i = 0; i < genesToTest.length; i += batchSize) {
      const batchIndices = genesToTest.slice(i, i + batchSize);

      // Process each gene in the batch
      for (const geneIdx of batchIndices) {
        // Create perturbed version
        const perturbed = new Float32Array(expression);

        if (perturbationMethod === 'zero') {
          perturbed[geneIdx] = 0.0;
        } else if (perturbationMethod === 'mean') {
          perturbed[geneIdx] = meanExpression;
        }

        // Get embedding for perturbed input
        const perturbedEmbedding = await this.getEmbedding(perturbed);

        // Measure L2 distance in embedding space
        const distance = this.l2Distance(baselineEmbedding, perturbedEmbedding);
        importances[geneIdx] = distance;
      }

      // Yield to event loop to keep UI responsive
      if (i % (batchSize * 5) === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    // Create results array
    const results: GeneImportance[] = [];
    for (let i = 0; i < this.nGenes; i++) {
      if (importances[i] > 0) {
        results.push({
          gene: this.genes[i],
          importance: importances[i],
          expression: expression[i],
          index: i,
        });
      }
    }

    // Sort by importance (descending) and return top N
    results.sort((a, b) => b.importance - a.importance);
    return results.slice(0, topN);
  }


  /**
   * Calculate feature importance for multiple cells in batch.
   *
   * This is more efficient than calling calculateImportance() multiple times
   * as it can share computation across cells.
   *
   * @param expressions - Array of expression vectors
   * @param topN - Number of top genes to return per cell
   * @param config - Configuration options
   * @returns Promise resolving to array of results (one per cell)
   */
  async calculateImportanceBatch(
    expressions: Float32Array[],
    topN: number = 5,
    config: FeatureImportanceConfig = {}
  ): Promise<GeneImportance[][]> {
    const results: GeneImportance[][] = [];

    for (const expression of expressions) {
      const cellResults = await this.calculateImportance(expression, topN, config);
      results.push(cellResults);
    }

    return results;
  }

  /**
   * Get genes that are commonly important across multiple cells.
   *
   * This is useful for identifying marker genes for a cell type.
   *
   * @param expressions - Array of expression vectors
   * @param topN - Number of top genes to consider per cell
   * @param config - Configuration options
   * @returns Map of gene names to frequency counts
   */
  async getCommonImportantGenes(
    expressions: Float32Array[],
    topN: number = 5,
    config: FeatureImportanceConfig = {}
  ): Promise<Map<string, number>> {
    const batchResults = await this.calculateImportanceBatch(expressions, topN, config);

    // Count gene frequencies
    const geneCounts = new Map<string, number>();
    for (const cellResults of batchResults) {
      for (const result of cellResults) {
        const count = geneCounts.get(result.gene) || 0;
        geneCounts.set(result.gene, count + 1);
      }
    }

    // Sort by frequency
    return new Map(Array.from(geneCounts.entries()).sort((a, b) => b[1] - a[1]));
  }
}

/**
 * Create a feature importance calculator from an ONNX model path.
 *
 * @param modelPath - Path to the ONNX model file
 * @param genesPath - Path to the genes.txt file
 * @returns Promise resolving to FeatureImportanceCalculator
 */
export async function createFeatureImportanceCalculator(
  modelPath: string,
  genesPath: string
): Promise<FeatureImportanceCalculator> {
  // Load ONNX session
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
  });

  // Load genes
  const response = await fetch(genesPath);
  const text = await response.text();
  const genes = text.trim().split('\n');

  return new FeatureImportanceCalculator(session, genes);
}

/**
 * Utility function to format importance results for display.
 *
 * @param results - Array of gene importance results
 * @returns Formatted string for console output
 */
export function formatImportanceResults(results: GeneImportance[]): string {
  let output = 'Rank  Gene            Importance   Expression\n';
  output += '----  --------------  -----------  -----------\n';

  results.forEach((result, index) => {
    const rank = String(index + 1).padEnd(4);
    const gene = result.gene.padEnd(14);
    const importance = result.importance.toFixed(6).padStart(11);
    const expression = result.expression.toFixed(2).padStart(11);
    output += `${rank}  ${gene}  ${importance}  ${expression}\n`;
  });

  return output;
}

/**
 * Example usage demonstrating the API.
 */
export async function exampleUsage() {
  // Create calculator
  const calculator = await createFeatureImportanceCalculator(
    '/models/scimilarity/model.onnx',
    '/models/scimilarity/genes.txt'
  );

  // Assume we have raw expression data for a cell
  // This would come from loading an H5AD file
  const cellExpression = new Float32Array(28230); // 28,230 genes
  // ... populate with actual expression data ...

  // Calculate occlusion-based attribution
  console.log('=== Occlusion-based Attribution ===');
  const topGenes = await calculator.calculateImportance(cellExpression, 10, {
    filterUnexpressed: true,
    minExpression: 0.1,
    perturbationMethod: 'zero',
  });
  console.log('Top 10 genes:');
  console.log(formatImportanceResults(topGenes));

  // Analyze multiple cells to find common markers
  const multipleCells: Float32Array[] = [
    /* array of expression vectors */
  ];
  const commonGenes = await calculator.getCommonImportantGenes(multipleCells, 10);

  console.log('\nCommon important genes:');
  for (const [gene, count] of commonGenes.entries()) {
    console.log(`  ${gene}: ${count}/${multipleCells.length} cells`);
  }
}
