# Foundation Models in CytoVerse

This document describes the foundation models integrated into CytoVerse for single-cell RNA-seq analysis and cell annotation.

## Overview

CytoVerse uses foundation models to generate cell embeddings that enable:
- **Cell annotation**: Identify cell types by finding nearest neighbors in reference datasets
- **Visualization**: Project cells into 2D space using parametric UMAP
- **Search**: Fast approximate nearest neighbor search via IVFPQ across millions of reference cells

All models run entirely in the browser using ONNX Runtime Web, enabling privacy-preserving analysis without server dependencies.

## Model Integration Architecture

### Integration Pipeline

1. **Export to ONNX**: Convert PyTorch models to ONNX format for browser compatibility
2. **Preprocessing**: Normalize and transform raw gene expression counts
3. **Embedding Generation**: Forward pass through foundation model
4. **IVFPQ Indexing**: Train inverted file with product quantization for fast search
5. **PUMAP Training**: Train parametric UMAP for 2D visualization
6. **Browser Deployment**: Load ONNX models via WebAssembly/WebGPU

### Directory Structure

```
public/models/<model-name>/
├── embedding/
│   ├── preprocessing.onnx    # Preprocessing transformations
│   ├── embedding.onnx         # Core embedding model
│   ├── model.onnx            # Combined preprocessing + embedding
│   ├── genes.txt             # Gene vocabulary (symbols)
│   ├── genes.json            # Gene metadata (Ensembl IDs, symbols, tokens)
│   └── config.json           # Model configuration
├── ivfpq/
│   ├── ivf_centroids.bin     # IVF cluster centroids
│   ├── ivf_coarse.onnx       # Coarse IVF search
│   ├── partitions/           # Quantized reference embeddings
│   ├── pq_codebooks.bin      # Product quantization codebooks
│   ├── pq_encode.onnx        # PQ encoding
│   └── pq_distance.onnx      # PQ distance computation
└── pumap/
    ├── model.onnx            # Parametric UMAP projection
    ├── x.bin, y.bin          # Reference 2D coordinates
    └── metadata.json         # UMAP configuration
```

### Adding a New Model

To integrate a new foundation model:

1. **Create export script** in `backend/src/<model>_export_model.py`:
   - Load pretrained PyTorch model
   - Create preprocessing wrapper (normalization, tokenization)
   - Create embedding wrapper (forward pass extraction)
   - Export to ONNX with opset 17+
   - Export gene vocabulary and metadata

2. **Generate embeddings** for reference dataset:
   - Process reference h5ad files through model
   - Save embeddings and labels to disk

3. **Train IVFPQ index**:
   - Use `backend/src/ivfpq_train.py` with model embeddings
   - Configure n_clusters and n_sub based on dataset size

4. **Train parametric UMAP**:
   - Use `backend/src/pumap_train.py` to create 2D projection
   - Export UMAP model to ONNX

5. **Update frontend**: Add model to `models.txt` and load in `worker.ts`

---

## Integrated Models

### SCimilarity

**Type**: Custom foundation model for single-cell RNA-seq
**Training**: ~20M cells across diverse tissues and conditions
**Paper**: [Nature 2024](https://doi.org/10.1038/s41586-024-08411-y)
**Repository**: [Genentech/scimilarity](https://github.com/Genentech/scimilarity)

#### Architecture
- **Model Type**: Feed-forward neural network (non-transformer)
- **Input**: Full gene expression vector (~20,000 genes)
- **Embedding Dimension**: 128D
- **Parameters**: ~50-80M
- **Model Size**: ~10-20 MB (ONNX)

#### Preprocessing
```python
# Normalize total counts to 10,000
normalized = (expression / total_counts) * 10000
# Log transform
transformed = log1p(normalized)
```

**Complexity**: O(n_genes) - linear time

#### Performance
- **Inference**: 50-100ms per cell
- **FLOPs per cell**: ~50-100 million
- **Preprocessing**: ~10-20ms per cell
- **Browser-friendly**: Excellent (fast, small model)

#### Strengths
- 5-10x faster inference than transformer models
- Simpler preprocessing (no tokenization)
- Smaller embeddings (128D) reduce memory
- Proven IVFPQ integration with 70-90% recall at 2-20x speedup
- Designed specifically for single-cell RNA-seq

#### Use Cases
- Real-time cell annotation in browser (<100ms/cell)
- Large-scale dataset exploration (millions of cells)
- Resource-constrained environments
- When speed and efficiency are prioritized

#### Integration Status
✅ Fully integrated - default model in CytoVerse

---

### Geneformer

**Type**: Transformer foundation model for transcriptomics
**Training**: 30M cells (V1) or 104M cells (V2)
**Paper**: [Nature 2023](https://doi.org/10.1038/s41586-023-06139-9)
**Repository**: [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)

#### Architecture
- **Model Type**: BERT transformer (BertForMaskedLM)
- **Input**: Top-K ranked genes (2048 for V1, 4096 for V2)
- **Embedding Dimension**: 256D (V1) or 768D (V2)
- **Layers**: 6 (V1-10M) or 12 (V2)
- **Attention Heads**: 4 (V1) or 12 (V2)
- **Parameters**: 10M (V1-10M) to 316M (V2-30L)
- **Model Size**: 40 MB (V1) to 500+ MB (V2)

#### Preprocessing
```python
# 1. Median normalization (corpus-level)
normalized = expression / gene_medians

# 2. Rank genes by expression
top_k_indices = topk(normalized, k=2048)

# 3. Map to token IDs
token_ids = gene_to_token[top_k_indices]

# 4. Prepend CLS token (V2 only)
if v2:
    token_ids = [CLS_TOKEN] + token_ids
```

**Complexity**: O(n_genes × log k) due to TopK operation

#### Performance
- **Inference**: 250-500ms per cell (V1), 500-1000ms (V2)
- **FLOPs per cell**: ~8 billion (V1), ~160 billion (V2)
- **Preprocessing**: 50-100ms per cell
- **Browser-friendly**: Moderate (larger, slower than SCimilarity)

#### Key Features
- **Contextual embeddings**: Self-attention captures gene-gene interactions
- **Rich representations**: Larger 256D embeddings
- **Sequence modeling**: Processes ranked gene lists with positional encoding
- **Transfer learning**: Can fine-tune for downstream tasks

#### Computational Breakdown

| Component | V1 (10M params) | V2 (104M+ params) |
|-----------|-----------------|-------------------|
| Preprocessing | ~11M ops | ~11M ops |
| Attention (6-12 layers) | ~6.4B ops | ~150B ops |
| Feed-forward (6-12 layers) | ~1.6B ops | ~3.2B ops |
| **Total per cell** | **~8B ops** | **~160B ops** |

**Bottleneck**: Self-attention is O(seq_len²) - dominates 80% of compute

#### Strengths
- Richer, more contextual embeddings (256D/768D)
- Proven transformer architecture (BERT-based)
- Better for fine-tuning on specialized tasks
- Captures gene regulatory relationships via attention

#### Trade-offs
- 5-10x slower inference than SCimilarity
- Larger models (40-500 MB vs 10-20 MB)
- More complex preprocessing (TopK ranking)
- Higher memory requirements in browser

#### Use Cases
- When embedding quality is more important than speed
- Fine-tuning for specialized cell type classification
- Research applications requiring interpretable attention patterns
- Offline analysis where latency is not critical

#### Integration Status
✅ Export pipeline complete
🔄 Browser integration in progress
📋 IVFPQ training pending

---

## Model Comparison

### Quick Reference

| Feature | SCimilarity | Geneformer V1 | Geneformer V2 |
|---------|-------------|---------------|---------------|
| **Inference Speed** | 50-100ms | 250-500ms | 500-1000ms |
| **FLOPs/cell** | 50-100M | 8B | 160B |
| **Model Size** | 10-20 MB | 40 MB | 500+ MB |
| **Embedding Dim** | 128D | 256D | 768D |
| **Preprocessing** | Simple (O(n)) | Complex (O(n log n)) | Complex (O(n log n)) |
| **Architecture** | Feed-forward | Transformer (6L) | Transformer (12L) |
| **Browser Perf** | Excellent | Good | Moderate |
| **Use Case** | Real-time annotation | Balanced | Rich embeddings |

### Performance Comparison

**Relative Speed** (SCimilarity = 1x baseline):
- SCimilarity: 1x (fastest)
- Geneformer V1: 5-10x slower
- Geneformer V2: 10-20x slower

**Model Complexity**:
- SCimilarity: ~50-80M parameters
- Geneformer V1: ~110M parameters
- Geneformer V2: ~104-316M parameters

**Embedding Richness** (information density):
- Geneformer V2 (768D) > Geneformer V1 (256D) > SCimilarity (128D)

### When to Use Each Model

**Choose SCimilarity when:**
- Real-time browser performance is critical (<100ms/cell)
- Processing large datasets (millions of cells)
- Memory/bandwidth is constrained
- Standard cell annotation is sufficient

**Choose Geneformer V1 when:**
- Need richer embeddings (256D) than SCimilarity
- Have moderate computational resources
- Want transformer-based representations
- Latency <500ms is acceptable

**Choose Geneformer V2 when:**
- Embedding quality is paramount
- Fine-tuning for specialized tasks
- Offline/batch processing
- Computational resources are abundant

---

## Browser Runtime Considerations

### ONNX Runtime Web
- **Execution Providers**: WebAssembly (CPU), WebGPU (experimental)
- **Opset Version**: 17+ for maximum compatibility
- **Precision**: Float32 (quantization to INT8 possible for speed)
- **Batch Size**: Typically 8-32 cells for optimal throughput

### Memory Constraints
- **Typical Browser Limit**: 2-4 GB WebAssembly heap
- **Model Loading**: Models loaded on-demand, cached
- **IVFPQ Partitions**: Streamed from HTTP, ~10-50 MB per search
- **Activation Memory**: Proportional to batch size × embedding dim

### Performance Tips
1. **Use WebGPU** when available for 2-5x speedup on attention/matmul
2. **Batch cells** (16-32) for better GPU utilization
3. **Cache models** in browser storage to avoid re-downloading
4. **Lazy load** IVFPQ partitions only when needed
5. **Consider quantization** (INT8) for 2-4x speed at minimal accuracy loss

---

## References

### SCimilarity
- Paper: [Probabilistic Cell Type Annotation](https://doi.org/10.1038/s41586-024-08411-y), Nature 2024
- Code: [github.com/Genentech/scimilarity](https://github.com/Genentech/scimilarity)
- Model & Dataset: [Zenodo](https://zenodo.org/records/10685499) (~30GB)

### Geneformer
- Paper: [Transfer Learning Enables Predictions in Network Biology](https://doi.org/10.1038/s41586-023-06139-9), Nature 2023
- Models: [HuggingFace](https://huggingface.co/ctheodoris/Geneformer)
- Package: [github.com/ctheodoris/Geneformer](https://github.com/ctheodoris/Geneformer)

### IVFPQ & Approximate Search
- Product Quantization: [Jégou et al. 2011](https://ieeexplore.ieee.org/document/5432202)
- IVF-PQ Tutorial: [Towards Data Science](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd/)
- Browser Implementation: [TorchPQ](https://github.com/DeMoriarty/TorchPQ)
