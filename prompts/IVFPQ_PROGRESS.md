# IVFPQ Implementation Progress

## Overview

This document tracks the implementation of IVFPQ (Inverted File Index Product Quantization) for the Cell Space project. IVFPQ enables approximate nearest neighbor search on large-scale single-cell embedding datasets with significant memory and computational savings.

## Original Prompt

Let's develop an IVFPQ (Inverted File Index Product Quantization) module for this project. It should leverage PyTorch and support exporting the trained PQ model to ONNX where it will be used in a companion package for Approximate Nearest Neighbor (ANN) search in the browser. It should also support exporting the inverted file index of the training vectors with ids as a series of k parquet files plus a top level parquet file with the centroids which the browser side code will use to identify which partions should be pulled over the wire given a set of PQ quantized search vectors. As a reference and source of term/name choices see: https://towardsdatascience.com/similarity-search-with-ivfpq-9c6348fd4db3

As a start break the development down into a set of milestones that we can develop and test. For example we might initialy develop the PQ model, train it on a subset of vectors from the training set and export the model to onnx and verify we can run the same on the same train vectors in the browser. Then we can develop the code to quantize all the training vectors and generate the k index files...that's just an example but please think carefully and break this down into bite size chunk to develop and test.

## Architecture

The IVFPQ system consists of:

1. **Product Quantization (PQ)**: Compresses high-dimensional vectors by decomposing them into subvectors and quantizing each independently
2. **Inverted File Index (IVF)**: Partitions the dataset using coarse quantization for efficient search
3. **Browser Integration**: ONNX-based models for client-side inference and search

## Milestone Progress

### ✅ Milestone 1: Core PQ Module Implementation

**Status**: ✅ **COMPLETE**

**Deliverables**:

- ✅ `src/cellspace/ivfpq/pq.py` - Core PQ implementation in PyTorch
- ✅ `tests/test_pq.py` - Core unit tests
- ✅ `src/cellspace/scripts/pq_train.py` - Training and testing scripts
- ✅ `test_real_embeddings.py` - Real data validation
- ✅ ONNX export functionality
- ✅ Integration with existing project structure

**Key Features**:

- Configurable subquantizers (M) and centroids (K)
- K-means training with convergence checking
- PyTorch and ONNX inference
- ONNX-compatible distance computation (no `cdist` dependency)
- Comprehensive error checking and validation

**Usage**:

```bash
# Unit test pq module
python -m pytest tests/test_pq.py

# Test with real embeddings (adaptive to dimension)
python -m pytest tests/test_real_embeddings.py

# Train PQ on subset of data (fast)
make pq-train

# Train PQ on full dataset
make pq-train-full

# Test trained model
make pq-train-test
```

**Performance Results**:

- **Synthetic data (128D)**: 6.7% relative error, 85x compression
- **Real embeddings (128D)**: 1.25% relative error, 43x compression
- **SCimilarity config (512D)**: ~32x compression target
- Training time: ~30 seconds for 1K vectors, 16 subquantizers
- ONNX export: ✅ Validated, browser-ready

**Validation**:

- ✅ All synthetic tests pass
- ✅ Real embedding tests pass
- ✅ ONNX export matches PyTorch output
- ✅ Save/load functionality works
- ✅ Multiple vector dimensions supported

### 🔄 Milestone 2: Browser-side PQ Decoding

**Status**: Planned

**Tasks**:

- [ ] Implement TypeScript PQ decoder for ONNX.js
- [ ] Create browser test page for PQ validation
- [ ] Performance benchmarking against full precision
- [ ] WebGL/WebGPU optimization exploration

**Target deliverables**:

- `web/src/ivfpq/pq.ts` - Browser PQ implementation
- `web/public/pq_test.html` - Browser validation page
- Performance benchmarks

### 📋 Milestone 3: Inverted File Index (IVF) Implementation

**Status**: Planned

**Tasks**:

- [ ] Implement coarse quantization with k-means
- [ ] Create partition assignment and search logic
- [ ] Integration with PQ module
- [ ] Configurable number of partitions

**Target deliverables**:

- `src/cellspace/ivfpq/ivf.py` - IVF implementation
- Training pipeline integration

### 📋 Milestone 4: IVFPQ Training Pipeline

**Status**: Planned

**Tasks**:

- [ ] End-to-end IVFPQ training on SCimilarity dataset
- [ ] Export format for browser consumption:
  - ONNX models for PQ encoding/decoding
  - Parquet files for coarse centroids
  - Partitioned parquet files with PQ codes and cell IDs
- [ ] Configuration management

### 📋 Milestone 5: Browser-side ANN Search

**Status**: Planned

**Tasks**:

- [ ] Partition selection logic
- [ ] Progressive loading of relevant partitions
- [ ] Distance computation and ranking
- [ ] Integration with scatter plot interface

### 📋 Milestone 6: Integration and Optimization

**Status**: Planned

**Tasks**:

- [ ] Full web app integration
- [ ] UI controls for search parameters
- [ ] Performance optimization
- [ ] Memory usage optimization

### 📋 Milestone 7: Scale Testing

**Status**: Planned

**Tasks**:

- [ ] Full 23.4M cell dataset testing
- [ ] Production-ready configuration
- [ ] Performance validation

## Technical Details

### Product Quantization Parameters

For SCimilarity embeddings (512 dimensions):

| Parameter | Value | Description                |
| --------- | ----- | -------------------------- |
| d         | 512   | Embedding dimension        |
| m         | 64    | Number of subquantizers    |
| k         | 256   | Centroids per subquantizer |
| d_sub     | 8     | Subvector dimension (d/m)  |

This configuration provides:

- **Compression**: 32x (16KB → 512 bytes per vector)
- **Memory per vector**: 64 bytes (64 × 8-bit codes)
- **Search complexity**: O(m×k) = O(64×256) distance computations

### File Structure

```
src/cellspace/ivfpq/
├── __init__.py          # Module exports
├── pq.py               # Product Quantization implementation
├── ivf.py              # Inverted File Index (TODO)
└── ivfpq.py            # Combined IVFPQ (TODO)

src/cellspace/scripts/
└── pq_train.py         # Training and testing scripts

tests/
└── test_pq.py          # Unit tests

web/src/ivfpq/          # Browser implementations (TODO)
├── pq.ts               # TypeScript PQ decoder
└── search.ts           # ANN search implementation
```

## References

- [Similarity Search with IVFPQ](https://towardsdatascience.com/similarity-search-with-ivfpq-9c6348fd4db3)
- [IVFPQ + HNSW for Billion-scale Search](https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/)
- [Product Quantization for Nearest Neighbor Search](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf) - Original paper
- [Wikipedia Search-by-vibes](https://www.leebutterman.com/2023/06/01/offline-realtime-embedding-search.html) - JavaScript PQ implementation
