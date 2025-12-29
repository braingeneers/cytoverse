# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CytoVerse is a browser-based single-cell RNA-seq analysis platform that enables cell annotation using foundation model embeddings entirely client-side. The application streams h5ad files from local storage without uploading data or requiring server computation, making it privacy-preserving and highly scalable.

**Key Innovation**: Complete ML pipeline running in the browser using WebAssembly, ONNX Runtime, and WebGPU for high-performance vector search and embedding inference.

## Architecture

### Dual-Language System

The codebase is split into two distinct environments:

1. **Python Pipeline** (`scripts/`) - Offline data preparation
   - Exports SCimilarity foundation model to ONNX format
   - Processes reference datasets through SCimilarity
   - Trains IVFPQ (Inverted File with Product Quantization) index for fast vector search
   - Trains parametric UMAP model for 2D visualization
   - Exports all artifacts as binary files and ONNX models

2. **TypeScript Frontend** (`src/`) - Real-time browser execution
   - Loads h5ad files via h5wasm
   - Computes embeddings using ONNX Runtime Web
   - Performs IVFPQ search for cell annotation
   - Renders visualizations using WebGL
   - Manages user-created reference indexes in IndexedDB

### Data Flow

**Python (Training/Preparation)**:
```
h5ad file → SCimilarity embedding → IVFPQ training → PUMAP training → ONNX models + Binary artifacts
```

**Browser (Runtime)**:
```
h5ad file → h5wasm → Gene expression → ONNX embedding model →
SCimilarity embeddings → PUMAP model → 2D coordinates →
IVFPQ search → Cell labels → WebGL visualization
```

### Core Components

**Python Scripts** (`scripts/`):
- `scimilarity_export_model.py` - Export embedding model to ONNX (3-stage: preprocessing → embedding → projection)
- `scimilarity_to_embeddings.py` - Export embeddings/labels from SCimilarity's TileDB database
- `h5ad_to_embeddings.py` - Process custom h5ad files through SCimilarity
- `ivfpq_train.py` - Train IVFPQ index with residual vectors
- `pumap_train.py` - Train parametric UMAP, map embeddings, export to ONNX
- `ivfpq.py` - Core IVFPQ implementation
- `pq.py` - Product Quantization implementation

**TypeScript Source** (`src/`):
- `App.vue` - Main Vue 3 + Vuetify interface
- `worker.ts` - Web Worker for embedding computation and IVFPQ search
- `ivfpq.ts` - Browser IVFPQ implementation using ONNX Runtime
- `pq.ts` - Product Quantization distance computation
- `userIvfpq.ts` - User-generated reference index management
- `userIndexService.ts` - IndexedDB storage for user references
- `ScatterPlotWebGL.vue` - WebGL-based 2D scatter plot

### IVFPQ Architecture

The search system uses **residual vectors** for improved accuracy:
- IVF partitions vectors using KMeans clustering
- Product Quantization compresses residuals into m subspaces (typically 32)
- Asymmetric distance computation between query and quantized database
- Partitions stored as binary files, loaded on-demand via HTTP

**Search Pipeline**:
1. Coarse Search: Find k nearest centroids using ONNX model
2. Partition Loading: Fetch relevant partitions from HTTP
3. Fine Search: Compute PQ distances within partitions
4. Ranking: Return top-k nearest neighbors with labels

### Web Worker Pattern

To keep the UI responsive:
- Main thread handles UI and visualization only
- Web Worker performs CPU-intensive ML operations (embedding, search)
- Message-based communication with progress updates
- Supports WebGPU acceleration when available

### User References

Users can save labeled cells as references for collaborative discovery:
- Embeddings stored in IndexedDB
- PQ codes and partition assignments cached
- Reuses base model's centroids and codebooks
- Enables identification of overlapping or complementary assays

## Development Commands

### Setup
```bash
# Python environment
# This project uses uv (https://docs.astral.sh/uv/) for Python dependency management
# uv manages dependencies in a virtualenv and provides faster installs than pip

uv venv                       # Create virtualenv in .venv/
source .venv/bin/activate     # Activate virtualenv
uv sync                       # Install all dependencies from pyproject.toml

# Node.js dependencies
npm install
npx playwright install
```

### Running Python Scripts with uv

```bash
# Option 1: Activate virtualenv first, then run normally
source .venv/bin/activate
python scripts/scimilarity_export_model.py [args]

# Option 2: Use uv run to automatically use the virtualenv (no activation needed)
uv run python scripts/scimilarity_export_model.py [args]
uv run pytest tests/unit
uv run black scripts/
uv run flake8 scripts/
```

### Managing Python Dependencies

```bash
# Add a new dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Remove a dependency
uv remove <package-name>

# Update all dependencies
uv sync --upgrade

# Lock dependencies without installing
uv lock
```

### Data Preparation

```bash
# Full pipeline: export model, embeddings, train IVFPQ, train PUMAP
make scimilarity

# Individual steps
make scimilarity-model       # Export embedding model to ONNX
make scimilarity-embeddings  # Export embeddings from TileDB
make ivfpq-train            # Train IVFPQ index
make pumap                  # Train + map + export PUMAP model (includes pumap-train, pumap-map, pumap-export)

# Create custom reference from h5ad file
export model_id="my_reference"
python scripts/h5ad_to_embeddings.py \
    <path_to_h5ad> \
    data/models/scimilarity/model_v1.1 \
    data/references/$model_id \
    --labels <obs_column_1> \
    --labels <obs_column_2>

make ivfpq-train pumap
```

### Development & Testing

```bash
# Start dev server (http://localhost:5173)
npm run dev

# Type checking
npm run type-check

# Run all tests (unit + e2e, Python + TypeScript)
make test

# Individual test suites
npx vitest run tests/unit              # TypeScript unit tests
python -m pytest tests/unit            # Python unit tests
python -m pytest tests/e2e             # Python E2E tests
npx playwright test tests/e2e/run.spec.ts  # Browser E2E tests

# Performance benchmarks
make benchmark
```

### Build & Deploy

```bash
# Production build
npm run build  # or: make build

# Preview production build locally
npm run preview

# Deploy to server (configured for cells-test.gi.ucsc.edu/cytoverse/)
make deploy
```

### Code Quality

```bash
# Python
black scripts/          # Format Python code
flake8 scripts/         # Lint Python code

# TypeScript/Vue
npx prettier --write .  # Format code (88 char width)
npx eslint .           # Lint code
```

## Key Patterns & Conventions

### ONNX Model Export
All PyTorch models are exported to ONNX for browser compatibility:
- Models must be exportable with torch.onnx.export
- Use opset_version 17 or higher
- Test exports with onnxruntime before deploying
- Embedding model is 3-stage: preprocessing.onnx → embedding.onnx → model.onnx

### Type Safety
- Python: Type hints throughout, enforced with mypy
- TypeScript: Strict mode enabled
- All data structures have explicit interface definitions

### Memory Management
- Python: Use memory-mapped arrays (np.memmap) for large datasets
- Browser: Stream h5ad files, process in batches
- Default batch size: 32 cells
- IndexedDB for efficient user reference storage

### Reproducibility
- Use `DEFAULT_SEED = 42` for all random operations
- Enable deterministic algorithms when possible
- Use stratified sampling for PUMAP training to preserve label distributions

### Code Style
- Python: Black formatting (88 char line width), Flake8 linting
- TypeScript: Prettier formatting (88 char width), ESLint
- Vue: Single File Components with TypeScript in `<script setup lang="ts">`

### Jupyter Notebooks
- Run notebooks from project root directory
- Use relative paths like `data/` (not `../data/`)
- Ensures consistent paths across development environments

## Important Technical Details

### Cross-Origin Headers
The app requires specific CORS headers for WebAssembly and SharedArrayBuffer:
```
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy: same-origin
```
These are configured in the Vite server and must be present on production servers.

### Data Directory Structure
- `data/` is a symlink to `/Users/rcurrie/data/cytoverse` (local dev only)
- Contains large model files (~30GB) and datasets
- Git-ignored to avoid repository bloat
- `public/models/` contains web-ready model artifacts (~1.2GB)

### Model Artifacts
Models are served from `public/models/` and copied to `dist/models/` during build:
- `models.txt` lists available models (one per line)
- Each model directory contains:
  - `embedding/` - ONNX embedding models + genes.txt vocabulary
  - `ivfpq/` - Search index (centroids, codebooks, partitions/)
  - `pumap/` - UMAP projection model + reference coordinates

### Performance Characteristics
- Search Speed: ~4ms per query (50 neighbors, 4 partitions)
- Index Size: ~4,835 partitions for 20M+ cells
- Compression: 32x via Product Quantization (256D → 32 bytes)
- Browser Streaming: Unlimited dataset size via h5wasm

## Development Philosophy

**Production-Safe Changes Only**:

1. **Analyze First**
   - Read codebase context before making changes
   - Find exact insertion points
   - Identify what could break downstream

2. **Minimal Changes Only**
   - Touch only what's required for the task
   - No refactoring, cleanup, or extras
   - Follow existing patterns exactly
   - Use type hints in Python/TypeScript

3. **Production Discipline**
   - If it works, don't touch it
   - Preserve all existing behavior
   - No new abstractions unless explicitly required
   - No logging/comments unless critical
   - Smallest working change wins

4. **Verify Impact**
   - Check downstream effects
   - Ensure no regressions
   - Consider edge cases
   - When ambiguous, do less

## Technology Stack

**Python**: Python 3.12, PyTorch, SCimilarity, scikit-learn, AnnData/Scanpy, ONNX/ONNXRuntime, TileDB

**Browser**: Vue 3, Vuetify 3, TypeScript, Vite, ONNX Runtime Web, h5wasm, regl-scatterplot, IndexedDB/idb

**Testing**: Vitest (TypeScript/JavaScript unit tests), Pytest (Python unit tests), Playwright (E2E browser tests)

## References

- SCimilarity: [paper](https://doi.org/10.1038/s41586-024-08411-y), [repo](https://github.com/Genentech/scimilarity)
- IVFPQ: Inverted File with Product Quantization for approximate nearest neighbor search
- Parametric UMAP: Neural network-based dimensionality reduction
- h5wasm: WebAssembly-based HDF5 reader for browsers
