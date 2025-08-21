# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Frontend (Vue/TypeScript)
```bash
npm install           # Install dependencies including workspace ivfpq module
npm run dev          # Start Vite dev server on http://localhost:5173
npm run build        # Type-check with vue-tsc and build for production
npm run type-check   # Run TypeScript type checking only
npm run lint         # Run ESLint on all files
npm run preview      # Preview production build
```

### Testing
```bash
npm run test         # Run Vitest unit tests
npm run test:ui      # Run Vitest with UI
npx vitest run ivfpq # Run IVFPQ TypeScript tests specifically
npx playwright test  # Run E2E tests (starts dev server automatically)

# Python tests
python -m pytest ivfpq/python/tests  # IVFPQ Python tests
python -m pytest tests/unit          # Unit tests
python -m pytest tests/unit/test_performance.py --capture=no --log-cli-level=DEBUG  # Performance benchmarks
```

### Python Environment
```bash
uv venv                    # Create virtual environment
source .venv/bin/activate  # Activate environment
uv sync                    # Install dependencies from pyproject.toml
```

### Model Generation
```bash
make scimilarity      # Generate all artifacts for scimilarity model
make update-models-list  # Update models.txt after adding new models
```

## Architecture Overview

### Core Data Processing Pipeline
The application performs single-cell RNA-seq analysis entirely in the browser:

1. **H5AD File Processing** (`src/worker.ts`): 
   - Streams h5ad files using h5wasm without uploading
   - Extracts gene expression data (sparse or dense format)
   - Processes cells in batches of 64

2. **Embedding Generation**:
   - Uses SCimilarity ONNX model to convert ~35-60k gene expressions to 128-1024 dim embeddings
   - Models located in `public/models/{modelID}/embedding/`
   - Normalizes embeddings to unit sphere

3. **Visualization & Search**:
   - **PUMAP**: Maps embeddings to 2D coordinates for visualization
   - **IVFPQ**: Performs approximate nearest neighbor search across 20M+ reference cells
     - Inverted File (IVF) partitions data into 256-316 clusters
     - Product Quantization (PQ) compresses embeddings to 16 bytes
     - Searches top N_PROBE partitions for efficiency

4. **Real-time Updates**:
   - Worker sends batch updates via `cell_batch_update` messages
   - App.vue maintains cell positions and labels in memory
   - ScatterPlotWebGL renders using regl-scatterplot

### Key Components

**Frontend (`src/`)**:
- `App.vue`: Main application controller, manages state and worker communication
- `ScatterPlotWebGL.vue`: High-performance scatter plot visualization
- `worker.ts`: Web Worker handling all data processing in background

**IVFPQ Module (`ivfpq/`)**:
- Dual Python/TypeScript implementation
- Python: Training and model generation
- TypeScript: Browser runtime for search operations
- Published as npm workspace package `@cytoverse/ivfpq`

**Model Artifacts (`public/models/{modelID}/`)**:
- `embedding/`: ONNX models for gene expression → embedding
- `pumap/`: ONNX model for 2D mapping + Arrow files with reference data
- `ivfpq/`: IVF centroids, PQ codebooks, and partitioned reference embeddings

### Worker Communication Protocol
Messages between App.vue and worker.ts:
- `start`: Initialize processing with model and file
- `cell_batch_update`: Batch of processed cells with coordinates and labels
- `progress`: Processing status updates
- `status`: Text status messages
- `finished`: Processing complete
- `error`: Error notifications

### State Management
- Training data (reference cells) loaded once at startup from Arrow files
- Test data (user cells) accumulated incrementally as processed
- IndexedDB stores results for CSV export
- Cell ID → array index mapping maintained for efficient updates

## Critical Implementation Details

- Always process cells in batches for performance (BATCH_SIZE = 64)
- Embeddings must be normalized to unit sphere before IVFPQ search
- Use Int32Array for category data to match Arrow format
- WebGPU currently disabled due to Apple Silicon limitations (max 16384 input vectors)
- Worker must be terminated and recreated when switching models
- PUMAP coordinates need denormalization using metadata's xCenter, yCenter, maxRange

## Model Training Pipeline

1. Export embeddings from reference dataset (e.g., SCimilarity)
2. Train parametric UMAP on stratified subset
3. Train IVFPQ index on embeddings
4. Export all models to ONNX format
5. Generate Arrow files for reference data
6. Place in `public/models/{modelID}/`