# Arc-Stack Model - Skip Rationale

## Decision: Arc-Stack is NOT compatible with CytoVerse browser architecture

## Browser Feasibility Test Results

### Test Configuration
- **Model**: Arc-Stack Large (444 MB ONNX)
- **Batch Size**: 256 cells × 15012 genes
- **Backends Tested**: WebGPU and WASM (CPU)

### Test Results

#### WebGPU Backend
**Status**: ❌ FAILED

**Error**:
```
[WebGPU] Kernel "[MatMul] /embedding/layers.0/cell_attn/MatMul" failed.
Error: Failed to generate kernel's output[0] with dims [65536,8,100,100].
Error: 568450800
```

**Cause**: WebGPU tensor dimension limits on Apple Silicon exceeded

#### WASM (CPU) Backend
**Status**: ❌ FAILED

**Error**:
```
Aborted(). Build with -sASSERTIONS for more info.
```

**Cause**: WebAssembly memory allocation failure

## Root Cause Analysis

### Current Implementation Problem

The `StackEmbeddingWrapper` creates artificial context by:
1. Repeating each input cell 256 times: `x.unsqueeze(1).expand(-1, 256, -1)`
2. This creates a batch of 256 × 256 = **65,536 cells**
3. Attention layers try to allocate tensors with first dimension = 65,536
4. Both WebGPU and WASM fail to allocate memory for these massive tensors

### Why This Approach Was Necessary

Arc-Stack's architecture **requires exactly 256 cells as context** for in-context learning:
- Model hardcodes `n_cells=256` in configuration
- Inter-cellular attention operates across all 256 cells in a batch
- Cannot process fewer than 256 cells without architectural changes (would require retraining)

### Architectural Incompatibility

**Arc-Stack's Design**:
- Processes cells in batches of exactly 256
- Uses in-context learning across cellular neighborhoods
- Sequential processing of 256-cell chunks
- Upsamples partial batches by repeating random cells

**CytoVerse's Architecture**:
- Streams cells individually or in small batches
- Processes arbitrary numbers of cells (often <256)
- No buffering or batching logic
- Designed for real-time, streaming inference

## Why Alternative Solutions Won't Work

### Option 1: True 256-Cell Batching
**Problem**: Would require buffering cells before inference
- Breaks CytoVerse's streaming architecture
- Adds latency (wait for 256 cells before processing)
- Defeats the purpose of client-side streaming
- Most queries are <256 cells

### Option 2: Reduce Batch Size in Model
**Problem**: Would require retraining the model
- Pre-trained model hardcodes n_cells=256
- Cannot be changed without retraining on new architecture
- Arc Institute would need to release different model variants

### Option 3: Backend-Only Inference
**Problem**: Defeats CytoVerse's privacy-preserving design
- Requires uploading data to server
- Loses client-side privacy guarantee
- Adds infrastructure complexity
- Contradicts core product value proposition

## Conclusion

**Arc-Stack is fundamentally incompatible with browser-based, streaming inference.**

The model's in-context learning design requires 256-cell batches, which:
1. Cannot be allocated in browser memory (tested on both WebGPU and WASM)
2. Would require architectural changes to CytoVerse (buffering, batching)
3. Would break the streaming, privacy-preserving design

**Recommendation**: Skip Arc-Stack implementation and focus on models that work well in browser:
- ✅ SCimilarity - Works excellently in browser
- ✅ Geneformer - Works well in browser
- ❌ Arc-Stack - Incompatible with browser architecture

## Artifacts Created During Investigation

The following artifacts were created during the feasibility investigation and should be removed:

- `backend/src/arc_stack_export_model.py` - Export script
- `public/models/arc-stack/` - ONNX models and metadata
- `public/test_arc_stack_browser.html` - Browser test
- `explore_stack_model.py` - Model exploration script
- `.claude/plans/rustling-toasting-wirth.md` - Implementation plan
- Arc-stack dependency in `pyproject.toml`

## Date
2026-01-13
