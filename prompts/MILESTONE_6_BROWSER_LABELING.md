# Milestone 6: Browser Labeling Implementation

## Overview

This milestone implements real-time cell labeling in the browser during embedding processing. The system will use the trained IVFPQ models to provide immediate cell type classification as embeddings are generated, enabling live visualization and feedback.

### Data Flow

```
35k dimension float32 Test vector (batch streamed from h5ad)
↓
Embedding ONNX model (SCimilarity now, others later)
↓
128-1024 dimension float32 Embedding vector
↓
PQ Encoding ONNX model + UMAP ONNX mapping model
↓
embedder.ts emits: (test_vector_id, pq_embedding, umap_coordinates)
↓
labeler.ts (if enabled): IVF Search using pq_embedding
↓
labeler.ts emits: (test_vector_id, pq_embedding, umap_coordinates, train_vector_id)
↓
App.tsx: Draw 2d coordinates (existing behavior continues)
```

## Implementation Phases

### Phase 1: PQ Encoding in Embedder

Add PQ encoding to embedder.ts so it emits batches of (test_vector_id, pq_embedding, umap_coordinates) instead of just embeddings and coordinates.

### Phase 2: Create Labeler Worker

Create labeler.ts web worker that receives the PQ embedding tuples from embedder.ts when labeling is enabled.

### Phase 3: IVF Search Implementation

Implement IVF search in labeler.ts to find the closest training vector ID for each PQ embedding using the IVFPQ assets.

### Phase 4: UI Toggle Control

Add a checkbox in App.tsx to enable/disable the labeling pipeline. When disabled, embedder.ts output goes directly to visualization.

### Phase 5: Integration Testing

Connect all components: embedder.ts → labeler.ts → App.tsx with the complete pipeline emitting (test_vector_id, pq_embedding, umap_coordinates, train_vector_id).

## Success Criteria

By the end of Milestone 6:

- ✅ Real-time cell labeling during embedding processing
- ✅ Live label statistics in sidebar
- ✅ Scatter plot colored by predicted cell types
- ✅ < 10ms labeling latency per cell
- ✅ No impact on main embedding pipeline performance
- ✅ Graceful handling of labeling errors/failures

## Risk Mitigation

**Performance Risk**: Real-time processing might slow down embedding

- **Mitigation**: Phase 6.1 validates performance impact early
- **Fallback**: Async labeling with delayed visualization updates

**Complexity Risk**: Full IVFPQ search might be too complex for browser

- **Mitigation**: Phase 6.2 isolates search implementation
- **Fallback**: Simplified k-NN search or pre-computed lookup tables

**Memory Risk**: Loading full IVFPQ database in browser

- **Mitigation**: Lazy loading of partitions, memory monitoring
- **Fallback**: Server-side labeling service for complex cases
