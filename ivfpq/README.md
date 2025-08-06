# IVFPQ TypeScript Implementation

This TypeScript package provides client-side implementations of IVFPQ (Inverted File with Product Quantization) algorithms using ONNX models.

## Running Unit Tests

The TypeScript vitest unit tests validate that the ONNX models work correctly in a browser-like environment.

### Prerequisites

1. **Generate Python artifacts**: First, you need to run the Python tests to generate the required ONNX models and artifacts:
   ```bash
   cd ivfpq/python
   pytest tests/ -v
   ```
   This creates the necessary ONNX models in `ivfpq/python/tests/artifacts/`:
   - `kmeans_init.onnx` - K-means initialization model
   - `kmeans_iteration.onnx` - K-means iteration model  
   - `pq_encode.onnx` - Product quantization encoding model
   - `pq_decode.onnx` - Product quantization decoding model
   - `pq_distance.onnx` - PQ distance computation model
   - `pq_distance_base.onnx` - Base PQ distance model
   - `pq_metadata.json` - PQ model metadata
   - `pq_codebooks.bin` - PQ codebooks data

2. **Install dependencies**: 
   ```bash
   cd ivfpq/typescript
   npm install
   ```

### Running Tests

From the TypeScript directory (`ivfpq/typescript/`):

```bash
# Run tests once
npm test:run

# Run tests in watch mode  
npm test

# Run with UI
npm run test -- --ui
```

From the root directory:
```bash
# Run all tests (includes TypeScript tests via workspace)
npm test

# Run with UI
npm run test:ui
```

### Test Structure

- **`tests/kmeans.test.ts`**: Tests ONNX K-means models (initialization and iteration)
- **`tests/pq.test.ts`**: Tests ONNX Product Quantization models (encode, decode, distance)

The tests generate their own test data and validate:
- Model loading and inference
- Output shapes and constraints
- Functional correctness
- Performance characteristics
- Edge cases

### Troubleshooting

**"Cannot find module" errors**: Ensure you've run the Python tests first to generate the ONNX artifacts.

**ONNX runtime errors**: The tests require `onnxruntime-web` which should be automatically installed with `npm install`.

**Path issues**: The tests expect artifacts at `../python/tests/artifacts/` relative to the test files.