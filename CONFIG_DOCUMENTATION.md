# Cytoverse Configuration System Documentation

## Overview

This document describes the production-level configuration management system implemented for Cytoverse using Hydra/OmegaConf for Python and a TypeScript configuration manager for the frontend.

## Configuration Architecture

### 1. Python Configuration (Hydra/OmegaConf)

**Location**: `config/` directory with environment-specific files

**Structure**:
```
config/
├── config.yaml                 # Main configuration file
├── python/
│   └── default.yaml            # Python/ML pipeline defaults
├── frontend/
│   └── default.yaml            # Frontend/worker defaults
└── environments/
    ├── dev.yaml                # Development overrides
    ├── staging.yaml            # Staging overrides
    └── prod.yaml               # Production overrides
```

**Usage**:
```python
from cytoverse.config import get_config, get_ivfpq_config

# Get full config for environment
config = get_config('prod')

# Get IVFPQ-specific config
ivfpq_config = get_ivfpq_config('prod')

# With overrides
config = get_config('dev', overrides=['ivfpq.model.n_partitions=128'])
```

### 2. Frontend Configuration (TypeScript)

**Location**: `src/config.ts`

**Usage**:
```typescript
import { configManager, workerConfig, modelConfig } from './config'

// Get worker configuration
const batchSize = workerConfig.batch_size
const useWebGPU = workerConfig.performance.use_webgpu

// Environment detection is automatic
const env = configManager.getEnvironment()
```

## Environment Configurations

### Development (`dev`)
- **Purpose**: Local development with debugging enabled
- **Python**: Smaller datasets, more logging, fewer iterations
- **Frontend**: WebAssembly backend, debug logging, longer timeouts
- **Key Settings**:
  - Training vectors: 100K (vs 10M in prod)
  - Debug enabled: true
  - WebGPU disabled for better debugging

### Staging (`staging`)
- **Purpose**: Pre-production testing with realistic data
- **Python**: Mid-size datasets, validation enabled
- **Frontend**: Production features with debug monitoring
- **Key Settings**:
  - Training vectors: 2M
  - Performance monitoring: enabled
  - WebGPU: enabled for testing

### Production (`prod`)
- **Purpose**: Full-scale production deployment
- **Python**: Full datasets, hyperparameter optimization
- **Frontend**: Maximum performance, minimal logging
- **Key Settings**:
  - Training vectors: 10M
  - Debug disabled: true
  - WebGPU: enabled
  - More partitions searched for better results

## Configuration Parameters

### IVFPQ Model Configuration
- `n_partitions`: Number of IVF partitions (128 dev, 256 staging, 512 prod)
- `pq_m`: Product quantization subspaces (16-32)
- `pq_k`: Centroids per subspace (256)
- `n_probe`: Partitions to search (2 dev, 4 staging, 8 prod)

### Worker Configuration
- `batch_size`: Processing batch size (32)
- `num_nearest_neighbors`: K for search (50)
- `num_partitions_to_search`: Performance vs accuracy tradeoff
- `use_webgpu`: Hardware acceleration preference

### Performance Tuning
- `max_concurrent_batches`: Parallelism control
- `max_embeddings_cache`: Memory management
- `timeout_ms`: Network timeout settings
- `retry_attempts`: Error recovery

## Usage Examples

### Python Training Script
```bash
# Development training
CYTOVERSE_ENV=dev python scripts/ivfpq_train_configured.py train scimilarity

# Production training
CYTOVERSE_ENV=prod python scripts/ivfpq_train_configured.py train scimilarity

# With overrides
CYTOVERSE_ENV=prod python scripts/ivfpq_train_configured.py train scimilarity \
  --config-overrides model.n_partitions=1024 training.max_iters=100
```

### Makefile Targets
```bash
# Configuration-based training
make ivfpq-train-dev model_id=scimilarity
make ivfpq-train-staging model_id=scimilarity  
make ivfpq-train-prod model_id=scimilarity

# Configuration validation
make config-validate-all

# Configuration info
make config-info-prod
```

### Frontend Build
```bash
# Development build
npm run build:dev

# Staging build  
npm run build:staging

# Production build
npm run build:prod
```

## Environment Variables

### Python
- `CYTOVERSE_ENV`: Environment name (dev/staging/prod)

### Frontend (Vite)
- `VITE_CYTOVERSE_ENV`: Environment name
- `VITE_DEBUG`: Debug mode toggle
- `VITE_LOG_LEVEL`: Logging level

## Configuration Validation

The system includes validation for:
- Required configuration sections
- Parameter value ranges
- Cross-parameter consistency (e.g., n_probe ≤ n_partitions)
- Environment-specific constraints

## Migration from Hard-coded Values

### Before (Hard-coded)
```typescript
// worker.ts
const NUM_NEAREST_NEIGHBORS = 50
const NUM_PARTITIONS_TO_SEARCH = 2
const BATCH_SIZE = 32
```

### After (Configuration-based)
```typescript
// worker.ts
import { workerConfig } from './config'

const NUM_NEAREST_NEIGHBORS = workerConfig.search.num_nearest_neighbors
const NUM_PARTITIONS_TO_SEARCH = workerConfig.search.num_partitions_to_search
const BATCH_SIZE = workerConfig.batch_size
```

## Benefits

1. **Environment Separation**: Clear dev/staging/prod configurations
2. **Type Safety**: TypeScript interfaces for frontend config
3. **Validation**: Automatic configuration validation
4. **Flexibility**: Runtime overrides and customization
5. **Documentation**: Self-documenting configuration structure
6. **Maintainability**: Centralized configuration management
7. **Production Ready**: Production-optimized defaults

## Dependencies Added

### Python
- `hydra-core>=1.3.0`: Configuration management framework
- `omegaconf>=2.3.0`: Configuration object management

### Frontend
- Native TypeScript configuration manager (no additional dependencies)

## Files Modified/Added

### Configuration Files
- `config/config.yaml`: Main configuration
- `config/python/default.yaml`: Python defaults
- `config/frontend/default.yaml`: Frontend defaults  
- `config/environments/{dev,staging,prod}.yaml`: Environment configs

### Python
- `cytoverse/config.py`: Configuration management module
- `scripts/ivfpq_train_configured.py`: Configuration-aware training script

### Frontend
- `src/config.ts`: TypeScript configuration manager
- `src/worker.ts`: Updated to use configuration

### Build System
- `Makefile`: Added configuration-based targets
- `package.json`: Added environment-specific build scripts
- `.env.*`: Environment variable files
- `pyproject.toml`: Added Hydra/OmegaConf dependencies

This configuration system provides a solid foundation for production deployment while maintaining flexibility for development and testing.