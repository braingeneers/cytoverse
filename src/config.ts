/**
 * Configuration management for Cytoverse frontend
 * 
 * This module provides centralized configuration management for the TypeScript/Vue frontend,
 * including web worker settings, model parameters, and environment-specific configurations.
 */

// Configuration interfaces
export interface WorkerConfig {
  batch_size: number
  search: {
    num_nearest_neighbors: number
    num_partitions_to_search: number
  }
  performance: {
    use_webgpu: boolean
    fallback_to_wasm: boolean
    max_concurrent_batches: number
  }
  memory: {
    max_embeddings_cache: number
    gc_threshold: number
  }
}

export interface ModelConfig {
  base_url: string
  timeout_ms: number
  retry_attempts: number
  retry_delay_ms: number
  types: {
    embedding: {
      format: string
      providers: string[]
    }
    mapping: {
      format: string
      providers: string[]
    }
    ivfpq: {
      format: string
      lazy_loading: boolean
    }
  }
}

export interface ONNXConfig {
  webgpu: {
    enabled: boolean
    device_preference: string
  }
  wasm: {
    num_threads: number
    simd: boolean
  }
}

export interface DebugConfig {
  enabled: boolean
  log_level: string
  performance_monitoring: boolean
  memory_tracking: boolean
}

export interface FrontendConfig {
  worker: WorkerConfig
  models: ModelConfig
  onnx: ONNXConfig
  debug: DebugConfig
  compatibility: {
    min_chrome_version: number
    min_firefox_version: number
    min_safari_version: number
  }
}

export interface AppConfig {
  name: string
  version: string
  environment: string
}

export interface CytoverseConfig {
  app: AppConfig
  frontend: FrontendConfig
}

// Environment detection
function getEnvironment(): string {
  // Check for build-time environment variable
  if (typeof window !== 'undefined' && (window as any).__CYTOVERSE_ENV__) {
    return (window as any).__CYTOVERSE_ENV__
  }
  
  // Check for import.meta.env in Vite context
  if (typeof import.meta !== 'undefined' && import.meta.env) {
    if (import.meta.env.VITE_CYTOVERSE_ENV) {
      return import.meta.env.VITE_CYTOVERSE_ENV
    }
    
    // Check for runtime environment indicators
    if (import.meta.env.DEV) {
      return 'dev'
    }
    
    if (import.meta.env.PROD) {
      // Check if we're in staging based on hostname or other indicators
      if (typeof window !== 'undefined' && window.location.hostname.includes('staging')) {
        return 'staging'
      }
      return 'prod'
    }
  }
  
  return 'dev'
}

// Default configurations for each environment
const defaultConfigs: Record<string, Partial<CytoverseConfig>> = {
  dev: {
    app: {
      name: 'cytoverse',
      version: '0.1.0',
      environment: 'dev'
    },
    frontend: {
      worker: {
        batch_size: 32,
        search: {
          num_nearest_neighbors: 50,
          num_partitions_to_search: 2
        },
        performance: {
          use_webgpu: false, // Use WebAssembly for better debugging
          fallback_to_wasm: true,
          max_concurrent_batches: 2
        },
        memory: {
          max_embeddings_cache: 50000,
          gc_threshold: 0.7
        }
      },
      models: {
        base_url: './models',
        timeout_ms: 60000, // Longer timeout for development
        retry_attempts: 3,
        retry_delay_ms: 1000,
        types: {
          embedding: {
            format: 'onnx',
            providers: ['wasm', 'webgpu']
          },
          mapping: {
            format: 'onnx',
            providers: ['wasm', 'webgpu']
          },
          ivfpq: {
            format: 'arrow',
            lazy_loading: true
          }
        }
      },
      onnx: {
        webgpu: {
          enabled: false,
          device_preference: 'high-performance'
        },
        wasm: {
          num_threads: 2,
          simd: true
        }
      },
      debug: {
        enabled: true,
        log_level: 'DEBUG',
        performance_monitoring: true,
        memory_tracking: true
      },
      compatibility: {
        min_chrome_version: 90,
        min_firefox_version: 88,
        min_safari_version: 14
      }
    }
  },
  
  staging: {
    app: {
      name: 'cytoverse',
      version: '0.1.0',
      environment: 'staging'
    },
    frontend: {
      worker: {
        performance: {
          use_webgpu: true, // Test production features
          fallback_to_wasm: true,
          max_concurrent_batches: 4
        }
      },
      models: {
        base_url: './models',
        timeout_ms: 20000,
        retry_attempts: 3,
        retry_delay_ms: 1000,
        types: {
          embedding: {
            format: 'onnx',
            providers: ['webgpu', 'wasm']
          },
          mapping: {
            format: 'onnx',
            providers: ['webgpu', 'wasm']
          },
          ivfpq: {
            format: 'arrow',
            lazy_loading: true
          }
        }
      },
      debug: {
        enabled: true,
        log_level: 'INFO',
        performance_monitoring: true,
        memory_tracking: false
      }
    }
  },
  
  prod: {
    app: {
      name: 'cytoverse',
      version: '0.1.0',
      environment: 'prod'
    },
    frontend: {
      worker: {
        search: {
          num_nearest_neighbors: 50,
          num_partitions_to_search: 4 // Search more partitions for better results
        },
        performance: {
          use_webgpu: true,
          fallback_to_wasm: true,
          max_concurrent_batches: 8
        },
        memory: {
          max_embeddings_cache: 100000,
          gc_threshold: 0.8
        }
      },
      models: {
        base_url: './models',
        timeout_ms: 15000,
        retry_attempts: 5,
        retry_delay_ms: 1000,
        types: {
          embedding: {
            format: 'onnx',
            providers: ['webgpu', 'wasm']
          },
          mapping: {
            format: 'onnx',
            providers: ['webgpu', 'wasm']
          },
          ivfpq: {
            format: 'arrow',
            lazy_loading: true
          }
        }
      },
      onnx: {
        webgpu: {
          enabled: true,
          device_preference: 'high-performance'
        },
        wasm: {
          num_threads: 4,
          simd: true
        }
      },
      debug: {
        enabled: false,
        log_level: 'WARN',
        performance_monitoring: false,
        memory_tracking: false
      }
    }
  }
}

// Deep merge utility
function deepMerge(target: any, source: any): any {
  const result = { ...target }
  
  for (const key in source) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      result[key] = deepMerge(target[key] || {}, source[key])
    } else {
      result[key] = source[key]
    }
  }
  
  return result
}

/**
 * Configuration manager for Cytoverse frontend
 */
export class ConfigManager {
  private config: CytoverseConfig
  private environment: string
  
  constructor(environment?: string) {
    this.environment = environment || getEnvironment()
    this.config = this.loadConfig()
  }
  
  private loadConfig(): CytoverseConfig {
    const baseConfig = defaultConfigs.dev
    const envConfig = defaultConfigs[this.environment] || {}
    
    return deepMerge(baseConfig, envConfig) as CytoverseConfig
  }
  
  /**
   * Get the complete configuration
   */
  getConfig(): CytoverseConfig {
    return this.config
  }
  
  /**
   * Get frontend-specific configuration
   */
  getFrontendConfig(): FrontendConfig {
    return this.config.frontend
  }
  
  /**
   * Get worker configuration
   */
  getWorkerConfig(): WorkerConfig {
    return this.config.frontend.worker
  }
  
  /**
   * Get model loading configuration
   */
  getModelConfig(): ModelConfig {
    return this.config.frontend.models
  }
  
  /**
   * Get ONNX runtime configuration
   */
  getONNXConfig(): ONNXConfig {
    return this.config.frontend.onnx
  }
  
  /**
   * Get debug configuration
   */
  getDebugConfig(): DebugConfig {
    return this.config.frontend.debug
  }
  
  /**
   * Override configuration values
   */
  override(overrides: Partial<CytoverseConfig>): void {
    this.config = deepMerge(this.config, overrides)
  }
  
  /**
   * Get current environment
   */
  getEnvironment(): string {
    return this.environment
  }
  
  /**
   * Check if debugging is enabled
   */
  isDebugEnabled(): boolean {
    return this.config.frontend.debug.enabled
  }
  
  /**
   * Log configuration summary
   */
  logConfigSummary(): void {
    if (this.isDebugEnabled()) {
      console.group('🔧 Cytoverse Configuration')
      console.log(`Environment: ${this.environment}`)
      console.log(`WebGPU Enabled: ${this.config.frontend.worker.performance.use_webgpu}`)
      console.log(`Batch Size: ${this.config.frontend.worker.batch_size}`)
      console.log(`Partitions to Search: ${this.config.frontend.worker.search.num_partitions_to_search}`)
      console.log(`Debug Mode: ${this.config.frontend.debug.enabled}`)
      console.groupEnd()
    }
  }
}

// Global configuration manager instance
export const configManager = new ConfigManager()

// Convenience exports
export const config = configManager.getConfig()
export const frontendConfig = configManager.getFrontendConfig()
export const workerConfig = configManager.getWorkerConfig()
export const modelConfig = configManager.getModelConfig()
export const onnxConfig = configManager.getONNXConfig()
export const debugConfig = configManager.getDebugConfig()

// Initialize logging
configManager.logConfigSummary()