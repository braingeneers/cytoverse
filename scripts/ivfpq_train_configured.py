#!/usr/bin/env python3

"""
Training script for IVFPQ (Inverted File Index Product Quantization) models.

This script provides complete IVFPQ training functionality using Hydra/OmegaConf
for configuration management and production-level settings.

Features:
- Hydra/OmegaConf configuration management
- Environment-specific configurations (dev/staging/prod)
- Train complete IVFPQ models with residual vectors
- Export browser-compatible artifacts under public/models/<model_id>/
- Integration with TypeScript browser implementation
- Performance testing with trained models
"""

import random
import typer
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from ivfpq.ivfpq import IVFPQ
from cytoverse.config import ConfigManager, get_ivfpq_config

app = typer.Typer(
    help="Train IVFPQ models with production-level configuration",
    add_completion=False,
)


def setup_logging(config: DictConfig) -> None:
    """Setup logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format
    )


def set_seed(seed: int, config: DictConfig) -> None:
    """Set random seed for reproducibility."""
    logger = logging.getLogger(__name__)
    logger.info(f"Setting global random seed: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.info("CUDA available: seeded all CUDA devices")
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS backend available: seeded via torch.manual_seed")
        
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Enabled deterministic algorithms for reproducibility")
    except Exception as e:
        logger.warning(f"Could not enable deterministic algorithms: {e}")


def detect_device(device_preference: str) -> str:
    """Detect and return the best available device."""
    logger = logging.getLogger(__name__)
    
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Auto-detected MPS (Apple Silicon) device")
        else:
            device = "cpu"
            logger.info("Auto-detected CPU device")
    else:
        device = device_preference
        logger.info(f"Using configured device: {device}")
        
    return device


def load_embeddings(embeddings_path: str, num_vectors: int, config: DictConfig) -> torch.Tensor:
    """Load embeddings from file with sampling."""
    logger = logging.getLogger(__name__)
    
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    if len(embeddings) > num_vectors:
        logger.info(f"Sampling {num_vectors:,} vectors from {len(embeddings):,}")
        indices = np.random.choice(len(embeddings), num_vectors, replace=False)
        embeddings = embeddings[indices]
    
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    return torch.from_numpy(embeddings).float()


def train_ivfpq_model(
    embeddings: torch.Tensor,
    config: DictConfig,
    device: str,
    model_id: str
) -> IVFPQ:
    """Train IVFPQ model with configuration."""
    logger = logging.getLogger(__name__)
    
    # Get model parameters from config
    model_config = config.model
    training_config = config.training
    
    d_sub = model_config.d_sub
    if d_sub is None:
        d_sub = embeddings.shape[1] // model_config.pq_m
        logger.info(f"Auto-calculated d_sub: {d_sub}")
    
    logger.info("Initializing IVFPQ model with configuration:")
    logger.info(f"  Dimensions: {embeddings.shape[1]}")
    logger.info(f"  Partitions: {model_config.n_partitions}")
    logger.info(f"  PQ M: {model_config.pq_m}")
    logger.info(f"  PQ K: {model_config.pq_k}")
    logger.info(f"  d_sub: {d_sub}")
    
    # Initialize IVFPQ
    ivfpq = IVFPQ(
        d=embeddings.shape[1],
        n_partitions=model_config.n_partitions,
        pq_m=model_config.pq_m,
        pq_k=model_config.pq_k,
        d_sub=d_sub,
        device=device
    )
    
    logger.info("Training IVFPQ model...")
    ivfpq.train(
        train_embeddings=embeddings,
        max_iters=training_config.max_iters,
        tolerance=training_config.tolerance,
        batch_size=training_config.batch_size
    )
    
    logger.info("IVFPQ training completed")
    return ivfpq


def export_browser_assets(
    ivfpq: IVFPQ, 
    output_dir: str, 
    model_id: str,
    config: DictConfig
) -> None:
    """Export browser-compatible assets."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting browser assets to {output_path}")
    
    if config.export.format == "arrow":
        ivfpq.export_browser_assets(str(output_path))
    else:
        raise ValueError(f"Unsupported export format: {config.export.format}")
    
    if config.export.validate_export:
        logger.info("Validating export...")
        # Quick validation by attempting to load
        test_ivfpq = IVFPQ.load_from_browser_assets(str(output_path))
        logger.info("Export validation successful")
    
    logger.info("Browser asset export completed")


@app.command()
def train(
    model_id: str = typer.Argument(..., help="Model ID for training (e.g., 'scimilarity')"),
    environment: str = typer.Option("dev", help="Environment: dev, staging, prod"),
    embeddings_path: Optional[str] = typer.Option(None, help="Path to embeddings file (overrides config)"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory (overrides config)"),
    num_vectors: Optional[int] = typer.Option(None, help="Number of training vectors (overrides config)"),
    config_overrides: Optional[list[str]] = typer.Option(None, help="Configuration overrides (key=value)"),
) -> None:
    """Train IVFPQ model with configuration management."""
    
    # Load configuration
    config_manager = ConfigManager()
    overrides = config_overrides or []
    config = config_manager.get_ivfpq_config(environment, overrides)
    
    # Resolve model-specific paths
    config = config_manager.resolve_model_paths(config, model_id)
    
    # Validate configuration
    config_manager.validate_config({'python': {'ivfpq': config}})
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Print configuration summary
    config_manager.print_config_summary({'app': {'environment': environment, 'version': '0.1.0'}, 'python': {'ivfpq': config}})
    
    # Override paths if provided
    embeddings_file = embeddings_path or config.data.embeddings_path
    output_directory = output_dir or config.data.output_dir
    training_vectors = num_vectors or config.training.num_training_vectors
    
    # Set seed for reproducibility
    set_seed(config.seed, config)
    
    # Detect device
    device = detect_device(config.device)
    
    try:
        # Load embeddings
        embeddings = load_embeddings(embeddings_file, training_vectors, config)
        
        # Train model
        ivfpq = train_ivfpq_model(embeddings, config, device, model_id)
        
        # Export browser assets
        if config.export.browser_assets:
            export_browser_assets(ivfpq, output_directory, model_id, config)
        
        logger.info("IVFPQ training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def config_info(
    environment: str = typer.Option("dev", help="Environment to show config for"),
) -> None:
    """Show configuration for specified environment."""
    config_manager = ConfigManager()
    config = config_manager.get_config(environment)
    config_manager.print_config_summary(config)


@app.command()
def validate_config(
    environment: str = typer.Option("dev", help="Environment to validate"),
) -> None:
    """Validate configuration for specified environment."""
    config_manager = ConfigManager()
    config = config_manager.get_config(environment)
    
    try:
        config_manager.validate_config(config)
        print(f"✅ Configuration for '{environment}' is valid")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()