#!/usr/bin/env python3
"""
Configuration management for Cytoverse using Hydra/OmegaConf.

This module provides centralized configuration management for all Python scripts
and machine learning pipelines in the Cytoverse project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """Configuration manager using Hydra/OmegaConf for Cytoverse."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory. Defaults to project config dir.
        """
        if config_dir is None:
            # Default to project config directory
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "config"
            
        self.config_dir = Path(config_dir).resolve()
        self._hydra_initialized = False
        
    def _ensure_hydra_initialized(self) -> None:
        """Ensure Hydra is initialized with the config directory."""
        if not self._hydra_initialized:
            initialize_config_dir(
                config_dir=str(self.config_dir),
                version_base=None
            )
            self._hydra_initialized = True
    
    def get_config(
        self, 
        environment: Optional[str] = None,
        overrides: Optional[list] = None
    ) -> DictConfig:
        """Get configuration for specified environment.
        
        Args:
            environment: Environment name (dev, staging, prod). 
                        Defaults to CYTOVERSE_ENV or 'dev'.
            overrides: List of config overrides in Hydra format.
                      
        Returns:
            Complete configuration object.
        """
        self._ensure_hydra_initialized()
        
        if environment is None:
            environment = os.getenv("CYTOVERSE_ENV", "dev")
            
        config_overrides = [f"environment={environment}"]
        if overrides:
            config_overrides.extend(overrides)
            
        return compose(config_name="config", overrides=config_overrides)
    
    def get_python_config(
        self, 
        environment: Optional[str] = None,
        overrides: Optional[list] = None
    ) -> DictConfig:
        """Get Python-specific configuration.
        
        Args:
            environment: Environment name.
            overrides: Config overrides.
            
        Returns:
            Python configuration section.
        """
        config = self.get_config(environment, overrides)
        return config.python
    
    def get_ivfpq_config(
        self, 
        environment: Optional[str] = None,
        overrides: Optional[list] = None
    ) -> DictConfig:
        """Get IVFPQ-specific configuration.
        
        Args:
            environment: Environment name.
            overrides: Config overrides.
            
        Returns:
            IVFPQ configuration section.
        """
        python_config = self.get_python_config(environment, overrides)
        return python_config.ivfpq
    
    def resolve_model_paths(self, config: DictConfig, model_id: str) -> DictConfig:
        """Resolve model-specific paths in configuration.
        
        Args:
            config: Configuration object.
            model_id: Model identifier for path resolution.
            
        Returns:
            Configuration with resolved paths.
        """
        # Create a copy to avoid modifying original
        resolved_config = OmegaConf.copy(config)
        
        # Set model_id for path interpolation
        OmegaConf.set_struct(resolved_config, False)
        resolved_config.model_id = model_id
        OmegaConf.set_struct(resolved_config, True)
        
        # Resolve interpolations
        return OmegaConf.create(OmegaConf.to_yaml(resolved_config))
    
    def validate_config(self, config: DictConfig) -> bool:
        """Validate configuration for common issues.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            True if configuration is valid.
            
        Raises:
            ValueError: If configuration has validation errors.
        """
        # Basic validation checks
        if "python" not in config:
            raise ValueError("Missing 'python' section in configuration")
            
        if "ivfpq" not in config.python:
            raise ValueError("Missing 'ivfpq' section in python configuration")
            
        # Validate IVFPQ parameters
        ivfpq = config.python.ivfpq
        
        if ivfpq.model.n_partitions <= 0:
            raise ValueError("n_partitions must be positive")
            
        if ivfpq.model.pq_m <= 0:
            raise ValueError("pq_m must be positive")
            
        if ivfpq.model.pq_k <= 0:
            raise ValueError("pq_k must be positive")
            
        if ivfpq.search.n_probe > ivfpq.model.n_partitions:
            raise ValueError("n_probe cannot exceed n_partitions")
            
        return True
    
    def print_config_summary(self, config: DictConfig) -> None:
        """Print a summary of the current configuration."""
        print("=== Cytoverse Configuration Summary ===")
        print(f"Environment: {config.app.environment}")
        print(f"App Version: {config.app.version}")
        
        if "python" in config and "ivfpq" in config.python:
            ivfpq = config.python.ivfpq
            print(f"\nIVFPQ Configuration:")
            print(f"  Partitions: {ivfpq.model.n_partitions}")
            print(f"  PQ M: {ivfpq.model.pq_m}")
            print(f"  PQ K: {ivfpq.model.pq_k}")
            print(f"  N Probe: {ivfpq.search.n_probe}")
            print(f"  Training Vectors: {ivfpq.training.num_training_vectors:,}")
            
        print("=" * 40)


# Global configuration manager instance
config_manager = ConfigManager()


def get_config(environment: Optional[str] = None, overrides: Optional[list] = None) -> DictConfig:
    """Convenience function to get configuration."""
    return config_manager.get_config(environment, overrides)


def get_ivfpq_config(environment: Optional[str] = None, overrides: Optional[list] = None) -> DictConfig:
    """Convenience function to get IVFPQ configuration."""
    return config_manager.get_ivfpq_config(environment, overrides)