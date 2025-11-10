"""
Configuration management for metaClassifier.

This module contains configuration loading and management utilities.
"""

from typing import Any, Dict, Optional, Union
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from .logger import get_logger


@dataclass
class Config:
    """Configuration class for metaClassifier."""
    
    # Model configuration
    model_name: str = "Lasso"
    model_params: Dict[str, Any] = None
    
    # Cross-validation configuration
    cv_strategy: str = "repeated_kfold"
    outer_folds: int = 5
    inner_folds: int = 3
    n_repeats: int = 1
    random_state: int = 42
    n_jobs: int = 1
    
    # Feature selection configuration
    feature_selection_method: str = "adaptive"
    max_features: int = 50
    
    # Adaptive filtering configuration
    adaptive_filtering_enabled: bool = True
    min_q: float = 0.05
    max_q: float = 0.2
    r_mid: float = 5.0
    steepness: float = 0.5
    
    # Data configuration
    data_paths: Dict[str, str] = None
    output_dir: str = "./results"
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.model_params is None:
            self.model_params = {}
        if self.data_paths is None:
            self.data_paths = {}


class ConfigManager:
    """Configuration manager for metaClassifier."""
    
    def __init__(self):
        self.logger = get_logger("ConfigManager")
        self.config = Config()
        
    def load_from_file(self, config_path: Union[str, Path]) -> 'ConfigManager':
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Self for method chaining
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.logger.info(f"Loading configuration from {config_path}")
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            self._load_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            self._load_json(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return self
        
    def _load_yaml(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        self._update_config(config_data)
        
    def _load_json(self, config_path: Path) -> None:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        self._update_config(config_data)
        
    def _update_config(self, config_data: Dict[str, Any]) -> None:
        """Update configuration with loaded data."""
        for key, value in config_data.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
        
        self.logger.info("Configuration loaded successfully")
        
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving configuration to {config_path}")
        
        config_data = asdict(self.config)
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        self.logger.info("Configuration saved successfully")
        
    def get_config(self) -> Config:
        """Get the current configuration."""
        return self.config
        
    def update_config(self, **kwargs) -> 'ConfigManager':
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            Self for method chaining
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
        
        return self
