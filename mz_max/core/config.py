"""
Configuration management for MZ Max

This module provides configuration management functionality for the MZ Max package.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from .exceptions import ConfigError


class Config:
    """
    Configuration manager for MZ Max.
    
    This class handles loading, saving, and managing configuration settings
    for the MZ Max package.
    """
    
    _instance = None
    _config = {}
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize_default_config()
        return cls._instance
    
    def _initialize_default_config(self):
        """Initialize default configuration settings."""
        self._config = {
            'general': {
                'random_seed': 42,
                'n_jobs': -1,
                'verbose': 1,
                'cache_dir': '~/.mz_max_cache',
                'log_level': 'INFO'
            },
            'training': {
                'early_stopping_patience': 10,
                'validation_split': 0.2,
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001
            },
            'optimization': {
                'n_trials': 100,
                'timeout': 3600,
                'n_jobs': 1,
                'sampler': 'TPE'
            },
            'visualization': {
                'figure_size': (10, 6),
                'style': 'seaborn',
                'color_palette': 'viridis',
                'save_format': 'png',
                'dpi': 300
            },
            'deployment': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 1,
                'timeout': 60
            },
            'data': {
                'max_memory_usage': '8GB',
                'chunk_size': 10000,
                'preprocessing_n_jobs': -1
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports nested keys with dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with a dictionary.
        
        Args:
            config_dict: Dictionary of configuration updates
        """
        self._deep_update(self._config, config_dict)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to configuration file (JSON or YAML)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ConfigError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                if filepath.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif filepath.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ConfigError(f"Unsupported configuration file format: {filepath.suffix}")
            
            self.update(config_data)
        except Exception as e:
            raise ConfigError(f"Error loading configuration file: {e}")
    
    def save_to_file(self, filepath: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save configuration to a file.
        
        Args:
            filepath: Path to save configuration file
            format: File format ('yaml' or 'json')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                if format.lower() in ['yml', 'yaml']:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ConfigError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigError(f"Error saving configuration file: {e}")
    
    def reset(self) -> None:
        """Reset configuration to default values."""
        self._initialize_default_config()
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration settings."""
        return self._config.copy()
    
    def load_environment_variables(self, prefix: str = 'MZMAX_') -> None:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # Try to parse as JSON, otherwise treat as string
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    # Try to parse as boolean or number
                    if value.lower() in ['true', 'false']:
                        parsed_value = value.lower() == 'true'
                    elif value.isdigit():
                        parsed_value = int(value)
                    elif value.replace('.', '').isdigit():
                        parsed_value = float(value)
                    else:
                        parsed_value = value
                
                self.set(config_key, parsed_value)
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
        """
        # Add configuration validation logic here
        required_keys = [
            'general.random_seed',
            'training.batch_size',
            'training.epochs'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                raise ConfigError(f"Required configuration key missing: {key}")
        
        # Validate specific values
        if self.get('training.batch_size') <= 0:
            raise ConfigError("Batch size must be positive")
        
        if self.get('training.epochs') <= 0:
            raise ConfigError("Number of epochs must be positive")
        
        return True


# Global configuration instance
config = Config()
