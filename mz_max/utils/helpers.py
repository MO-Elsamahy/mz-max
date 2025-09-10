"""
Helper utilities for MZ Max

This module provides common utility functions used across the package.
"""

import os
import json
import pickle
import hashlib
import tempfile
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object as pickle file.
    
    Args:
        obj: Object to save
        filepath: Output file path
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_file_hash(filepath: Union[str, Path]) -> str:
    """
    Get SHA-256 hash of file.
    
    Args:
        filepath: File path
        
    Returns:
        File hash
    """
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_data_info(data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """
    Get information about data.
    
    Args:
        data: Input data
        
    Returns:
        Data information
    """
    info = {}
    
    if isinstance(data, np.ndarray):
        info.update({
            'type': 'numpy.ndarray',
            'shape': data.shape,
            'dtype': str(data.dtype),
            'memory_usage': data.nbytes,
            'has_nan': bool(np.isnan(data).any()) if data.dtype.kind in 'fc' else False
        })
    
    elif isinstance(data, pd.DataFrame):
        info.update({
            'type': 'pandas.DataFrame',
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'has_nan': data.isnull().any().any(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns)
        })
    
    return info


def validate_input(data: Any, expected_type: type = None, 
                  expected_shape: Tuple = None) -> bool:
    """
    Validate input data.
    
    Args:
        data: Input data
        expected_type: Expected data type
        expected_shape: Expected shape
        
    Returns:
        True if valid
    """
    if expected_type and not isinstance(data, expected_type):
        return False
    
    if expected_shape and hasattr(data, 'shape'):
        if data.shape != expected_shape:
            return False
    
    return True


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_temp_file(suffix: str = '', prefix: str = 'mzmax_') -> str:
    """
    Create temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        
    Returns:
        Temporary file path
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)  # Close file descriptor
    return path


def create_temp_dir(prefix: str = 'mzmax_') -> str:
    """
    Create temporary directory.
    
    Args:
        prefix: Directory prefix
        
    Returns:
        Temporary directory path
    """
    return tempfile.mkdtemp(prefix=prefix)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safe division with default value for division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value if b is zero
        
    Returns:
        Division result or default
    """
    return a / b if b != 0 else default


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').total if os.name != 'nt' else psutil.disk_usage('C:').total
    }


class Timer:
    """Simple timer context manager."""
    
    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timer."""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timer."""
        import time
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on exception.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
