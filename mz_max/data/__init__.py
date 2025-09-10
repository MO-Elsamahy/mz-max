"""
Data module for MZ Max

This module provides data loading and preprocessing utilities.
"""

# Import modules with error handling
try:
    from .loaders import *
except ImportError as e:
    print(f"Warning: Could not import data loaders: {e}")
    
try:
    from .preprocessing import *
except ImportError as e:
    print(f"Warning: Could not import data preprocessing: {e}")

# Basic exports
__all__ = [
    # Data loading
    'load_dataset',
    'load_csv', 
    'load_json',
    
    # Preprocessing
    'clean_data',
    'scale_features',
    'encode_categorical',
    'handle_outliers',
    'DataPreprocessor',
]