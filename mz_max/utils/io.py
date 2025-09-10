"""
Input/Output utilities for MZ Max

This module provides file I/O operations and data serialization utilities.
"""

import os
import json
import pickle
import joblib
import yaml
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

from .logging import get_logger

logger = get_logger(__name__)


def save_model(model: Any, filepath: Union[str, Path], 
               format: str = 'joblib', metadata: Optional[Dict] = None) -> None:
    """
    Save model to file.
    
    Args:
        model: Model to save
        filepath: Output file path
        format: Serialization format ('joblib', 'pickle')
        metadata: Optional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'joblib':
            joblib.dump(model, filepath)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = filepath.with_suffix('.metadata.json')
            save_json(metadata, metadata_path)
        
        logger.info(f"Model saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise


def load_model(filepath: Union[str, Path], format: str = 'auto') -> Any:
    """
    Load model from file.
    
    Args:
        filepath: Input file path
        format: Serialization format ('auto', 'joblib', 'pickle')
        
    Returns:
        Loaded model
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        if format == 'auto':
            # Auto-detect format based on extension
            if filepath.suffix in ['.joblib', '.pkl']:
                format = 'joblib'
            else:
                format = 'pickle'
        
        if format == 'joblib':
            model = joblib.load(filepath)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Model loaded from {filepath}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def save_json(data: Dict[str, Any], filepath: Union[str, Path], 
              indent: int = 2) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        logger.debug(f"JSON saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {str(e)}")
        raise


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.debug(f"JSON loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON: {str(e)}")
        raise


def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save data as YAML file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        logger.debug(f"YAML saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save YAML: {str(e)}")
        raise


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from YAML file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        logger.debug(f"YAML loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load YAML: {str(e)}")
        raise


def save_csv(data: pd.DataFrame, filepath: Union[str, Path], 
             index: bool = False, **kwargs) -> None:
    """
    Save DataFrame as CSV file.
    
    Args:
        data: DataFrame to save
        filepath: Output file path
        index: Whether to save index
        **kwargs: Additional arguments for pandas.to_csv
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        data.to_csv(filepath, index=index, **kwargs)
        logger.info(f"CSV saved to {filepath} ({data.shape[0]} rows, {data.shape[1]} columns)")
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        raise


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Input file path
        **kwargs: Additional arguments for pandas.read_csv
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        data = pd.read_csv(filepath, **kwargs)
        logger.info(f"CSV loaded from {filepath} ({data.shape[0]} rows, {data.shape[1]} columns)")
        return data
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        raise


def save_parquet(data: pd.DataFrame, filepath: Union[str, Path], 
                 **kwargs) -> None:
    """
    Save DataFrame as Parquet file.
    
    Args:
        data: DataFrame to save
        filepath: Output file path
        **kwargs: Additional arguments for pandas.to_parquet
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        data.to_parquet(filepath, **kwargs)
        logger.info(f"Parquet saved to {filepath} ({data.shape[0]} rows, {data.shape[1]} columns)")
    except Exception as e:
        logger.error(f"Failed to save Parquet: {str(e)}")
        raise


def load_parquet(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from Parquet file.
    
    Args:
        filepath: Input file path
        **kwargs: Additional arguments for pandas.read_parquet
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    
    try:
        data = pd.read_parquet(filepath, **kwargs)
        logger.info(f"Parquet loaded from {filepath} ({data.shape[0]} rows, {data.shape[1]} columns)")
        return data
    except Exception as e:
        logger.error(f"Failed to load Parquet: {str(e)}")
        raise


def save_numpy(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """
    Save NumPy array to file.
    
    Args:
        array: NumPy array to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        np.save(filepath, array)
        logger.info(f"NumPy array saved to {filepath} (shape: {array.shape})")
    except Exception as e:
        logger.error(f"Failed to save NumPy array: {str(e)}")
        raise


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load NumPy array from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded NumPy array
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"NumPy file not found: {filepath}")
    
    try:
        array = np.load(filepath)
        logger.info(f"NumPy array loaded from {filepath} (shape: {array.shape})")
        return array
    except Exception as e:
        logger.error(f"Failed to load NumPy array: {str(e)}")
        raise


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: File path
        
    Returns:
        File size in bytes
    """
    return Path(filepath).stat().st_size


def list_files(directory: Union[str, Path], pattern: str = "*", 
               recursive: bool = False) -> List[Path]:
    """
    List files in directory.
    
    Args:
        directory: Directory path
        pattern: File pattern (glob)
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter to only files (not directories)
    files = [f for f in files if f.is_file()]
    
    return sorted(files)


def ensure_directory(directory: Union[str, Path]) -> Path:
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


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    import shutil
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.copy2(src_path, dst_path)
        logger.info(f"File copied from {src_path} to {dst_path}")
    except Exception as e:
        logger.error(f"Failed to copy file: {str(e)}")
        raise


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Move file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    import shutil
    
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.move(str(src_path), str(dst_path))
        logger.info(f"File moved from {src_path} to {dst_path}")
    except Exception as e:
        logger.error(f"Failed to move file: {str(e)}")
        raise


def delete_file(filepath: Union[str, Path]) -> None:
    """
    Delete file.
    
    Args:
        filepath: File path to delete
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"File not found for deletion: {filepath}")
        return
    
    try:
        filepath.unlink()
        logger.info(f"File deleted: {filepath}")
    except Exception as e:
        logger.error(f"Failed to delete file: {str(e)}")
        raise


class FileHandler:
    """
    Context manager for file operations.
    """
    
    def __init__(self, filepath: Union[str, Path], mode: str = 'r', 
                 encoding: str = 'utf-8'):
        """
        Initialize file handler.
        
        Args:
            filepath: File path
            mode: File mode
            encoding: File encoding
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self.encoding = encoding
        self.file = None
    
    def __enter__(self):
        """Open file."""
        try:
            # Ensure directory exists for write modes
            if 'w' in self.mode or 'a' in self.mode:
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if 'b' in self.mode:
                self.file = open(self.filepath, self.mode)
            else:
                self.file = open(self.filepath, self.mode, encoding=self.encoding)
            
            return self.file
        except Exception as e:
            logger.error(f"Failed to open file {self.filepath}: {str(e)}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file."""
        if self.file:
            self.file.close()
            logger.debug(f"File closed: {self.filepath}")


def auto_detect_format(filepath: Union[str, Path]) -> str:
    """
    Auto-detect file format based on extension.
    
    Args:
        filepath: File path
        
    Returns:
        Detected format
    """
    suffix = Path(filepath).suffix.lower()
    
    format_map = {
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.csv': 'csv',
        '.parquet': 'parquet',
        '.pkl': 'pickle',
        '.pickle': 'pickle',
        '.joblib': 'joblib',
        '.npy': 'numpy',
        '.npz': 'numpy',
        '.txt': 'text',
        '.log': 'text'
    }
    
    return format_map.get(suffix, 'unknown')
