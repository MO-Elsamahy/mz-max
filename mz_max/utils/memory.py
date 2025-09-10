"""
Memory management utilities for MZ Max

This module provides memory monitoring and optimization utilities.
"""

import gc
import sys
import psutil
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from .logging import get_logger

logger = get_logger(__name__)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage statistics
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return {}


def get_object_size(obj: Any) -> int:
    """
    Get size of Python object in bytes.
    
    Args:
        obj: Object to measure
        
    Returns:
        Size in bytes
    """
    return sys.getsizeof(obj)


def get_dataframe_memory_usage(df: pd.DataFrame, deep: bool = True) -> Dict[str, Any]:
    """
    Get detailed memory usage of DataFrame.
    
    Args:
        df: DataFrame to analyze
        deep: Whether to get deep memory usage
        
    Returns:
        Memory usage information
    """
    memory_usage = df.memory_usage(deep=deep)
    
    return {
        'total_mb': memory_usage.sum() / 1024 / 1024,
        'index_mb': memory_usage.iloc[0] / 1024 / 1024,
        'columns_mb': {col: memory_usage.iloc[i+1] / 1024 / 1024 
                      for i, col in enumerate(df.columns)},
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict()
    }


def optimize_dataframe_memory(df: pd.DataFrame, 
                            aggressive: bool = False) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        aggressive: Whether to use aggressive optimization
        
    Returns:
        Optimized DataFrame
    """
    original_memory = df.memory_usage(deep=True).sum()
    optimized_df = df.copy()
    
    # Optimize numeric columns
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type != 'object':
            # Integer optimization
            if str(col_type).startswith('int'):
                min_val = optimized_df[col].min()
                max_val = optimized_df[col].max()
                
                if min_val >= -128 and max_val <= 127:
                    optimized_df[col] = optimized_df[col].astype('int8')
                elif min_val >= -32768 and max_val <= 32767:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')
            
            # Float optimization
            elif str(col_type).startswith('float'):
                if aggressive:
                    # Try float32 if precision allows
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                else:
                    # Conservative float64 to float32
                    if col_type == 'float64':
                        optimized_df[col] = optimized_df[col].astype('float32')
        
        # String optimization
        elif col_type == 'object':
            if aggressive:
                # Convert to category if beneficial
                unique_count = optimized_df[col].nunique()
                total_count = len(optimized_df[col])
                
                if unique_count / total_count < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
    
    optimized_memory = optimized_df.memory_usage(deep=True).sum()
    reduction_percent = (original_memory - optimized_memory) / original_memory * 100
    
    logger.info(f"Memory optimization: {original_memory/1024/1024:.2f} MB -> "
               f"{optimized_memory/1024/1024:.2f} MB "
               f"({reduction_percent:.1f}% reduction)")
    
    return optimized_df


def clear_memory():
    """Force garbage collection to free memory."""
    collected = gc.collect()
    logger.info(f"Garbage collection freed {collected} objects")


def get_largest_objects(n: int = 10) -> List[Dict[str, Any]]:
    """
    Get information about the largest objects in memory.
    
    Args:
        n: Number of largest objects to return
        
    Returns:
        List of object information
    """
    import gc
    
    objects = gc.get_objects()
    object_info = []
    
    for obj in objects:
        try:
            size = sys.getsizeof(obj)
            obj_type = type(obj).__name__
            
            # Get additional info for common types
            additional_info = {}
            if isinstance(obj, (list, tuple)):
                additional_info['length'] = len(obj)
            elif isinstance(obj, dict):
                additional_info['keys'] = len(obj)
            elif isinstance(obj, np.ndarray):
                additional_info['shape'] = obj.shape
                additional_info['dtype'] = str(obj.dtype)
            elif isinstance(obj, pd.DataFrame):
                additional_info['shape'] = obj.shape
                additional_info['columns'] = len(obj.columns)
            
            object_info.append({
                'type': obj_type,
                'size_bytes': size,
                'size_mb': size / 1024 / 1024,
                **additional_info
            })
        
        except (TypeError, AttributeError):
            # Skip objects that can't be measured
            continue
    
    # Sort by size and return top n
    object_info.sort(key=lambda x: x['size_bytes'], reverse=True)
    return object_info[:n]


def memory_profiler(func):
    """
    Decorator to profile memory usage of function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        # Get initial memory
        initial_memory = get_memory_usage()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_memory = get_memory_usage()
        
        if initial_memory and final_memory:
            memory_diff = final_memory['rss_mb'] - initial_memory['rss_mb']
            logger.info(f"{func.__name__} memory usage: {memory_diff:+.2f} MB")
        
        return result
    
    return wrapper


class MemoryMonitor:
    """
    Context manager for monitoring memory usage.
    """
    
    def __init__(self, name: str = "operation"):
        """
        Initialize memory monitor.
        
        Args:
            name: Name of operation being monitored
        """
        self.name = name
        self.initial_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        """Start monitoring."""
        self.initial_memory = get_memory_usage()
        self.peak_memory = self.initial_memory.copy() if self.initial_memory else {}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and report."""
        final_memory = get_memory_usage()
        
        if self.initial_memory and final_memory:
            memory_diff = final_memory['rss_mb'] - self.initial_memory['rss_mb']
            peak_diff = self.peak_memory['rss_mb'] - self.initial_memory['rss_mb']
            
            logger.info(f"{self.name} - Memory usage: {memory_diff:+.2f} MB, "
                       f"Peak: {peak_diff:+.2f} MB")
    
    def update_peak(self):
        """Update peak memory usage."""
        current_memory = get_memory_usage()
        if current_memory and self.peak_memory:
            if current_memory['rss_mb'] > self.peak_memory['rss_mb']:
                self.peak_memory = current_memory


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """
    Split DataFrame into chunks for memory-efficient processing.
    
    Args:
        df: DataFrame to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of DataFrame chunks
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i + chunk_size])
    
    logger.info(f"DataFrame split into {len(chunks)} chunks of size {chunk_size}")
    return chunks


def estimate_memory_requirement(data_shape: tuple, dtype: str = 'float64') -> float:
    """
    Estimate memory requirement for data of given shape and dtype.
    
    Args:
        data_shape: Shape of data
        dtype: Data type
        
    Returns:
        Estimated memory in MB
    """
    dtype_sizes = {
        'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
        'uint8': 1, 'uint16': 2, 'uint32': 4, 'uint64': 8,
        'float16': 2, 'float32': 4, 'float64': 8,
        'bool': 1, 'object': 8  # Approximate for object
    }
    
    element_size = dtype_sizes.get(dtype, 8)  # Default to 8 bytes
    total_elements = np.prod(data_shape)
    total_bytes = total_elements * element_size
    
    return total_bytes / 1024 / 1024  # Convert to MB


def check_memory_availability(required_mb: float) -> bool:
    """
    Check if required memory is available.
    
    Args:
        required_mb: Required memory in MB
        
    Returns:
        True if memory is available
    """
    memory_info = get_memory_usage()
    
    if not memory_info:
        logger.warning("Cannot check memory availability - psutil not available")
        return True  # Assume available if can't check
    
    available_mb = memory_info['available_mb']
    
    if required_mb > available_mb:
        logger.warning(f"Insufficient memory: required {required_mb:.1f} MB, "
                      f"available {available_mb:.1f} MB")
        return False
    
    return True


def suggest_chunk_size(data_shape: tuple, dtype: str = 'float64', 
                      max_memory_mb: float = 1024) -> int:
    """
    Suggest optimal chunk size for processing large data.
    
    Args:
        data_shape: Shape of data
        dtype: Data type
        max_memory_mb: Maximum memory to use in MB
        
    Returns:
        Suggested chunk size
    """
    if len(data_shape) < 2:
        return data_shape[0]
    
    # Calculate memory per row
    row_shape = (1,) + data_shape[1:]
    memory_per_row = estimate_memory_requirement(row_shape, dtype)
    
    # Calculate chunk size that fits in memory limit
    chunk_size = int(max_memory_mb / memory_per_row)
    
    # Ensure chunk size is reasonable
    chunk_size = max(1, min(chunk_size, data_shape[0]))
    
    logger.info(f"Suggested chunk size: {chunk_size} rows "
               f"(~{chunk_size * memory_per_row:.1f} MB per chunk)")
    
    return chunk_size


class MemoryEfficientProcessor:
    """
    Base class for memory-efficient data processing.
    """
    
    def __init__(self, max_memory_mb: float = 1024):
        """
        Initialize processor.
        
        Args:
            max_memory_mb: Maximum memory to use
        """
        self.max_memory_mb = max_memory_mb
    
    def process_in_chunks(self, data: pd.DataFrame, 
                         process_func, **kwargs) -> pd.DataFrame:
        """
        Process DataFrame in memory-efficient chunks.
        
        Args:
            data: Data to process
            process_func: Function to apply to each chunk
            **kwargs: Additional arguments for process_func
            
        Returns:
            Processed DataFrame
        """
        # Estimate memory requirement
        memory_per_row = get_dataframe_memory_usage(data.head(100))['total_mb'] / 100
        chunk_size = int(self.max_memory_mb / memory_per_row)
        chunk_size = max(100, min(chunk_size, len(data)))
        
        logger.info(f"Processing {len(data)} rows in chunks of {chunk_size}")
        
        results = []
        for chunk in chunk_dataframe(data, chunk_size):
            with MemoryMonitor(f"chunk_{len(results)+1}"):
                processed_chunk = process_func(chunk, **kwargs)
                results.append(processed_chunk)
                
                # Force garbage collection after each chunk
                clear_memory()
        
        # Combine results
        return pd.concat(results, ignore_index=True)
