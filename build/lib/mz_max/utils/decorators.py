"""
Decorators for MZ Max

This module provides useful decorators for the MZ Max package.
"""

import time
import functools
import warnings
from typing import Any, Callable, Optional, Union
from .logging import get_logger

logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def deprecated(reason: str = "This function is deprecated"):
    """
    Decorator to mark functions as deprecated.
    
    Args:
        reason: Deprecation reason
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(**type_checks):
    """
    Decorator to validate function input types.
    
    Args:
        **type_checks: Keyword arguments mapping parameter names to expected types
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' expected {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier for delay
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {str(e)}")
                        raise e
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {str(e)}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


def cache_result(ttl: Optional[float] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live for cache in seconds (None for no expiration)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if result is cached and not expired
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or current_time - timestamp < ttl:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            
            return result
        
        # Add cache clearing method
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper
    return decorator


def log_calls(level: str = 'INFO'):
    """
    Decorator to log function calls.
    
    Args:
        level: Logging level
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_func = getattr(logger, level.lower())
            log_func(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                log_func(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {str(e)}")
                raise
        return wrapper
    return decorator


def singleton(cls):
    """
    Decorator to create singleton classes.
    
    Args:
        cls: Class to make singleton
        
    Returns:
        Singleton class
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def property_cached(func: Callable) -> property:
    """
    Decorator to create cached properties.
    
    Args:
        func: Property getter function
        
    Returns:
        Cached property
    """
    attr_name = f'_{func.__name__}_cached'
    
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return property(wrapper)


def ensure_fitted(func: Callable) -> Callable:
    """
    Decorator to ensure model is fitted before calling method.
    
    Args:
        func: Method to wrap
        
    Returns:
        Wrapped method
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_fitted') or not self._fitted:
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet. "
                           f"Call 'fit' with appropriate arguments before using this method.")
        return func(self, *args, **kwargs)
    return wrapper


def requires_dependencies(*dependencies):
    """
    Decorator to check for required dependencies.
    
    Args:
        *dependencies: Required package names
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing_deps = []
            
            for dep in dependencies:
                try:
                    __import__(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            if missing_deps:
                raise ImportError(
                    f"Missing required dependencies for {func.__name__}: {', '.join(missing_deps)}. "
                    f"Install with: pip install {' '.join(missing_deps)}"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def memory_limit(max_memory_mb: int):
    """
    Decorator to limit memory usage of function.
    
    Args:
        max_memory_mb: Maximum memory usage in MB
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                result = func(*args, **kwargs)
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = final_memory - initial_memory
                
                if memory_used > max_memory_mb:
                    logger.warning(
                        f"{func.__name__} used {memory_used:.1f} MB, "
                        f"exceeding limit of {max_memory_mb} MB"
                    )
                
                return result
                
            except ImportError:
                logger.warning("psutil not available for memory monitoring")
                return func(*args, **kwargs)
        return wrapper
    return decorator


def parallel_execution(n_jobs: int = -1):
    """
    Decorator to enable parallel execution for functions that support it.
    
    Args:
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Add n_jobs parameter if not present
            if 'n_jobs' not in kwargs:
                kwargs['n_jobs'] = n_jobs
            return func(*args, **kwargs)
        return wrapper
    return decorator
