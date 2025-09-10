"""
Logging utilities for MZ Max

This module provides logging configuration and utilities for the MZ Max package.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def get_logger(name: str = 'mz_max') -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_logging(level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Set up logging configuration for MZ Max.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    
    # Add file handler if specified
    handlers = [console_handler]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure MZ Max logger
    mz_logger = logging.getLogger('mz_max')
    mz_logger.handlers.clear()
    for handler in handlers:
        mz_logger.addHandler(handler)
    mz_logger.setLevel(numeric_level)
    mz_logger.propagate = False


# Default logger for the package
logger = get_logger('mz_max')
