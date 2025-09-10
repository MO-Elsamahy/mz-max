"""
Custom exceptions for MZ Max

This module defines custom exception classes used throughout the MZ Max package.
"""


class MZMaxError(Exception):
    """Base exception class for MZ Max."""
    pass


class ModelNotFoundError(MZMaxError):
    """Raised when a requested model is not found."""
    pass


class DataError(MZMaxError):
    """Raised when there are issues with data processing or validation."""
    pass


class TrainingError(MZMaxError):
    """Raised when there are issues during model training."""
    pass


class PredictionError(MZMaxError):
    """Raised when there are issues during prediction."""
    pass


class ConfigError(MZMaxError):
    """Raised when there are configuration issues."""
    pass


class DependencyError(MZMaxError):
    """Raised when required dependencies are missing."""
    pass


class VersionError(MZMaxError):
    """Raised when there are version compatibility issues."""
    pass


class SerializationError(MZMaxError):
    """Raised when there are issues with model serialization/deserialization."""
    pass


class OptimizationError(MZMaxError):
    """Raised when there are issues with hyperparameter optimization."""
    pass
