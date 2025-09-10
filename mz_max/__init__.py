"""
MZ Max - The Most Powerful ML/DL Python Package

A comprehensive machine learning and deep learning framework that provides
state-of-the-art algorithms, tools, and utilities for AI development.
"""

__version__ = "1.0.0"
__author__ = "MZ Development Team"
__email__ = "dev@mzmax.ai"
__description__ = "The most powerful Python package for Machine Learning and Deep Learning"

# Core imports
# Import core modules with error handling
try:
    from . import core
except ImportError as e:
    print(f"Warning: Could not import core: {e}")

try:
    from . import utils
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")

try:
    from . import data
except ImportError as e:
    print(f"Warning: Could not import data: {e}")

try:
    from . import models
except ImportError as e:
    print(f"Warning: Could not import models: {e}")

try:
    from . import automl
except ImportError as e:
    print(f"Warning: Could not import automl: {e}")

try:
    from . import ui
except ImportError as e:
    print(f"Warning: Could not import ui: {e}")

try:
    from . import enterprise
except ImportError as e:
    print(f"Warning: Could not import enterprise: {e}")

try:
    from . import advanced
except ImportError as e:
    print(f"Warning: Could not import advanced: {e}")

try:
    from . import production
except ImportError as e:
    print(f"Warning: Could not import production: {e}")

try:
    from . import integrations
except ImportError as e:
    print(f"Warning: Could not import integrations: {e}")

# Convenience imports with error handling
try:
    from .core.base import BaseEstimator, BaseTransformer
except ImportError:
    BaseEstimator = BaseTransformer = None

try:
    from .utils.logging import get_logger
except ImportError:
    get_logger = None

try:
    from .data.loaders import load_dataset
except ImportError:
    load_dataset = None

try:
    from .models.registry import list_models, load_model
except ImportError:
    list_models = load_model = None

# Package metadata
__all__ = [
    # Core Modules
    'core',
    'utils', 
    'data',
    'preprocessing',
    'models',
    'training',
    'automl',
    'deep_learning',
    'optimization',
    'interpretability',
    'visualization',
    'deployment',
    'ui',
    
    # Enterprise & Advanced Modules
    'enterprise',
    'advanced',
    'production',
    'integrations',
    
    # Classes
    'BaseEstimator',
    'BaseTransformer',
    
    # Functions
    'get_logger',
    'load_dataset',
    'list_models',
    'load_model',
]

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_version():
    """Get the package version."""
    return __version__

def get_info():
    """Get package information."""
    return {
        'name': 'mz-max',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': __description__,
        'python_requires': '>=3.8',
    }
