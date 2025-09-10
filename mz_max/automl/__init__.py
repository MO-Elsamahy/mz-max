"""
AutoML module for MZ Max

This module provides automated machine learning capabilities including
automated model selection, hyperparameter optimization, and pipeline
construction.
"""

from .auto_classifier import AutoClassifier
from .auto_regressor import AutoRegressor
from .auto_clusterer import AutoClusterer
from .pipeline import AutoPipeline
from .optimization import HyperparameterOptimizer
from .ensemble import AutoEnsemble
from .feature_selection import AutoFeatureSelector

__all__ = [
    'AutoClassifier',
    'AutoRegressor', 
    'AutoClusterer',
    'AutoPipeline',
    'HyperparameterOptimizer',
    'AutoEnsemble',
    'AutoFeatureSelector',
]
