"""
Advanced ML module for MZ Max

This module provides cutting-edge machine learning algorithms,
advanced techniques, and research-level implementations.
"""

from .neural_architecture_search import NASOptimizer, AutoArchitecture
from .meta_learning import MetaLearner, FewShotLearner
from .federated_learning import FederatedTrainer, SecureAggregation
from .quantum_ml import QuantumClassifier, QuantumFeatureMap
from .explainable_ai import ExplainableModel, CausalInference
from .continual_learning import ContinualLearner, MemoryReplay
from .multi_modal import MultiModalLearner, CrossModalAlignment
from .graph_neural_networks import GNNClassifier, GraphAutoEncoder
from .reinforcement_learning import RLAgent, PolicyOptimizer
from .time_series_advanced import TimeSeriesForecaster, AnomalyDetector

__all__ = [
    # Neural Architecture Search
    'NASOptimizer',
    'AutoArchitecture',
    
    # Meta Learning
    'MetaLearner',
    'FewShotLearner',
    
    # Federated Learning
    'FederatedTrainer',
    'SecureAggregation',
    
    # Quantum ML
    'QuantumClassifier',
    'QuantumFeatureMap',
    
    # Explainable AI
    'ExplainableModel',
    'CausalInference',
    
    # Continual Learning
    'ContinualLearner',
    'MemoryReplay',
    
    # Multi-Modal Learning
    'MultiModalLearner',
    'CrossModalAlignment',
    
    # Graph Neural Networks
    'GNNClassifier',
    'GraphAutoEncoder',
    
    # Reinforcement Learning
    'RLAgent',
    'PolicyOptimizer',
    
    # Advanced Time Series
    'TimeSeriesForecaster',
    'AnomalyDetector',
]
