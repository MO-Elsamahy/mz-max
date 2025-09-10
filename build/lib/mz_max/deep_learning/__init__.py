"""
Deep Learning module for MZ Max

This module provides comprehensive deep learning capabilities including
PyTorch and TensorFlow integrations, pre-trained models, and advanced
training utilities.
"""

from .pytorch import *
from .tensorflow import *
from .models import *
from .training import *
from .layers import *
from .optimizers import *
from .losses import *

__all__ = [
    # PyTorch components
    'PyTorchModel',
    'PyTorchTrainer',
    'PyTorchDataLoader',
    
    # TensorFlow components
    'TensorFlowModel', 
    'TensorFlowTrainer',
    'TensorFlowDataLoader',
    
    # Models
    'CNN',
    'ResNet',
    'Transformer',
    'LSTM',
    'GAN',
    'VAE',
    
    # Training utilities
    'Trainer',
    'TrainingConfig',
    'EarlyStopping',
    'ModelCheckpoint',
    
    # Custom layers
    'Attention',
    'MultiHeadAttention',
    'PositionalEncoding',
    
    # Optimizers
    'AdamW',
    'Lion',
    'AdaBelief',
    
    # Loss functions
    'FocalLoss',
    'LabelSmoothingLoss',
    'ContrastiveLoss',
]
