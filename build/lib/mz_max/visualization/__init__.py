"""
Visualization module for MZ Max

This module provides comprehensive visualization capabilities for
machine learning and deep learning models, data exploration,
and model interpretability.
"""

from .plotting import *
from .model_viz import *
from .data_viz import *
from .interactive import *
from .interpretability import *

__all__ = [
    # Basic plotting
    'plot_distribution',
    'plot_correlation',
    'plot_scatter',
    'plot_line',
    'plot_bar',
    'plot_histogram',
    'plot_boxplot',
    'plot_heatmap',
    
    # Model visualization
    'plot_learning_curve',
    'plot_validation_curve',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_feature_importance',
    'plot_residuals',
    'plot_prediction_error',
    
    # Data visualization
    'plot_missing_data',
    'plot_outliers',
    'plot_class_distribution',
    'plot_feature_distributions',
    'plot_correlation_matrix',
    'plot_pairwise_relationships',
    
    # Interactive plots
    'interactive_scatter',
    'interactive_histogram',
    'interactive_correlation',
    'create_dashboard',
    
    # Interpretability
    'plot_shap_values',
    'plot_lime_explanation',
    'plot_permutation_importance',
    'plot_partial_dependence',
    'plot_feature_interaction',
]
