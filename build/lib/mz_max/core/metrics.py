"""
Comprehensive metrics module for MZ Max

This module provides a unified interface for computing various evaluation
metrics for classification, regression, clustering, and other ML tasks.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)


class Metrics:
    """
    Comprehensive metrics computation class.
    
    This class provides methods to compute various evaluation metrics
    for different types of machine learning tasks.
    """
    
    @staticmethod
    def classification_metrics(y_true: Union[np.ndarray, pd.Series], 
                             y_pred: Union[np.ndarray, pd.Series],
                             y_proba: Optional[np.ndarray] = None,
                             average: str = 'weighted') -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary of classification metrics
        """
        metrics_dict = {}
        
        # Basic classification metrics
        metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
        metrics_dict['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics_dict['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics_dict['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC AUC (if probabilities are provided)
        if y_proba is not None:
            try:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    # Binary classification
                    metrics_dict['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                    # Multi-class classification
                    metrics_dict['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
                else:
                    # Binary with single probability column
                    metrics_dict['roc_auc'] = roc_auc_score(y_true, y_proba)
            except Exception:
                metrics_dict['roc_auc'] = np.nan
        
        # Confusion matrix statistics
        cm = sklearn_metrics.confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics_dict['true_negatives'] = int(tn)
            metrics_dict['false_positives'] = int(fp)
            metrics_dict['false_negatives'] = int(fn)
            metrics_dict['true_positives'] = int(tp)
            metrics_dict['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics_dict['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Classification report
        try:
            report = sklearn_metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics_dict['classification_report'] = report
        except Exception:
            pass
        
        return metrics_dict
    
    @staticmethod
    def regression_metrics(y_true: Union[np.ndarray, pd.Series], 
                          y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Compute comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        metrics_dict = {}
        
        # Basic regression metrics
        metrics_dict['mse'] = mean_squared_error(y_true, y_pred)
        metrics_dict['rmse'] = np.sqrt(metrics_dict['mse'])
        metrics_dict['mae'] = mean_absolute_error(y_true, y_pred)
        metrics_dict['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics_dict['mape'] = Metrics._mean_absolute_percentage_error(y_true, y_pred)
        metrics_dict['max_error'] = sklearn_metrics.max_error(y_true, y_pred)
        metrics_dict['explained_variance'] = sklearn_metrics.explained_variance_score(y_true, y_pred)
        
        # Residual statistics
        residuals = y_true - y_pred
        metrics_dict['mean_residual'] = float(np.mean(residuals))
        metrics_dict['std_residual'] = float(np.std(residuals))
        
        return metrics_dict
    
    @staticmethod
    def clustering_metrics(X: Union[np.ndarray, pd.DataFrame],
                          labels: Union[np.ndarray, pd.Series],
                          true_labels: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, float]:
        """
        Compute clustering evaluation metrics.
        
        Args:
            X: Input data
            labels: Cluster labels
            true_labels: True labels (if available)
            
        Returns:
            Dictionary of clustering metrics
        """
        metrics_dict = {}
        
        # Internal validation metrics
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters
            metrics_dict['silhouette_score'] = silhouette_score(X, labels)
            metrics_dict['calinski_harabasz_score'] = sklearn_metrics.calinski_harabasz_score(X, labels)
            metrics_dict['davies_bouldin_score'] = sklearn_metrics.davies_bouldin_score(X, labels)
        
        # External validation metrics (if true labels are provided)
        if true_labels is not None:
            metrics_dict['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
            metrics_dict['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, labels)
            metrics_dict['homogeneity_score'] = sklearn_metrics.homogeneity_score(true_labels, labels)
            metrics_dict['completeness_score'] = sklearn_metrics.completeness_score(true_labels, labels)
            metrics_dict['v_measure_score'] = sklearn_metrics.v_measure_score(true_labels, labels)
        
        return metrics_dict
    
    @staticmethod
    def ranking_metrics(y_true: Union[np.ndarray, pd.Series],
                       y_score: Union[np.ndarray, pd.Series],
                       k: Optional[int] = None) -> Dict[str, float]:
        """
        Compute ranking evaluation metrics.
        
        Args:
            y_true: True relevance scores
            y_score: Predicted scores
            k: Number of top items to consider
            
        Returns:
            Dictionary of ranking metrics
        """
        metrics_dict = {}
        
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        if k is None:
            k = len(y_true)
        
        # Precision@K
        metrics_dict[f'precision_at_{k}'] = np.sum(y_true_sorted[:k]) / k
        
        # Recall@K
        total_relevant = np.sum(y_true)
        if total_relevant > 0:
            metrics_dict[f'recall_at_{k}'] = np.sum(y_true_sorted[:k]) / total_relevant
        
        # NDCG@K
        metrics_dict[f'ndcg_at_{k}'] = Metrics._ndcg_at_k(y_true, y_score, k)
        
        return metrics_dict
    
    @staticmethod
    def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K."""
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate DCG@K
        dcg = 0.0
        for i in range(min(k, len(y_true_sorted))):
            dcg += (2 ** y_true_sorted[i] - 1) / np.log2(i + 2)
        
        # Calculate IDCG@K (ideal DCG)
        y_true_ideal = np.sort(y_true)[::-1]
        idcg = 0.0
        for i in range(min(k, len(y_true_ideal))):
            idcg += (2 ** y_true_ideal[i] - 1) / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def compute_all_metrics(task_type: str,
                           y_true: Union[np.ndarray, pd.Series],
                           y_pred: Union[np.ndarray, pd.Series],
                           X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                           y_proba: Optional[np.ndarray] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Compute all relevant metrics for a given task type.
        
        Args:
            task_type: Type of ML task ('classification', 'regression', 'clustering', 'ranking')
            y_true: True values/labels
            y_pred: Predicted values/labels
            X: Input data (required for clustering)
            y_proba: Predicted probabilities (optional for classification)
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dictionary of computed metrics
        """
        if task_type.lower() == 'classification':
            return Metrics.classification_metrics(y_true, y_pred, y_proba, **kwargs)
        elif task_type.lower() == 'regression':
            return Metrics.regression_metrics(y_true, y_pred, **kwargs)
        elif task_type.lower() == 'clustering':
            if X is None:
                raise ValueError("Input data X is required for clustering metrics")
            return Metrics.clustering_metrics(X, y_pred, y_true, **kwargs)
        elif task_type.lower() == 'ranking':
            return Metrics.ranking_metrics(y_true, y_pred, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
