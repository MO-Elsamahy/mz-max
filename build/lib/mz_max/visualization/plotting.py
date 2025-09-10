"""
Core plotting functions for MZ Max

This module provides essential plotting functions for data visualization
and model analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Any
import warnings
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve, validation_curve

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

def plot_distribution(data: Union[pd.Series, np.ndarray], 
                     title: Optional[str] = None,
                     bins: int = 30,
                     kde: bool = True,
                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot distribution of a variable.
    
    Args:
        data: Data to plot
        title: Plot title
        bins: Number of histogram bins
        kde: Whether to show kernel density estimate
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, pd.Series):
        data_name = data.name or 'Variable'
        data = data.dropna()
    else:
        data_name = 'Variable'
        data = data[~np.isnan(data)] if data.dtype.kind in 'fc' else data
    
    # Plot histogram
    ax.hist(data, bins=bins, alpha=0.7, density=kde, edgecolor='black', linewidth=0.5)
    
    # Add KDE if requested
    if kde:
        try:
            from scipy import stats
            kde_data = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde_data(x_range), 'r-', linewidth=2, label='KDE')
            ax.legend()
        except ImportError:
            warnings.warn("scipy not available, skipping KDE")
    
    ax.set_xlabel(data_name)
    ax.set_ylabel('Density' if kde else 'Frequency')
    ax.set_title(title or f'Distribution of {data_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation(data: pd.DataFrame,
                    method: str = 'pearson',
                    figsize: Tuple[int, int] = (12, 10),
                    annot: bool = True,
                    cmap: str = 'RdBu_r') -> plt.Figure:
    """
    Plot correlation matrix.
    
    Args:
        data: DataFrame to compute correlations
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size
        annot: Whether to annotate cells
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute correlation matrix
    corr = data.corr(method=method)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr, mask=mask, annot=annot, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(f'{method.title()} Correlation Matrix')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: Union[np.ndarray, pd.Series],
                         y_pred: Union[np.ndarray, pd.Series],
                         labels: Optional[List[str]] = None,
                         normalize: Optional[str] = None,
                         figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Normalization method ('true', 'pred', 'all', None)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True, ax=ax,
                xticklabels=labels, yticklabels=labels)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    title = 'Confusion Matrix'
    if normalize:
        title += f' (Normalized by {normalize})'
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: Union[np.ndarray, pd.Series],
                  y_score: Union[np.ndarray, pd.Series],
                  pos_label: Optional[Any] = None,
                  figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores or probabilities
        pos_label: Positive class label
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true: Union[np.ndarray, pd.Series],
                               y_score: Union[np.ndarray, pd.Series],
                               pos_label: Optional[Any] = None,
                               figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores or probabilities
        pos_label: Positive class label
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    pr_auc = auc(recall, precision)
    
    # Plot precision-recall curve
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AUC = {pr_auc:.2f})')
    
    # Add baseline
    baseline = np.sum(y_true == (pos_label or 1)) / len(y_true)
    ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
               label=f'Baseline (Precision = {baseline:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance: Union[np.ndarray, pd.Series],
                          feature_names: Optional[List[str]] = None,
                          top_k: Optional[int] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance: Feature importance values
        feature_names: Feature names
        top_k: Number of top features to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(importance, pd.Series):
        feature_names = importance.index.tolist()
        importance = importance.values
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]
    
    # Create DataFrame for easier handling
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Select top_k features if specified
    if top_k is not None:
        df = df.tail(top_k)
    
    # Plot horizontal bar chart
    bars = ax.barh(df['feature'], df['importance'], color='skyblue', edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df['importance'])):
        ax.text(value + max(df['importance']) * 0.01, i, f'{value:.3f}',
                va='center', ha='left', fontsize=9)
    
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_learning_curve(estimator, X: Union[np.ndarray, pd.DataFrame], 
                       y: Union[np.ndarray, pd.Series],
                       cv: int = 5,
                       train_sizes: Optional[np.ndarray] = None,
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot learning curve.
    
    Args:
        estimator: Estimator to evaluate
        X: Training data
        y: Target values
        cv: Cross-validation folds
        train_sizes: Training set sizes to use
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot curves
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color='red')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_residuals(y_true: Union[np.ndarray, pd.Series],
                  y_pred: Union[np.ndarray, pd.Series],
                  figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot residuals for regression analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    residuals = y_true - y_pred
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_scatter(x: Union[np.ndarray, pd.Series],
                y: Union[np.ndarray, pd.Series],
                hue: Optional[Union[np.ndarray, pd.Series]] = None,
                size: Optional[Union[np.ndarray, pd.Series]] = None,
                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create scatter plot.
    
    Args:
        x: X-axis data
        y: Y-axis data
        hue: Color grouping variable
        size: Size variable
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create DataFrame for seaborn
    data_dict = {'x': x, 'y': y}
    if hue is not None:
        data_dict['hue'] = hue
    if size is not None:
        data_dict['size'] = size
    
    df = pd.DataFrame(data_dict)
    
    # Create scatter plot
    sns.scatterplot(data=df, x='x', y='y', hue='hue' if hue is not None else None,
                   size='size' if size is not None else None, ax=ax, alpha=0.7)
    
    ax.set_xlabel(x.name if hasattr(x, 'name') and x.name else 'X')
    ax.set_ylabel(y.name if hasattr(y, 'name') and y.name else 'Y')
    ax.set_title('Scatter Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_boxplot(data: Union[pd.DataFrame, List[Union[np.ndarray, pd.Series]]],
                labels: Optional[List[str]] = None,
                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create box plot.
    
    Args:
        data: Data to plot
        labels: Labels for each box
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, pd.DataFrame):
        data.boxplot(ax=ax)
        ax.set_title('Box Plot')
    else:
        ax.boxplot(data, labels=labels)
        ax.set_title('Box Plot')
        ax.set_ylabel('Values')
    
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_heatmap(data: Union[pd.DataFrame, np.ndarray],
                xticklabels: Optional[List[str]] = None,
                yticklabels: Optional[List[str]] = None,
                annot: bool = True,
                cmap: str = 'viridis',
                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create heatmap.
    
    Args:
        data: Data to plot
        xticklabels: X-axis labels
        yticklabels: Y-axis labels
        annot: Whether to annotate cells
        cmap: Colormap
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, annot=annot, cmap=cmap, ax=ax,
                xticklabels=xticklabels, yticklabels=yticklabels,
                square=True, linewidths=0.5)
    
    ax.set_title('Heatmap')
    plt.tight_layout()
    return fig
