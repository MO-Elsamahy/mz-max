"""
Base classes for MZ Max

This module defines the fundamental base classes that all estimators,
transformers, and models inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator as SklearnBaseEstimator


class BaseEstimator(SklearnBaseEstimator, ABC):
    """
    Base class for all estimators in MZ Max.
    
    This class extends scikit-learn's BaseEstimator to provide additional
    functionality specific to MZ Max.
    """
    
    def __init__(self, **kwargs):
        """Initialize the base estimator."""
        self._fitted = False
        self._feature_names = None
        self._n_features = None
        self.set_params(**kwargs)
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Fit the estimator to training data.
        
        Args:
            X: Training data
            y: Target values (optional for unsupervised learning)
            
        Returns:
            self: Returns the instance itself
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predictions
        """
        pass
    
    def is_fitted(self) -> bool:
        """Check if the estimator has been fitted."""
        return self._fitted
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input data."""
        if isinstance(X, pd.DataFrame):
            if self._feature_names is None:
                self._feature_names = X.columns.tolist()
            X = X.values
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
        
        if self._n_features is None:
            self._n_features = X.shape[1]
        elif X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}")
        
        return X
    
    def save(self, filepath: str) -> None:
        """Save the fitted estimator to disk."""
        import joblib
        if not self.is_fitted():
            raise ValueError("Cannot save unfitted estimator")
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseEstimator':
        """Load a fitted estimator from disk."""
        import joblib
        return joblib.load(filepath)


class BaseTransformer(BaseEstimator):
    """
    Base class for all transformers in MZ Max.
    
    Transformers are estimators that can transform data without necessarily
    having a prediction target.
    """
    
    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform the input data.
        
        Args:
            X: Input data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit the transformer and transform the data in one step.
        
        Args:
            X: Input data
            y: Target values (optional)
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """For transformers, predict is equivalent to transform."""
        return self.transform(X)


class BaseModel(BaseEstimator):
    """
    Base class for all models in MZ Max.
    
    This class provides additional functionality specific to machine learning
    and deep learning models.
    """
    
    def __init__(self, **kwargs):
        """Initialize the base model."""
        super().__init__(**kwargs)
        self._model = None
        self._training_history = []
        self._metrics = {}
    
    @property
    def model(self) -> Any:
        """Get the underlying model object."""
        return self._model
    
    @property
    def training_history(self) -> List[Dict[str, Any]]:
        """Get the training history."""
        return self._training_history
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get the model metrics."""
        return self._metrics
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        return self._compute_metrics(y, predictions)
    
    @abstractmethod
    def _compute_metrics(self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None if not available
        """
        if hasattr(self._model, 'feature_importances_'):
            return self._model.feature_importances_
        elif hasattr(self._model, 'coef_'):
            return np.abs(self._model.coef_).flatten()
        else:
            return None
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        Predict class probabilities if available.
        
        Args:
            X: Input data
            
        Returns:
            Predicted probabilities or None if not available
        """
        if hasattr(self._model, 'predict_proba'):
            X = self._validate_input(X)
            return self._model.predict_proba(X)
        else:
            return None
    
    def get_params_summary(self) -> Dict[str, Any]:
        """Get a summary of model parameters."""
        params = self.get_params()
        summary = {
            'model_type': self.__class__.__name__,
            'fitted': self.is_fitted(),
            'n_features': self._n_features,
            'feature_names': self._feature_names,
            'parameters': params
        }
        return summary
