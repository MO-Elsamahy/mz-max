"""
Automated Classification for MZ Max

This module provides automated classification capabilities with model
selection, hyperparameter optimization, and ensemble methods.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import optuna

from ..core.base import BaseEstimator
from ..core.metrics import Metrics
from ..core.exceptions import TrainingError


class AutoClassifier(BaseEstimator):
    """
    Automated classifier that automatically selects the best model
    and hyperparameters for classification tasks.
    """
    
    def __init__(self,
                 time_limit: int = 300,
                 max_trials: int = 100,
                 cv_folds: int = 5,
                 scoring: str = 'accuracy',
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = True):
        """
        Initialize AutoClassifier.
        
        Args:
            time_limit: Time limit in seconds for optimization
            max_trials: Maximum number of trials for hyperparameter optimization
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for model evaluation
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
        """
        super().__init__()
        self.time_limit = time_limit
        self.max_trials = max_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.best_model = None
        self.best_score = -np.inf
        self.model_scores = {}
        self.optimization_history = []
        
        # Define model search space
        self.model_space = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 4, 5, 6, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 4, 5, 6, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [31, 50, 70, 100],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000, 3000]
                }
            },
            'svm': {
                'model': SVC,
                'params': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                    'probability': [True]
                }
            },
            'knn': {
                'model': KNeighborsClassifier,
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'naive_bayes': {
                'model': GaussianNB,
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the AutoClassifier to training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            self: Returns the instance itself
        """
        X = self._validate_input(X)
        y = np.asarray(y)
        
        if self.verbose:
            print("Starting automated model selection and hyperparameter optimization...")
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Try each model type
        for model_name, model_config in self.model_space.items():
            if self.verbose:
                print(f"\nOptimizing {model_name}...")
            
            try:
                best_score, best_params = self._optimize_model(
                    model_config, X, y, cv, model_name
                )
                
                self.model_scores[model_name] = {
                    'score': best_score,
                    'params': best_params
                }
                
                if best_score > self.best_score:
                    self.best_score = best_score
                    self.best_model = model_config['model'](**best_params, random_state=self.random_state)
                    
                if self.verbose:
                    print(f"{model_name} best score: {best_score:.4f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error optimizing {model_name}: {str(e)}")
                continue
        
        if self.best_model is None:
            raise TrainingError("No models could be successfully trained")
        
        # Train the best model on full data
        self.best_model.fit(X, y)
        self._fitted = True
        
        if self.verbose:
            print(f"\nBest model: {self.best_model.__class__.__name__}")
            print(f"Best cross-validation score: {self.best_score:.4f}")
        
        return self
    
    def _optimize_model(self, model_config: Dict, X: np.ndarray, y: np.ndarray, 
                       cv: Any, model_name: str) -> Tuple[float, Dict]:
        """Optimize hyperparameters for a specific model."""
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_values in model_config['params'].items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                        else:
                            params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Handle special cases
            if model_name == 'logistic_regression' and params.get('penalty') == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            # Create model
            try:
                model = model_config['model'](**params, random_state=self.random_state, n_jobs=1)
                
                # Evaluate model
                scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring, n_jobs=1)
                return scores.mean()
            
            except Exception:
                return -np.inf
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=min(self.max_trials, 50),  # Limit trials per model
            timeout=self.time_limit // len(self.model_space),
            show_progress_bar=False
        )
        
        return study.best_value, study.best_params
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted():
            raise ValueError("AutoClassifier must be fitted before making predictions")
        
        X = self._validate_input(X)
        return self.best_model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted():
            raise ValueError("AutoClassifier must be fitted before making predictions")
        
        X = self._validate_input(X)
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            raise ValueError("Best model does not support probability prediction")
    
    def _compute_metrics(self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        y_proba = None
        if hasattr(self.best_model, 'predict_proba'):
            try:
                y_proba = self.best_model.predict_proba(self._validate_input(y_true))
            except:
                pass
        
        return Metrics.classification_metrics(y_true, y_pred, y_proba)
    
    def get_model_rankings(self) -> pd.DataFrame:
        """
        Get rankings of all evaluated models.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.model_scores:
            return pd.DataFrame()
        
        rankings = []
        for model_name, results in self.model_scores.items():
            rankings.append({
                'model': model_name,
                'cv_score': results['score'],
                'is_best': model_name == self.best_model.__class__.__name__.lower()
            })
        
        df = pd.DataFrame(rankings)
        return df.sort_values('cv_score', ascending=False).reset_index(drop=True)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from the best model."""
        if not self.is_fitted():
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            return self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            return np.abs(self.best_model.coef_).flatten()
        else:
            return None
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters."""
        if not self.is_fitted():
            return {}
        
        return self.best_model.get_params()
    
    def summary(self) -> str:
        """Get a summary of the AutoClassifier results."""
        if not self.is_fitted():
            return "AutoClassifier has not been fitted yet."
        
        summary = f"""
AutoClassifier Summary:
======================
Best Model: {self.best_model.__class__.__name__}
Best CV Score: {self.best_score:.4f}
Number of Models Evaluated: {len(self.model_scores)}

Model Rankings:
{self.get_model_rankings().to_string(index=False)}

Best Model Parameters:
{self.get_best_params()}
"""
        return summary
