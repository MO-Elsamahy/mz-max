"""
Model registry implementation for MZ Max

This module provides the model registry functionality with access
to pre-trained models and custom model architectures.
"""

from typing import Dict, List, Optional, Any
import importlib
from ..core.registry import model_registry
from ..core.exceptions import ModelNotFoundError


def list_models(task_type: Optional[str] = None, framework: Optional[str] = None) -> List[str]:
    """
    List available models.
    
    Args:
        task_type: Filter by task type (optional)
        framework: Filter by framework (optional)
        
    Returns:
        List of model names
    """
    if task_type and framework:
        models = [name for name in model_registry.list_items() 
                 if (model_registry.get_info(name).get('task_type') == task_type and
                     model_registry.get_info(name).get('framework') == framework)]
    elif task_type:
        models = model_registry.list_by_task(task_type)
    elif framework:
        models = model_registry.list_by_framework(framework)
    else:
        models = model_registry.list_items()
    
    return sorted(models)


def load_model(name: str, **kwargs) -> Any:
    """
    Load a model from the registry.
    
    Args:
        name: Model name
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Model instance
    """
    try:
        return model_registry.create(name, **kwargs)
    except Exception as e:
        raise ModelNotFoundError(f"Failed to load model '{name}': {str(e)}")


def get_model_info(name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        name: Model name
        
    Returns:
        Model information dictionary
    """
    return model_registry.get_info(name)


# Register some default models
def _register_default_models():
    """Register default models from scikit-learn and other libraries."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        # Classification models
        model_registry.register('random_forest_classifier', RandomForestClassifier, 
                              description="Random Forest Classifier", 
                              task_type='classification', framework='sklearn')
        
        model_registry.register('logistic_regression', LogisticRegression,
                              description="Logistic Regression",
                              task_type='classification', framework='sklearn')
        
        model_registry.register('svc', SVC,
                              description="Support Vector Classifier",
                              task_type='classification', framework='sklearn')
        
        model_registry.register('knn_classifier', KNeighborsClassifier,
                              description="K-Nearest Neighbors Classifier",
                              task_type='classification', framework='sklearn')
        
        model_registry.register('naive_bayes', GaussianNB,
                              description="Gaussian Naive Bayes",
                              task_type='classification', framework='sklearn')
        
        model_registry.register('decision_tree_classifier', DecisionTreeClassifier,
                              description="Decision Tree Classifier",
                              task_type='classification', framework='sklearn')
        
        model_registry.register('gradient_boosting_classifier', GradientBoostingClassifier,
                              description="Gradient Boosting Classifier",
                              task_type='classification', framework='sklearn')
        
        # Regression models
        model_registry.register('random_forest_regressor', RandomForestRegressor,
                              description="Random Forest Regressor",
                              task_type='regression', framework='sklearn')
        
        model_registry.register('linear_regression', LinearRegression,
                              description="Linear Regression",
                              task_type='regression', framework='sklearn')
        
        model_registry.register('svr', SVR,
                              description="Support Vector Regressor",
                              task_type='regression', framework='sklearn')
        
        model_registry.register('knn_regressor', KNeighborsRegressor,
                              description="K-Nearest Neighbors Regressor",
                              task_type='regression', framework='sklearn')
        
        model_registry.register('decision_tree_regressor', DecisionTreeRegressor,
                              description="Decision Tree Regressor",
                              task_type='regression', framework='sklearn')
        
        model_registry.register('gradient_boosting_regressor', GradientBoostingRegressor,
                              description="Gradient Boosting Regressor",
                              task_type='regression', framework='sklearn')
        
    except ImportError:
        pass
    
    # Register XGBoost models if available
    try:
        import xgboost as xgb
        
        model_registry.register('xgb_classifier', xgb.XGBClassifier,
                              description="XGBoost Classifier",
                              task_type='classification', framework='xgboost')
        
        model_registry.register('xgb_regressor', xgb.XGBRegressor,
                              description="XGBoost Regressor",
                              task_type='regression', framework='xgboost')
        
    except ImportError:
        pass
    
    # Register LightGBM models if available
    try:
        import lightgbm as lgb
        
        model_registry.register('lgb_classifier', lgb.LGBMClassifier,
                              description="LightGBM Classifier",
                              task_type='classification', framework='lightgbm')
        
        model_registry.register('lgb_regressor', lgb.LGBMRegressor,
                              description="LightGBM Regressor",
                              task_type='regression', framework='lightgbm')
        
    except ImportError:
        pass
    
    # Register CatBoost models if available
    try:
        import catboost as cb
        
        model_registry.register('catboost_classifier', cb.CatBoostClassifier,
                              description="CatBoost Classifier",
                              task_type='classification', framework='catboost')
        
        model_registry.register('catboost_regressor', cb.CatBoostRegressor,
                              description="CatBoost Regressor",
                              task_type='regression', framework='catboost')
        
    except ImportError:
        pass


# Register default models on import
_register_default_models()
