"""
Data preprocessing utilities for MZ Max

This module provides data preprocessing functions including
cleaning, transformation, and feature engineering utilities.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from ..utils.logging import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame, 
               remove_duplicates: bool = True,
               handle_missing: str = 'drop',
               missing_threshold: float = 0.5) -> pd.DataFrame:
    """
    Clean DataFrame by handling duplicates and missing values.
    
    Args:
        df: Input DataFrame
        remove_duplicates: Whether to remove duplicate rows
        handle_missing: How to handle missing values ('drop', 'fill', 'keep')
        missing_threshold: Threshold for dropping columns with missing values
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    logger.info(f"Starting data cleaning. Initial shape: {df_clean.shape}")
    
    # Remove duplicates
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    if handle_missing == 'drop':
        # Drop columns with too many missing values
        missing_cols = []
        for col in df_clean.columns:
            missing_ratio = df_clean[col].isnull().sum() / len(df_clean)
            if missing_ratio > missing_threshold:
                missing_cols.append(col)
        
        if missing_cols:
            df_clean = df_clean.drop(columns=missing_cols)
            logger.info(f"Dropped columns with >{missing_threshold*100}% missing: {missing_cols}")
        
        # Drop rows with any missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            logger.info(f"Dropped {removed_rows} rows with missing values")
    
    elif handle_missing == 'fill':
        # Fill missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # Fill categorical columns with mode
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown')
        
        logger.info("Filled missing values")
    
    logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean


def scale_features(X: Union[pd.DataFrame, np.ndarray], 
                  method: str = 'standard',
                  feature_range: Tuple[float, float] = (0, 1)) -> Union[pd.DataFrame, np.ndarray]:
    """
    Scale features using different methods.
    
    Args:
        X: Input features
        method: Scaling method ('standard', 'minmax', 'robust')
        feature_range: Range for MinMax scaling
        
    Returns:
        Scaled features
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    if isinstance(X, pd.DataFrame):
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Features scaled using {method} method")
    return X_scaled


def encode_categorical(df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      method: str = 'onehot',
                      drop_first: bool = True) -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: Input DataFrame
        columns: Columns to encode (None for auto-detection)
        method: Encoding method ('onehot', 'label', 'target')
        drop_first: Whether to drop first category in one-hot encoding
        
    Returns:
        DataFrame with encoded categorical variables
    """
    df_encoded = df.copy()
    
    # Auto-detect categorical columns if not specified
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not columns:
        logger.info("No categorical columns found")
        return df_encoded
    
    logger.info(f"Encoding categorical columns: {columns} using {method} method")
    
    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=drop_first)
    
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        for col in columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    elif method == 'target':
        # Target encoding requires target variable - placeholder implementation
        logger.warning("Target encoding requires target variable. Using label encoding instead.")
        from sklearn.preprocessing import LabelEncoder
        for col in columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    logger.info(f"Categorical encoding completed. Shape: {df_encoded.shape}")
    return df_encoded


def handle_outliers(X: Union[pd.DataFrame, np.ndarray],
                   method: str = 'iqr',
                   threshold: float = 1.5,
                   action: str = 'remove') -> Union[pd.DataFrame, np.ndarray]:
    """
    Handle outliers in data.
    
    Args:
        X: Input data
        method: Outlier detection method ('iqr', 'zscore', 'isolation')
        threshold: Threshold for outlier detection
        action: Action to take ('remove', 'clip', 'transform')
        
    Returns:
        Data with outliers handled
    """
    if isinstance(X, pd.DataFrame):
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
    else:
        X_processed = X.copy()
        numeric_cols = range(X.shape[1])
    
    outlier_mask = np.zeros(len(X_processed), dtype=bool)
    
    for col in numeric_cols:
        if isinstance(X, pd.DataFrame):
            values = X_processed[col].values
        else:
            values = X_processed[:, col]
        
        if method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_outliers = (values < lower_bound) | (values > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            col_outliers = z_scores > threshold
        
        elif method == 'isolation':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            col_outliers = iso_forest.fit_predict(values.reshape(-1, 1)) == -1
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outlier_mask |= col_outliers
        
        # Handle outliers based on action
        if action == 'clip':
            if method == 'iqr':
                if isinstance(X, pd.DataFrame):
                    X_processed[col] = np.clip(X_processed[col], lower_bound, upper_bound)
                else:
                    X_processed[:, col] = np.clip(X_processed[:, col], lower_bound, upper_bound)
    
    if action == 'remove':
        if isinstance(X, pd.DataFrame):
            X_processed = X_processed[~outlier_mask]
        else:
            X_processed = X_processed[~outlier_mask]
        
        logger.info(f"Removed {outlier_mask.sum()} outlier rows")
    
    elif action == 'clip':
        logger.info(f"Clipped outliers using {method} method")
    
    return X_processed


def impute_missing_values(X: Union[pd.DataFrame, np.ndarray],
                         strategy: str = 'mean',
                         n_neighbors: int = 5) -> Union[pd.DataFrame, np.ndarray]:
    """
    Impute missing values.
    
    Args:
        X: Input data with missing values
        strategy: Imputation strategy ('mean', 'median', 'mode', 'knn')
        n_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        Data with imputed missing values
    """
    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
    
    if isinstance(X, pd.DataFrame):
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        X_imputed = X.copy()
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            X_imputed[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
        # Impute categorical columns with mode
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_imputed[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    else:
        X_imputed = imputer.fit_transform(X)
    
    logger.info(f"Missing values imputed using {strategy} strategy")
    return X_imputed


def feature_selection(X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     k: int = 10,
                     method: str = 'f_classif') -> Union[pd.DataFrame, np.ndarray]:
    """
    Select top k features based on statistical tests.
    
    Args:
        X: Input features
        y: Target variable
        k: Number of features to select
        method: Selection method ('f_classif', 'f_regression')
        
    Returns:
        Selected features
    """
    if method == 'f_classif':
        score_func = f_classif
    elif method == 'f_regression':
        score_func = f_regression
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    selector = SelectKBest(score_func=score_func, k=k)
    
    if isinstance(X, pd.DataFrame):
        X_selected = pd.DataFrame(
            selector.fit_transform(X, y),
            columns=X.columns[selector.get_support()],
            index=X.index
        )
    else:
        X_selected = selector.fit_transform(X, y)
    
    logger.info(f"Selected top {k} features using {method}")
    return X_selected


def create_polynomial_features(X: Union[pd.DataFrame, np.ndarray],
                             degree: int = 2,
                             interaction_only: bool = False,
                             include_bias: bool = False) -> Union[pd.DataFrame, np.ndarray]:
    """
    Create polynomial features.
    
    Args:
        X: Input features
        degree: Polynomial degree
        interaction_only: Whether to include interaction terms only
        include_bias: Whether to include bias term
        
    Returns:
        Polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias
    )
    
    if isinstance(X, pd.DataFrame):
        X_poly = pd.DataFrame(
            poly.fit_transform(X),
            index=X.index
        )
        # Create feature names
        feature_names = poly.get_feature_names_out(X.columns)
        X_poly.columns = feature_names
    else:
        X_poly = poly.fit_transform(X)
    
    logger.info(f"Created polynomial features with degree {degree}. Shape: {X_poly.shape}")
    return X_poly


def detect_and_remove_constant_features(X: Union[pd.DataFrame, np.ndarray],
                                       threshold: float = 0.01) -> Union[pd.DataFrame, np.ndarray]:
    """
    Detect and remove constant or quasi-constant features.
    
    Args:
        X: Input features
        threshold: Variance threshold for quasi-constant detection
        
    Returns:
        Features with constant features removed
    """
    from sklearn.feature_selection import VarianceThreshold
    
    selector = VarianceThreshold(threshold=threshold)
    
    if isinstance(X, pd.DataFrame):
        X_filtered = pd.DataFrame(
            selector.fit_transform(X),
            columns=X.columns[selector.get_support()],
            index=X.index
        )
        removed_features = X.columns[~selector.get_support()].tolist()
        if removed_features:
            logger.info(f"Removed constant/quasi-constant features: {removed_features}")
    else:
        X_filtered = selector.fit_transform(X)
        removed_count = X.shape[1] - X_filtered.shape[1]
        if removed_count > 0:
            logger.info(f"Removed {removed_count} constant/quasi-constant features")
    
    return X_filtered


def balance_dataset(X: Union[pd.DataFrame, np.ndarray],
                   y: Union[pd.Series, np.ndarray],
                   method: str = 'smote',
                   random_state: int = 42) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Balance dataset using various techniques.
    
    Args:
        X: Input features
        y: Target variable
        method: Balancing method ('smote', 'undersample', 'oversample')
        random_state: Random state for reproducibility
        
    Returns:
        Balanced features and target
    """
    try:
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            balancer = SMOTE(random_state=random_state)
        elif method == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            balancer = RandomUnderSampler(random_state=random_state)
        elif method == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            balancer = RandomOverSampler(random_state=random_state)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        
        logger.info(f"Dataset balanced using {method}. New shape: {X_balanced.shape}")
        return X_balanced, y_balanced
    
    except ImportError:
        logger.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")
        return X, y


class DataPreprocessor:
    """
    Comprehensive data preprocessor with configurable pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {
            'clean_data': True,
            'handle_outliers': True,
            'scale_features': True,
            'encode_categorical': True,
            'feature_selection': False,
            'balance_dataset': False
        }
        self.is_fitted = False
        self.feature_names = None
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit preprocessor and transform data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Transformed features and target
        """
        logger.info("Starting data preprocessing pipeline")
        
        X_processed = X.copy()
        y_processed = y.copy() if y is not None else None
        
        # Clean data
        if self.config.get('clean_data', True):
            X_processed = clean_data(X_processed)
            if y is not None:
                y_processed = y_processed.loc[X_processed.index]
        
        # Handle outliers
        if self.config.get('handle_outliers', True):
            outlier_method = self.config.get('outlier_method', 'iqr')
            outlier_action = self.config.get('outlier_action', 'remove')
            
            if outlier_action == 'remove':
                # Get outlier mask and apply to both X and y
                original_shape = X_processed.shape[0]
                X_processed = handle_outliers(X_processed, method=outlier_method, action=outlier_action)
                if y is not None and X_processed.shape[0] < original_shape:
                    y_processed = y_processed.loc[X_processed.index]
            else:
                X_processed = handle_outliers(X_processed, method=outlier_method, action=outlier_action)
        
        # Encode categorical variables
        if self.config.get('encode_categorical', True):
            encoding_method = self.config.get('encoding_method', 'onehot')
            X_processed = encode_categorical(X_processed, method=encoding_method)
        
        # Scale features
        if self.config.get('scale_features', True):
            scaling_method = self.config.get('scaling_method', 'standard')
            X_processed = scale_features(X_processed, method=scaling_method)
        
        # Feature selection
        if self.config.get('feature_selection', False) and y is not None:
            k_features = self.config.get('k_features', 10)
            selection_method = self.config.get('selection_method', 'f_classif')
            X_processed = feature_selection(X_processed, y_processed, k=k_features, method=selection_method)
        
        # Balance dataset
        if self.config.get('balance_dataset', False) and y is not None:
            balance_method = self.config.get('balance_method', 'smote')
            X_processed, y_processed = balance_dataset(X_processed, y_processed, method=balance_method)
        
        self.feature_names = X_processed.columns.tolist() if isinstance(X_processed, pd.DataFrame) else None
        self.is_fitted = True
        
        logger.info(f"Preprocessing pipeline completed. Final shape: {X_processed.shape}")
        return X_processed, y_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # This is a simplified transform - in practice, you'd need to store
        # fitted transformers and apply them here
        logger.warning("Transform method is simplified. Use fit_transform for full preprocessing.")
        return X
