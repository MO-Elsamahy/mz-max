"""
Test suite for AutoML functionality
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from mz_max.automl import AutoClassifier, AutoRegressor


class TestAutoClassifier:
    """Test AutoClassifier functionality."""
    
    @pytest.fixture
    def classification_data(self):
        """Create test classification data."""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=10,
            n_classes=3,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_automl_classifier_initialization(self):
        """Test AutoClassifier initialization."""
        classifier = AutoClassifier(
            time_limit=60,
            max_trials=10,
            cv_folds=3,
            random_state=42
        )
        
        assert classifier.time_limit == 60
        assert classifier.max_trials == 10
        assert classifier.cv_folds == 3
        assert classifier.random_state == 42
        assert not classifier.is_fitted()
    
    def test_automl_classifier_fit_predict(self, classification_data):
        """Test AutoClassifier fit and predict."""
        X_train, X_test, y_train, y_test = classification_data
        
        classifier = AutoClassifier(
            time_limit=30,  # Short time for testing
            max_trials=5,
            cv_folds=3,
            verbose=False,
            random_state=42
        )
        
        # Fit the classifier
        classifier.fit(X_train, y_train)
        
        # Check if fitted
        assert classifier.is_fitted()
        assert classifier.best_model is not None
        assert classifier.best_score > 0
        
        # Make predictions
        predictions = classifier.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert predictions.dtype in [np.int32, np.int64]
        
        # Check prediction range
        unique_predictions = np.unique(predictions)
        unique_labels = np.unique(y_train)
        assert all(pred in unique_labels for pred in unique_predictions)
    
    def test_automl_classifier_predict_proba(self, classification_data):
        """Test AutoClassifier predict_proba."""
        X_train, X_test, y_train, y_test = classification_data
        
        classifier = AutoClassifier(
            time_limit=30,
            max_trials=5,
            verbose=False,
            random_state=42
        )
        
        classifier.fit(X_train, y_train)
        
        # Test predict_proba
        if hasattr(classifier.best_model, 'predict_proba'):
            probabilities = classifier.predict_proba(X_test)
            
            assert probabilities.shape[0] == len(X_test)
            assert probabilities.shape[1] == len(np.unique(y_train))
            
            # Check if probabilities sum to 1
            prob_sums = np.sum(probabilities, axis=1)
            np.testing.assert_array_almost_equal(prob_sums, np.ones(len(X_test)))
    
    def test_automl_classifier_evaluate(self, classification_data):
        """Test AutoClassifier evaluate method."""
        X_train, X_test, y_train, y_test = classification_data
        
        classifier = AutoClassifier(
            time_limit=30,
            max_trials=5,
            verbose=False,
            random_state=42
        )
        
        classifier.fit(X_train, y_train)
        
        # Evaluate model
        metrics = classifier.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_automl_classifier_get_model_rankings(self, classification_data):
        """Test get_model_rankings method."""
        X_train, X_test, y_train, y_test = classification_data
        
        classifier = AutoClassifier(
            time_limit=30,
            max_trials=5,
            verbose=False,
            random_state=42
        )
        
        classifier.fit(X_train, y_train)
        
        # Get model rankings
        rankings = classifier.get_model_rankings()
        
        assert isinstance(rankings, pd.DataFrame)
        assert len(rankings) > 0
        assert 'model' in rankings.columns
        assert 'cv_score' in rankings.columns
        
        # Check if sorted by score
        scores = rankings['cv_score'].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


class TestAutoRegressor:
    """Test AutoRegressor functionality."""
    
    @pytest.fixture
    def regression_data(self):
        """Create test regression data."""
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            noise=0.1,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_automl_regressor_initialization(self):
        """Test AutoRegressor initialization."""
        regressor = AutoRegressor(
            time_limit=60,
            max_trials=10,
            cv_folds=3,
            random_state=42
        )
        
        assert regressor.time_limit == 60
        assert regressor.max_trials == 10
        assert regressor.cv_folds == 3
        assert regressor.random_state == 42
        assert not regressor.is_fitted()
    
    def test_automl_regressor_fit_predict(self, regression_data):
        """Test AutoRegressor fit and predict."""
        X_train, X_test, y_train, y_test = regression_data
        
        regressor = AutoRegressor(
            time_limit=30,
            max_trials=5,
            cv_folds=3,
            verbose=False,
            random_state=42
        )
        
        # Fit the regressor
        regressor.fit(X_train, y_train)
        
        # Check if fitted
        assert regressor.is_fitted()
        assert regressor.best_model is not None
        
        # Make predictions
        predictions = regressor.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert predictions.dtype in [np.float32, np.float64]
    
    def test_automl_regressor_evaluate(self, regression_data):
        """Test AutoRegressor evaluate method."""
        X_train, X_test, y_train, y_test = regression_data
        
        regressor = AutoRegressor(
            time_limit=30,
            max_trials=5,
            verbose=False,
            random_state=42
        )
        
        regressor.fit(X_train, y_train)
        
        # Evaluate model
        metrics = regressor.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        
        # Check R² range
        assert -1 <= metrics['r2'] <= 1
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0


class TestAutoMLIntegration:
    """Integration tests for AutoML components."""
    
    def test_pandas_dataframe_input(self):
        """Test AutoML with pandas DataFrame input."""
        # Create DataFrame
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        target = pd.Series(y, name='target')
        
        classifier = AutoClassifier(
            time_limit=30,
            max_trials=3,
            verbose=False,
            random_state=42
        )
        
        # Fit with DataFrame
        classifier.fit(df, target)
        
        assert classifier.is_fitted()
        
        # Predict with DataFrame
        predictions = classifier.predict(df.head(10))
        assert len(predictions) == 10
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        
        classifier = AutoClassifier(
            time_limit=30,
            max_trials=3,
            verbose=False,
            random_state=42
        )
        
        classifier.fit(X, y)
        
        # Get feature importance
        importance = classifier.get_feature_importance()
        
        if importance is not None:
            assert len(importance) == X.shape[1]
            assert all(imp >= 0 for imp in importance)
    
    def test_model_persistence(self, tmp_path):
        """Test model save and load functionality."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        
        classifier = AutoClassifier(
            time_limit=30,
            max_trials=3,
            verbose=False,
            random_state=42
        )
        
        classifier.fit(X, y)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        classifier.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_classifier = AutoClassifier.load(str(model_path))
        
        assert loaded_classifier.is_fitted()
        
        # Test predictions are the same
        original_pred = classifier.predict(X[:10])
        loaded_pred = loaded_classifier.predict(X[:10])
        
        np.testing.assert_array_equal(original_pred, loaded_pred)


if __name__ == "__main__":
    pytest.main([__file__])
