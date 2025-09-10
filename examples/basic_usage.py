"""
Basic usage examples for MZ Max

This script demonstrates the basic functionality of MZ Max.
"""

import mz_max as mz
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def classification_example():
    """Example of using MZ Max for classification."""
    print("="*50)
    print("CLASSIFICATION EXAMPLE")
    print("="*50)
    
    # Load a dataset
    print("Loading iris dataset...")
    data = mz.load_dataset('iris')
    print(f"Dataset shape: {data.shape}")
    print(f"Dataset columns: {data.columns.tolist()}")
    
    # Prepare features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train model with AutoML
    print("\nTraining AutoClassifier...")
    model = mz.automl.AutoClassifier(
        time_limit=60,  # 1 minute for demo
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")
    
    # Get model rankings
    print("\nModel Rankings:")
    rankings = model.get_model_rankings()
    print(rankings)
    
    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print("\nFeature Importance:")
        for i, (feature, imp) in enumerate(zip(X.columns, importance)):
            print(f"  {feature}: {imp:.4f}")
    
    return model, X_test, y_test, predictions


def regression_example():
    """Example of using MZ Max for regression."""
    print("\n" + "="*50)
    print("REGRESSION EXAMPLE")
    print("="*50)
    
    # Load a dataset
    print("Loading diabetes dataset...")
    data = mz.load_dataset('diabetes')
    print(f"Dataset shape: {data.shape}")
    
    # Prepare features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with AutoML
    print("\nTraining AutoRegressor...")
    model = mz.automl.AutoRegressor(
        time_limit=60,  # 1 minute for demo
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest R² Score: {metrics['r2']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    
    return model, X_test, y_test, predictions


def visualization_example(model, X_test, y_test, predictions):
    """Example of using MZ Max visualization."""
    print("\n" + "="*50)
    print("VISUALIZATION EXAMPLE")
    print("="*50)
    
    # Import visualization functions
    from mz_max.visualization import plot_confusion_matrix, plot_feature_importance
    
    # Plot confusion matrix (for classification)
    if hasattr(model, 'predict_proba'):
        print("Creating confusion matrix...")
        fig = plot_confusion_matrix(y_test, predictions)
        fig.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Plot feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print("Creating feature importance plot...")
        fig = plot_feature_importance(importance, X_test.columns.tolist())
        fig.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved as 'feature_importance.png'")


def data_loading_example():
    """Example of data loading capabilities."""
    print("\n" + "="*50)
    print("DATA LOADING EXAMPLE")
    print("="*50)
    
    # List available datasets
    print("Available built-in datasets:")
    datasets = ['iris', 'diabetes', 'wine', 'breast_cancer', 'boston']
    for dataset in datasets:
        print(f"  - {dataset}")
    
    # Load different types of datasets
    print("\nLoading wine dataset...")
    wine_data = mz.load_dataset('wine')
    print(f"Wine dataset shape: {wine_data.shape}")
    print(f"Wine dataset target distribution:")
    print(wine_data['target'].value_counts())
    
    # Example of loading external data (if file exists)
    try:
        print("\nTrying to load external CSV file...")
        # This would work if you have a CSV file
        # external_data = mz.data.load_data_file('your_data.csv')
        # print(f"External data shape: {external_data.shape}")
        print("(No external CSV file provided for demo)")
    except:
        print("(No external CSV file found)")


def model_registry_example():
    """Example of using the model registry."""
    print("\n" + "="*50)
    print("MODEL REGISTRY EXAMPLE")
    print("="*50)
    
    # List available models
    print("Available models:")
    models = mz.list_models()
    for model in models[:10]:  # Show first 10
        print(f"  - {model}")
    
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more models")
    
    # Load a specific model
    print("\nLoading Random Forest Classifier...")
    rf_model = mz.load_model('random_forest_classifier', n_estimators=100)
    print(f"Model type: {type(rf_model)}")
    
    # Get model info
    print("\nModel information:")
    info = mz.models.get_model_info('random_forest_classifier')
    print(f"  Description: {info.get('description', 'N/A')}")
    print(f"  Task type: {info.get('task_type', 'N/A')}")
    print(f"  Framework: {info.get('framework', 'N/A')}")


def main():
    """Run all examples."""
    print("MZ Max - Basic Usage Examples")
    print("============================")
    
    # Classification example
    model, X_test, y_test, predictions = classification_example()
    
    # Regression example
    regression_example()
    
    # Visualization example
    visualization_example(model, X_test, y_test, predictions)
    
    # Data loading example
    data_loading_example()
    
    # Model registry example
    model_registry_example()
    
    print("\n" + "="*50)
    print("ALL EXAMPLES COMPLETED!")
    print("="*50)
    print("\nCheck the generated files:")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("\nFor more examples, see the documentation at: https://docs.mzmax.ai")


if __name__ == "__main__":
    main()
