"""
Command Line Interface for MZ Max

This module provides a command-line interface for common MZ Max operations.
"""

import argparse
import sys
from typing import Optional, List
import pandas as pd
import numpy as np
from pathlib import Path

from .automl import AutoClassifier, AutoRegressor
from .data.loaders import load_dataset
from .utils.logging import setup_logging, get_logger
from .visualization.plotting import plot_confusion_matrix, plot_feature_importance
from . import __version__


def main():
    """Main entry point for the MZ Max CLI."""
    parser = argparse.ArgumentParser(
        description="MZ Max - The Most Powerful ML/DL Python Package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mzmax train --data data.csv --target target_column --task classification
  mzmax predict --model model.pkl --data test.csv --output predictions.csv
  mzmax info --model-list
  mzmax benchmark --data data.csv --target target_column
        """
    )
    
    parser.add_argument('--version', action='version', version=f'MZ Max {__version__}')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model automatically')
    train_parser.add_argument('--data', required=True, help='Path to training data (CSV file)')
    train_parser.add_argument('--target', required=True, help='Target column name')
    train_parser.add_argument('--task', choices=['classification', 'regression'], required=True,
                            help='Machine learning task type')
    train_parser.add_argument('--output', default='model.pkl', help='Output model path')
    train_parser.add_argument('--time-limit', type=int, default=300, help='Time limit in seconds')
    train_parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--data', required=True, help='Path to prediction data (CSV file)')
    predict_parser.add_argument('--output', default='predictions.csv', help='Output predictions path')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get information about MZ Max')
    info_parser.add_argument('--model-list', action='store_true', help='List available models')
    info_parser.add_argument('--dataset-list', action='store_true', help='List available datasets')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark multiple models')
    benchmark_parser.add_argument('--data', required=True, help='Path to data (CSV file)')
    benchmark_parser.add_argument('--target', required=True, help='Target column name')
    benchmark_parser.add_argument('--output', default='benchmark_results.csv', help='Output results path')
    benchmark_parser.add_argument('--time-limit', type=int, default=600, help='Time limit in seconds')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level, log_file=args.log_file)
    logger = get_logger('mz_max.cli')
    
    # Execute command
    if args.command == 'train':
        train_command(args, logger)
    elif args.command == 'predict':
        predict_command(args, logger)
    elif args.command == 'info':
        info_command(args, logger)
    elif args.command == 'benchmark':
        benchmark_command(args, logger)
    else:
        parser.print_help()
        sys.exit(1)


def train_command(args, logger):
    """Execute the train command."""
    logger.info("Starting model training...")
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Prepare features and target
    X = data.drop(columns=[args.target])
    y = data[args.target]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    
    # Train model
    if args.task == 'classification':
        model = AutoClassifier(
            time_limit=args.time_limit,
            cv_folds=args.cv_folds,
            verbose=args.verbose
        )
    else:
        model = AutoRegressor(
            time_limit=args.time_limit,
            cv_folds=args.cv_folds,
            verbose=args.verbose
        )
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    test_score = model.evaluate(X_test, y_test)
    logger.info(f"Test score: {test_score}")
    
    # Save model
    logger.info(f"Saving model to {args.output}")
    model.save(args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(model.summary())
    
    logger.info("Training completed successfully!")


def predict_command(args, logger):
    """Execute the predict command."""
    logger.info("Starting prediction...")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    from .core.base import BaseEstimator
    model = BaseEstimator.load(args.model)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(data)
    
    # Save predictions
    logger.info(f"Saving predictions to {args.output}")
    pred_df = pd.DataFrame({
        'prediction': predictions
    })
    
    # Add probabilities if available
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(data)
            for i in range(probabilities.shape[1]):
                pred_df[f'probability_class_{i}'] = probabilities[:, i]
        except:
            pass
    
    pred_df.to_csv(args.output, index=False)
    
    print(f"\nPredictions saved to {args.output}")
    print(f"Number of predictions: {len(predictions)}")
    
    logger.info("Prediction completed successfully!")


def info_command(args, logger):
    """Execute the info command."""
    if args.model_list:
        from .models import list_models
        models = list_models()
        print("\nAvailable Models:")
        print("="*50)
        for model in models:
            print(f"  - {model}")
        print(f"\nTotal: {len(models)} models")
    
    elif args.dataset_list:
        print("\nBuilt-in Datasets:")
        print("="*50)
        datasets = [
            'iris', 'boston', 'diabetes', 'wine', 'breast_cancer',
            'digits', 'california_housing', 'fetch_20newsgroups'
        ]
        for dataset in datasets:
            print(f"  - {dataset}")
        print(f"\nTotal: {len(datasets)} datasets")
    
    else:
        print(f"\nMZ Max v{__version__}")
        print("The Most Powerful ML/DL Python Package")
        print("="*50)
        print("For more information, visit: https://github.com/mzmax/mz-max")
        print("Documentation: https://docs.mzmax.ai")
        print("\nUse --help for available commands and options")


def benchmark_command(args, logger):
    """Execute the benchmark command."""
    logger.info("Starting benchmark...")
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Prepare features and target
    X = data.drop(columns=[args.target])
    y = data[args.target]
    
    # Determine task type
    if len(np.unique(y)) <= 10 and np.all(np.unique(y) == np.unique(y).astype(int)):
        task_type = 'classification'
        model = AutoClassifier(time_limit=args.time_limit, verbose=args.verbose)
    else:
        task_type = 'regression'
        model = AutoRegressor(time_limit=args.time_limit, verbose=args.verbose)
    
    logger.info(f"Detected task type: {task_type}")
    
    # Train model
    logger.info("Training models for benchmark...")
    model.fit(X, y)
    
    # Get results
    results = model.get_model_rankings()
    
    # Save results
    logger.info(f"Saving benchmark results to {args.output}")
    results.to_csv(args.output, index=False)
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(results.to_string(index=False))
    
    logger.info("Benchmark completed successfully!")


if __name__ == '__main__':
    main()
