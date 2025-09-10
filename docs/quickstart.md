---
layout: default
title: Quick Start Guide
description: Get started with MZ Max in minutes
---

# Quick Start Guide

Get up and running with MZ Max in just a few minutes.

## 1. Installation

First, install MZ Max :

```bash
pip install mz-max
```

## 2. Basic Usage

### Load Your First Dataset

```python
import mz_max as mz

# Load a built-in dataset
data = mz.load_dataset('iris')
print(f"Dataset shape: {data.shape}")
print(data.head())
```

### Train Your First Model

```python
# Create an AutoML pipeline
automl = mz.AutoML(task_type='classification')

# Train the model
model = automl.fit(data)

# Make predictions
predictions = model.predict(data.drop('target', axis=1))
print(f"Predictions: {predictions[:5]}")
```

### Evaluate Model Performance

```python
from sklearn.model_selection import train_test_split

# Split data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
model = automl.fit(X_train, y_train)
metrics = mz.evaluate_model(model, X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

## 3. Professional User Interfaces

### Launch Web Dashboard

```bash
# Command line
mzmax-dashboard

# Or programmatically
python -c "from mz_max.ui.dashboard import create_dashboard; create_dashboard()"
```

Access at: [http://localhost:8501](http://localhost:8501)

### Launch REST API & Web App

```bash
# Command line
mzmax-webapp

# Or programmatically  
python -c "from mz_max.ui.web_app import launch_web_app; launch_web_app()"
```

Access at: [http://localhost:8000](http://localhost:8000)
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Launch Desktop GUI

```bash
# Command line
mzmax-gui

# Or programmatically
python -c "from mz_max.ui.gui import launch_gui; launch_gui()"
```

### Use Jupyter Widgets

```python
# In a Jupyter notebook
from mz_max.ui.widgets import create_complete_workflow

# Create interactive ML workflow
widgets = create_complete_workflow()
```

## 4. Advanced Features

### Enterprise Security

```python
from mz_max.enterprise import SecurityManager

# Initialize security manager
security = SecurityManager()

# Encrypt sensitive data
sensitive_data = {"customer_id": 12345, "credit_score": 750}
encrypted = security.encrypt_data(sensitive_data)
print(f"Encrypted: {encrypted}")

# Decrypt data
decrypted = security.decrypt_data(encrypted)
print(f"Decrypted: {decrypted}")
```

### Cloud Integration

```python
from mz_max.cloud import AWSIntegration

# Connect to AWS (requires credentials)
aws = AWSIntegration(region='us-east-1')

# Deploy model to SageMaker
# aws.deploy_to_sagemaker(model=trained_model)
```

### Custom Data Pipeline

```python
from mz_max.data import DataPreprocessor

# Advanced preprocessing
preprocessor = DataPreprocessor(
    handle_missing='auto',
    feature_scaling='standard',
    categorical_encoding='target'
)

# Process your data
processed_data = preprocessor.fit_transform(raw_data)
```

## 5. Working with Different Data Types

### CSV Files

```python
import pandas as pd

# Load CSV data
data = pd.read_csv('your_data.csv')

# Use with MZ Max
automl = mz.AutoML()
model = automl.fit(data)
```

### Image Data

```python
from mz_max.deep_learning import ImageClassifier

# Create image classifier
classifier = ImageClassifier(
    model_type='resnet50',
    num_classes=10
)

# Train on image data
# classifier.fit(image_data, labels)
```

### Text Data

```python
from mz_max.nlp import TextClassifier

# Create text classifier
text_classifier = TextClassifier(
    model_type='transformer',
    pretrained_model='bert-base-uncased'
)

# Train on text data
# text_classifier.fit(texts, labels)
```

## 6. Model Deployment

### Local Serving

```python
from mz_max.deployment import ModelServer

# Create model server
server = ModelServer(model=trained_model)

# Start serving
server.start(host='0.0.0.0', port=8080)
```

### Docker Deployment

```bash
# Build Docker image
docker build -t my-ml-app .

# Run container
docker run -p 8080:8080 my-ml-app
```

### Kubernetes Deployment

```python
from mz_max.deployment import KubernetesDeployer

# Deploy to Kubernetes
deployer = KubernetesDeployer()
deployment = deployer.deploy(
    model=trained_model,
    replicas=3,
    auto_scaling=True
)
```

## 7. Common Workflows

### Complete ML Pipeline

```python
import mz_max as mz

# 1. Load and explore data
data = mz.load_dataset('wine')
print(f"Dataset info: {data.info()}")

# 2. Data preprocessing
processed_data = mz.preprocess(
    data=data,
    target='target',
    test_size=0.2,
    random_state=42
)

# 3. Model training
automl = mz.AutoML(
    task_type='classification',
    optimization_metric='f1_score',
    time_budget=300  # 5 minutes
)

model = automl.fit(processed_data['X_train'], processed_data['y_train'])

# 4. Model evaluation
predictions = model.predict(processed_data['X_test'])
metrics = mz.evaluate_model(predictions, processed_data['y_test'])

print(f"Model Performance:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# 5. Model deployment
server = mz.ModelServer(model)
server.start()
```

### Hyperparameter Optimization

```python
from mz_max.optimization import HyperparameterOptimizer

# Define search space
search_space = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Optimize hyperparameters
optimizer = HyperparameterOptimizer(
    model_type='xgboost',
    search_space=search_space,
    n_trials=50
)

best_model = optimizer.optimize(X_train, y_train, X_val, y_val)
```

## 8. Tips and Best Practices

### Performance Optimization

```python
# Use parallel processing
automl = mz.AutoML(n_jobs=-1)  # Use all CPU cores

# Enable GPU acceleration (if available)
automl = mz.AutoML(use_gpu=True)

# Optimize memory usage for large datasets
automl = mz.AutoML(memory_limit='8GB')
```

### Data Quality

```python
from mz_max.data import DataValidator

# Validate data quality
validator = DataValidator()
report = validator.validate(data)

print(f"Data quality score: {report['quality_score']:.2f}")
print(f"Issues found: {len(report['issues'])}")
```

### Model Monitoring

```python
from mz_max.monitoring import ModelMonitor

# Monitor model performance
monitor = ModelMonitor(model=production_model)
drift_report = monitor.detect_drift(new_data)

if drift_report['drift_detected']:
    print("Data drift detected! Consider retraining the model.")
```

## Next Steps

Now that you've completed the quick start:

1. **Explore the UI** - Try all four professional interfaces
2. **Read Examples** - Check out [detailed examples](examples/)
3. **Learn Advanced Features** - Dive into [enterprise security](security.html)
4. **Deploy to Production** - Follow the [deployment guide](deployment.html)
5. **Join the Community** - Contribute on [GitHub](https://github.com/mzmax/mz-max)

## Getting Help

- **Documentation** - Browse the complete [API reference](api.html)
- **Examples** - See real-world [use cases](examples/)
- **Issues** - Report bugs on [GitHub Issues](https://github.com/mzmax/mz-max/issues)
- **Support** - Email us at elsamahy771@gmail.com

---

Ready to build something amazing? Start with the [User Interface Guide](interfaces.html) to explore all the professional tools at your disposal.
