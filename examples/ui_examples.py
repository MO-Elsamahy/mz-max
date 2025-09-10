"""
MZ Max Professional UI Examples

This module demonstrates how to use all the professional UI components
in MZ Max for different use cases and environments.
"""

def streamlit_dashboard_example():
    """Example of using the Streamlit dashboard."""
    print("""
🚀 MZ Max Streamlit Dashboard Example
=====================================

The professional Streamlit dashboard provides a comprehensive web interface
for machine learning workflows.

## Quick Start:
```python
# Method 1: Using the launcher
python launch_dashboard.py

# Method 2: Direct import
from mz_max.ui.dashboard import create_dashboard
create_dashboard()

# Method 3: Command line
mzmax-dashboard
```

## Features:
✅ Interactive data exploration
✅ Professional AutoML studio  
✅ Enterprise security center
✅ Real-time analytics
✅ Modern responsive UI
✅ Multi-dataset support
✅ Advanced visualizations

## Access:
🌐 Dashboard: http://localhost:8501
📊 Perfect for: Data scientists, analysts, demos
    """)


def fastapi_webapp_example():
    """Example of using the FastAPI web application."""
    print("""
🌐 MZ Max FastAPI Web Application Example
=========================================

The enterprise FastAPI application provides REST APIs and a modern web interface.

## Quick Start:
```python
# Method 1: Using the launcher
python launch_webapp.py

# Method 2: Direct import
from mz_max.ui.web_app import launch_web_app
launch_web_app(host="0.0.0.0", port=8000)

# Method 3: Command line
mzmax-webapp
```

## API Examples:

### Load Dataset:
```python
import requests

response = requests.post("http://localhost:8000/api/v1/data/load", 
                        json={"name": "iris"})
print(response.json())
```

### Make Prediction:
```python
prediction_data = {
    "data": [[5.1, 3.5, 1.4, 0.2]],
    "model_name": "iris_classifier"
}

response = requests.post("http://localhost:8000/api/v1/ml/predict",
                        json=prediction_data)
print(response.json())
```

### Encrypt Data:
```python
encrypt_data = {
    "data": {"customer_id": 12345, "score": 0.95}
}

response = requests.post("http://localhost:8000/api/v1/security/encrypt",
                        json=encrypt_data)
print(response.json())
```

## Access:
🔗 Web Interface: http://localhost:8000
📖 API Docs: http://localhost:8000/docs
📚 ReDoc: http://localhost:8000/redoc
🏢 Perfect for: Production deployments, enterprise integration
    """)


def desktop_gui_example():
    """Example of using the desktop GUI application."""
    print("""
🖥️ MZ Max Desktop GUI Example
=============================

The professional desktop application provides a native interface
with modern styling and comprehensive features.

## Quick Start:
```python
# Method 1: Using the launcher
python launch_gui.py

# Method 2: Direct import
from mz_max.ui.gui import launch_gui
launch_gui()

# Method 3: Command line
mzmax-gui
```

## Features:
✅ Multi-tab workflow organization
✅ Interactive data exploration
✅ Real-time ML training with progress
✅ Advanced data visualization
✅ Enterprise security center
✅ File drag & drop support
✅ Background task processing

## Usage Tips:
• Use the Data tab to load datasets or files
• Train models in the ML tab with real-time progress
• Encrypt sensitive data in the Security tab
• All operations run in background threads
• Professional color scheme and modern styling

💻 Perfect for: Desktop users, offline work, local development
    """)


def jupyter_widgets_example():
    """Example of using Jupyter widgets."""
    print("""
📱 MZ Max Jupyter Widgets Example
=================================

Interactive widgets for Jupyter notebooks provide a seamless
ML workflow within your notebook environment.

## Quick Start:
```python
# In a Jupyter notebook cell:
from mz_max.ui.widgets import (
    DataExplorationWidget, 
    AutoMLWidget,
    PredictionWidget,
    SecurityWidget,
    create_complete_workflow
)

# Method 1: Complete workflow
widgets = create_complete_workflow()

# Method 2: Individual widgets
data_widget = DataExplorationWidget()
data_widget.display()

automl_widget = AutoMLWidget()
automl_widget.display()

prediction_widget = PredictionWidget()
prediction_widget.display()

security_widget = SecurityWidget()
security_widget.display()
```

## Widget Features:

### Data Explorer:
• Load built-in datasets or upload files
• Interactive data visualization
• Statistical summaries
• Distribution plots, correlation heatmaps

### AutoML Studio:
• Automated model training
• Real-time progress tracking
• Model performance metrics
• Feature importance analysis

### Prediction Center:
• Make predictions with trained models
• Input validation and formatting
• Confidence scores
• Batch prediction support

### Security Center:
• Data encryption/decryption
• API key generation
• Secure data handling
• Enterprise-grade security

📚 Perfect for: Jupyter notebooks, research, interactive analysis
    """)


def integration_examples():
    """Examples of integrating UI components."""
    print("""
🔧 MZ Max UI Integration Examples
=================================

## Custom Streamlit App:
```python
import streamlit as st
from mz_max.data.loaders import load_dataset
from mz_max.enterprise.security import SecurityManager

st.title("My Custom ML App")

# Load data
dataset = st.selectbox("Dataset", ["iris", "wine", "diabetes"])
data = load_dataset(dataset)
st.dataframe(data.head())

# Security
security = SecurityManager()
sensitive_data = st.text_input("Sensitive data:")
if st.button("Encrypt"):
    encrypted = security.encrypt_data(sensitive_data)
    st.code(encrypted)
```

## Custom FastAPI Integration:
```python
from fastapi import FastAPI
from mz_max.ui.web_app import create_app

# Create custom app
app = FastAPI(title="My ML API")

# Add MZ Max routes
mz_app = create_app()
app.mount("/mzmax", mz_app)

# Add custom routes
@app.get("/custom")
def custom_endpoint():
    return {"message": "Custom endpoint"}
```

## Embedding in Existing Applications:
```python
# Django integration
from django.http import JsonResponse
from mz_max.data.loaders import load_dataset

def ml_view(request):
    data = load_dataset("iris")
    return JsonResponse({"shape": data.shape})

# Flask integration
from flask import Flask, jsonify
from mz_max.enterprise.security import SecurityManager

app = Flask(__name__)
security = SecurityManager()

@app.route("/encrypt", methods=["POST"])
def encrypt_data():
    data = request.json["data"]
    encrypted = security.encrypt_data(data)
    return jsonify({"encrypted": encrypted})
```

🚀 Perfect for: Custom applications, existing systems integration
    """)


def deployment_examples():
    """Examples of deploying UI components."""
    print("""
🚀 MZ Max UI Deployment Examples
================================

## Docker Deployment:
```dockerfile
FROM python:3.9-slim

COPY . /app
WORKDIR /app

RUN pip install -e .
RUN pip install streamlit fastapi uvicorn

# Streamlit
EXPOSE 8501
CMD ["python", "launch_dashboard.py"]

# Or FastAPI
EXPOSE 8000
CMD ["python", "launch_webapp.py"]
```

## Kubernetes Deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mzmax-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mzmax-dashboard
  template:
    metadata:
      labels:
        app: mzmax-dashboard
    spec:
      containers:
      - name: dashboard
        image: mzmax:latest
        ports:
        - containerPort: 8501
        command: ["python", "launch_dashboard.py"]
---
apiVersion: v1
kind: Service
metadata:
  name: mzmax-service
spec:
  selector:
    app: mzmax-dashboard
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

## Cloud Deployment:

### AWS (using ECS):
```python
# Task definition for AWS ECS
{
    "family": "mzmax-dashboard",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "containerDefinitions": [
        {
            "name": "dashboard",
            "image": "your-repo/mzmax:latest",
            "portMappings": [
                {
                    "containerPort": 8501,
                    "protocol": "tcp"
                }
            ],
            "command": ["python", "launch_dashboard.py"]
        }
    ]
}
```

### Google Cloud Run:
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/mzmax
gcloud run deploy --image gcr.io/PROJECT-ID/mzmax --port 8501
```

### Azure Container Instances:
```bash
# Deploy to Azure
az container create \
    --resource-group myResourceGroup \
    --name mzmax-dashboard \
    --image your-registry/mzmax:latest \
    --ports 8501 \
    --command-line "python launch_dashboard.py"
```

🌐 Perfect for: Production deployments, scalable applications
    """)


def main():
    """Run all UI examples."""
    print("🚀 MZ Max Professional UI Examples")
    print("=" * 50)
    
    streamlit_dashboard_example()
    fastapi_webapp_example()
    desktop_gui_example()
    jupyter_widgets_example()
    integration_examples()
    deployment_examples()
    
    print("""
🎉 Summary
==========

MZ Max provides four professional UI options:

1. 🌐 **Streamlit Dashboard** - Interactive web interface
2. 🏢 **FastAPI Web App** - Enterprise REST API + web UI  
3. 🖥️ **Desktop GUI** - Native application with modern styling
4. 📱 **Jupyter Widgets** - Interactive notebook components

Each interface provides:
✅ Professional design and user experience
✅ Enterprise-grade security features
✅ Real-time machine learning capabilities
✅ Comprehensive data exploration tools
✅ Modern responsive layouts
✅ Production-ready deployment options

Choose the interface that best fits your use case and environment!

🌟 Happy Machine Learning with MZ Max Professional! 🌟
    """)


if __name__ == "__main__":
    main()