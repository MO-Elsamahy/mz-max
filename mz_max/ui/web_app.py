"""
Professional FastAPI Web Application for MZ Max

This module provides a comprehensive, enterprise-grade REST API
and web interface for machine learning operations.
"""

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uvicorn
import asyncio
import json
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import tempfile
from pathlib import Path

# Import MZ Max components
try:
    from ..data.loaders import load_dataset
    from ..data.preprocessing import clean_data, scale_features
    from ..enterprise.security import SecurityManager
    from ..utils.logging import get_logger
    from ..utils.memory import get_memory_usage
    from ..core.exceptions import MLError
except ImportError as e:
    print(f"Warning: Could not import MZ Max components: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class DatasetRequest(BaseModel):
    name: str = Field(..., description="Dataset name to load")
    
class DatasetResponse(BaseModel):
    name: str
    shape: tuple
    columns: List[str]
    sample_data: Dict[str, Any]
    
class PredictionRequest(BaseModel):
    data: List[List[float]] = Field(..., description="Input data for prediction")
    model_name: Optional[str] = Field("default", description="Model name to use")
    
class PredictionResponse(BaseModel):
    predictions: List[float]
    confidence: Optional[List[float]] = None
    model_name: str
    timestamp: datetime
    
class EncryptionRequest(BaseModel):
    data: Union[str, Dict[str, Any]] = Field(..., description="Data to encrypt")
    
class EncryptionResponse(BaseModel):
    encrypted_data: str
    timestamp: datetime
    
class DecryptionRequest(BaseModel):
    encrypted_data: str = Field(..., description="Encrypted data to decrypt")
    
class DecryptionResponse(BaseModel):
    decrypted_data: Union[str, Dict[str, Any]]
    timestamp: datetime
    
class TrainingRequest(BaseModel):
    dataset_name: str
    model_type: str = Field(default="random_forest", description="Type of model to train")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    started_at: datetime
    
class SystemInfo(BaseModel):
    version: str
    uptime: str
    memory_usage: Dict[str, float]
    active_models: int
    total_requests: int
    
class APIKey(BaseModel):
    user_id: str
    permissions: List[str]
    
# Initialize FastAPI app
app = FastAPI(
    title="MZ Max Professional API",
    description="Enterprise-grade Machine Learning and Deep Learning API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

# Global variables
security_manager = None
request_count = 0
start_time = datetime.now()
active_jobs = {}

# Initialize security manager
try:
    security_manager = SecurityManager()
    logger.info("Security manager initialized")
except Exception as e:
    logger.warning(f"Security manager initialization failed: {e}")

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    # In production, validate the token here
    return {"user_id": "demo_user", "permissions": ["read", "write", "predict"]}

# HTML Templates
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MZ Max Professional API</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .card {
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.95);
            border: none;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .hero-section {
            padding: 4rem 0;
            text-align: center;
            color: white;
        }
        .feature-card {
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        .api-endpoint {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        .status-badge {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <div class="hero-section">
        <div class="container">
            <h1 class="display-4 fw-bold mb-4">
                <i class="fas fa-rocket me-3"></i>MZ Max Professional API
            </h1>
            <p class="lead mb-4">Enterprise-grade Machine Learning and Deep Learning Platform</p>
            <span class="status-badge">
                <i class="fas fa-circle me-2"></i>API Status: Online
            </span>
        </div>
    </div>

    <div class="container my-5">
        <div class="row">
            <div class="col-lg-4 mb-4">
                <div class="card feature-card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-database fa-3x text-primary mb-3"></i>
                        <h5 class="card-title">Data Management</h5>
                        <p class="card-text">Load, process, and analyze datasets with enterprise-grade tools.</p>
                        <a href="/docs#/default/load_dataset_api_v1_data_load_post" class="btn btn-primary">Try Now</a>
                    </div>
                </div>
            </div>
            
            <div class="col-lg-4 mb-4">
                <div class="card feature-card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-brain fa-3x text-success mb-3"></i>
                        <h5 class="card-title">Machine Learning</h5>
                        <p class="card-text">Train and deploy ML models with automated optimization.</p>
                        <a href="/docs#/default/train_model_api_v1_ml_train_post" class="btn btn-success">Explore</a>
                    </div>
                </div>
            </div>
            
            <div class="col-lg-4 mb-4">
                <div class="card feature-card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-shield-alt fa-3x text-warning mb-3"></i>
                        <h5 class="card-title">Enterprise Security</h5>
                        <p class="card-text">Military-grade encryption and security features.</p>
                        <a href="/docs#/default/encrypt_data_api_v1_security_encrypt_post" class="btn btn-warning">Secure</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-5">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-code me-2"></i>Quick Start API Examples</h5>
                    </div>
                    <div class="card-body">
                        <div class="api-endpoint">
                            <strong>GET /api/v1/info</strong> - Get system information
                        </div>
                        <div class="api-endpoint">
                            <strong>POST /api/v1/data/load</strong> - Load dataset
                        </div>
                        <div class="api-endpoint">
                            <strong>POST /api/v1/ml/predict</strong> - Make predictions
                        </div>
                        <div class="api-endpoint">
                            <strong>POST /api/v1/security/encrypt</strong> - Encrypt sensitive data
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title"><i class="fas fa-chart-line me-2"></i>System Metrics</h6>
                        <ul class="list-unstyled">
                            <li><strong>Version:</strong> 1.0.0</li>
                            <li><strong>Uptime:</strong> <span id="uptime">Loading...</span></li>
                            <li><strong>Requests:</strong> <span id="requests">0</span></li>
                            <li><strong>Memory:</strong> <span id="memory">Loading...</span></li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title"><i class="fas fa-link me-2"></i>Quick Links</h6>
                        <div class="d-grid gap-2">
                            <a href="/docs" class="btn btn-outline-primary">
                                <i class="fas fa-book me-2"></i>API Documentation
                            </a>
                            <a href="/redoc" class="btn btn-outline-info">
                                <i class="fas fa-file-alt me-2"></i>ReDoc Documentation
                            </a>
                            <a href="/api/v1/health" class="btn btn-outline-success">
                                <i class="fas fa-heartbeat me-2"></i>Health Check
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="text-center text-white py-4 mt-5">
        <div class="container">
            <p class="mb-0">
                <i class="fas fa-star me-2"></i>
                <strong>MZ Max</strong> - The Ultimate Machine Learning Platform
                <i class="fas fa-star ms-2"></i>
            </p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Update system metrics
        async function updateMetrics() {
            try {
                const response = await fetch('/api/v1/info');
                const data = await response.json();
                
                document.getElementById('uptime').textContent = data.uptime;
                document.getElementById('requests').textContent = data.total_requests;
                document.getElementById('memory').textContent = 
                    Math.round(data.memory_usage.rss_mb) + ' MB';
            } catch (error) {
                console.error('Failed to update metrics:', error);
            }
        }
        
        // Update metrics every 30 seconds
        updateMetrics();
        setInterval(updateMetrics, 30000);
    </script>
</body>
</html>
"""

# API Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with professional web interface."""
    global request_count
    request_count += 1
    return html_template

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    global request_count
    request_count += 1
    
    memory_info = get_memory_usage()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "memory_usage": memory_info,
        "uptime": str(datetime.now() - start_time)
    }

@app.get("/api/v1/info", response_model=SystemInfo)
async def get_system_info():
    """Get comprehensive system information."""
    global request_count
    request_count += 1
    
    memory_info = get_memory_usage()
    uptime = datetime.now() - start_time
    
    return SystemInfo(
        version="1.0.0",
        uptime=str(uptime),
        memory_usage=memory_info,
        active_models=len(active_jobs),
        total_requests=request_count
    )

@app.post("/api/v1/data/load", response_model=DatasetResponse)
async def load_dataset_api(request: DatasetRequest):
    """Load a dataset and return information about it."""
    global request_count
    request_count += 1
    
    try:
        data = load_dataset(request.name)
        
        return DatasetResponse(
            name=request.name,
            shape=data.shape,
            columns=data.columns.tolist(),
            sample_data=data.head(3).to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")

@app.post("/api/v1/ml/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, user = Depends(get_current_user)):
    """Make predictions using a trained model."""
    global request_count
    request_count += 1
    
    try:
        # Simulate prediction (in real implementation, load actual model)
        input_data = np.array(request.data)
        
        # Mock predictions
        predictions = np.random.random(len(input_data)).tolist()
        confidence = np.random.uniform(0.7, 0.99, len(input_data)).tolist()
        
        return PredictionResponse(
            predictions=predictions,
            confidence=confidence,
            model_name=request.model_name,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/api/v1/ml/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training in the background."""
    global request_count
    request_count += 1
    
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_count}"
    
    # Add background training task
    background_tasks.add_task(simulate_training, job_id, request)
    
    active_jobs[job_id] = {
        "status": "started",
        "request": request,
        "started_at": datetime.now()
    }
    
    return TrainingResponse(
        job_id=job_id,
        status="started",
        message="Model training initiated",
        started_at=datetime.now()
    )

@app.get("/api/v1/ml/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a training job."""
    global request_count
    request_count += 1
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@app.post("/api/v1/security/encrypt", response_model=EncryptionResponse)
async def encrypt_data(request: EncryptionRequest):
    """Encrypt sensitive data."""
    global request_count
    request_count += 1
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        encrypted = security_manager.encrypt_data(request.data)
        
        return EncryptionResponse(
            encrypted_data=encrypted,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Encryption failed: {str(e)}")

@app.post("/api/v1/security/decrypt", response_model=DecryptionResponse)
async def decrypt_data(request: DecryptionRequest):
    """Decrypt encrypted data."""
    global request_count
    request_count += 1
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        decrypted = security_manager.decrypt_data(request.encrypted_data)
        
        return DecryptionResponse(
            decrypted_data=decrypted,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")

@app.post("/api/v1/security/api-key")
async def generate_api_key(request: APIKey):
    """Generate a new API key."""
    global request_count
    request_count += 1
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not available")
    
    try:
        api_key_info = security_manager.generate_api_key(request.user_id, request.permissions)
        return api_key_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"API key generation failed: {str(e)}")

@app.get("/api/v1/data/datasets")
async def list_datasets():
    """List available datasets."""
    global request_count
    request_count += 1
    
    return {
        "datasets": [
            {"name": "iris", "description": "Iris flower classification dataset"},
            {"name": "wine", "description": "Wine classification dataset"},
            {"name": "diabetes", "description": "Diabetes regression dataset"},
            {"name": "breast_cancer", "description": "Breast cancer classification dataset"},
            {"name": "digits", "description": "Handwritten digits classification dataset"}
        ]
    }

@app.post("/api/v1/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload and process a data file."""
    global request_count
    request_count += 1
    
    if not file.filename.endswith(('.csv', '.json', '.xlsx')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process file based on extension
        if file.filename.endswith('.csv'):
            data = pd.read_csv(tmp_file_path)
        elif file.filename.endswith('.json'):
            data = pd.read_json(tmp_file_path)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "filename": file.filename,
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "sample": data.head(3).to_dict(),
            "message": "File uploaded and processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed: {str(e)}")

# Background tasks
async def simulate_training(job_id: str, request: TrainingRequest):
    """Simulate model training process."""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "training"
        
        # Simulate training time
        await asyncio.sleep(10)  # 10 seconds for demo
        
        # Complete training
        active_jobs[job_id].update({
            "status": "completed",
            "accuracy": 0.945,
            "completed_at": datetime.now(),
            "model_path": f"models/{job_id}.pkl"
        })
        
        logger.info(f"Training job {job_id} completed")
        
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now()
        })
        logger.error(f"Training job {job_id} failed: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Application factory
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app

def launch_web_app(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Launch the web application."""
    logger.info(f"🚀 Starting MZ Max Professional API on {host}:{port}")
    logger.info(f"📖 API Documentation: http://{host}:{port}/docs")
    logger.info(f"🌐 Web Interface: http://{host}:{port}")
    
    uvicorn.run(app, host=host, port=port, reload=reload)

if __name__ == "__main__":
    launch_web_app()