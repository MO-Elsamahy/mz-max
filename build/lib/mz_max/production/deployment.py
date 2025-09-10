"""
Production Deployment for MZ Max

This module provides comprehensive deployment solutions for
machine learning models in production environments.
"""

import os
import json
import yaml
import docker
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import shutil

from ..utils.logging import get_logger
from ..core.exceptions import DeploymentError

logger = get_logger(__name__)


class ProductionDeployment:
    """
    Production deployment manager for ML models.
    """
    
    def __init__(self, deployment_config: Optional[Dict[str, Any]] = None):
        """
        Initialize production deployment.
        
        Args:
            deployment_config: Deployment configuration
        """
        self.config = deployment_config or self._default_config()
        self.deployment_history = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration."""
        return {
            'environment': 'production',
            'scaling': {
                'min_replicas': 2,
                'max_replicas': 10,
                'target_cpu_utilization': 70
            },
            'resources': {
                'cpu_request': '500m',
                'cpu_limit': '2000m',
                'memory_request': '1Gi',
                'memory_limit': '4Gi'
            },
            'health_check': {
                'enabled': True,
                'path': '/health',
                'initial_delay': 30,
                'timeout': 10,
                'period': 30
            },
            'monitoring': {
                'metrics_enabled': True,
                'logging_level': 'INFO',
                'alerts_enabled': True
            }
        }
    
    def deploy_model(self, model, model_name: str, version: str = "v1") -> Dict[str, Any]:
        """
        Deploy model to production.
        
        Args:
            model: Trained model to deploy
            model_name: Name for the deployed model
            version: Model version
            
        Returns:
            Deployment information
        """
        try:
            logger.info(f"Starting deployment of {model_name} {version}")
            
            # Create deployment package
            deployment_package = self._create_deployment_package(model, model_name, version)
            
            # Generate deployment manifests
            manifests = self._generate_manifests(model_name, version, deployment_package)
            
            # Deploy to target environment
            deployment_result = self._deploy_to_environment(manifests, model_name, version)
            
            # Record deployment
            deployment_record = {
                'model_name': model_name,
                'version': version,
                'deployed_at': datetime.utcnow().isoformat(),
                'deployment_id': deployment_result['deployment_id'],
                'status': 'deployed',
                'endpoints': deployment_result['endpoints'],
                'config': self.config.copy()
            }
            
            self.deployment_history.append(deployment_record)
            
            logger.info(f"Successfully deployed {model_name} {version}")
            return deployment_record
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise DeploymentError(f"Failed to deploy {model_name}: {str(e)}")
    
    def _create_deployment_package(self, model, model_name: str, version: str) -> Dict[str, Any]:
        """Create deployment package with model and dependencies."""
        package_dir = Path(tempfile.mkdtemp()) / f"{model_name}-{version}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            model_path = package_dir / "model.pkl"
            model.save(str(model_path))
            
            # Create requirements.txt
            requirements_path = package_dir / "requirements.txt"
            requirements = self._generate_requirements()
            requirements_path.write_text(requirements)
            
            # Create serving application
            app_path = package_dir / "app.py"
            app_code = self._generate_serving_app(model_name)
            app_path.write_text(app_code)
            
            # Create Dockerfile
            dockerfile_path = package_dir / "Dockerfile"
            dockerfile_content = self._generate_dockerfile()
            dockerfile_path.write_text(dockerfile_content)
            
            # Create configuration files
            config_path = package_dir / "config.json"
            config_path.write_text(json.dumps(self.config, indent=2))
            
            return {
                'package_dir': str(package_dir),
                'model_path': str(model_path),
                'app_path': str(app_path),
                'dockerfile_path': str(dockerfile_path),
                'requirements_path': str(requirements_path)
            }
            
        except Exception as e:
            # Cleanup on failure
            shutil.rmtree(package_dir, ignore_errors=True)
            raise e
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt for deployment."""
        return """
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.3.0
torch>=2.0.0
joblib>=1.3.0
prometheus-client>=0.16.0
psutil>=5.9.0
        """.strip()
    
    def _generate_serving_app(self, model_name: str) -> str:
        """Generate FastAPI serving application."""
        return f'''
import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from prometheus_client import Counter, Histogram, generate_latest
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('prediction_errors_total', 'Total prediction errors')

# Load model
model = joblib.load('/app/model.pkl')
config = json.load(open('/app/config.json'))

app = FastAPI(title="{model_name} API", version="1.0.0")

class PredictionRequest(BaseModel):
    data: List[List[float]]
    
class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    timestamp: str

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {{"status": "healthy", "model": "{model_name}", "timestamp": time.time()}}

@app.get("/info")
async def model_info():
    """Model information endpoint."""
    return {{
        "model_name": "{model_name}",
        "model_type": type(model).__name__,
        "config": config,
        "timestamp": time.time()
    }}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions."""
    start_time = time.time()
    
    try:
        # Convert input to numpy array
        input_data = np.array(request.data)
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Convert to list for JSON serialization
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        elif not isinstance(predictions, list):
            predictions = [predictions]
        
        # Record metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return PredictionResponse(
            predictions=predictions,
            model_version="v1",
            timestamp=str(time.time())
        )
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        '''.strip()
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for model serving."""
        return '''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
        '''.strip()
    
    def _generate_manifests(self, model_name: str, version: str, 
                          deployment_package: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifests."""
        
        # Kubernetes Deployment
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'{model_name}-{version}',
                'labels': {
                    'app': model_name,
                    'version': version
                }
            },
            'spec': {
                'replicas': self.config['scaling']['min_replicas'],
                'selector': {
                    'matchLabels': {
                        'app': model_name,
                        'version': version
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': model_name,
                            'version': version
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': model_name,
                            'image': f'{model_name}:{version}',
                            'ports': [{'containerPort': 8000}],
                            'resources': {
                                'requests': {
                                    'cpu': self.config['resources']['cpu_request'],
                                    'memory': self.config['resources']['memory_request']
                                },
                                'limits': {
                                    'cpu': self.config['resources']['cpu_limit'],
                                    'memory': self.config['resources']['memory_limit']
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': self.config['health_check']['initial_delay'],
                                'timeoutSeconds': self.config['health_check']['timeout'],
                                'periodSeconds': self.config['health_check']['period']
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 10,
                                'timeoutSeconds': 5,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        }
        
        # Kubernetes Service
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{model_name}-service',
                'labels': {
                    'app': model_name
                }
            },
            'spec': {
                'selector': {
                    'app': model_name
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8000,
                    'protocol': 'TCP'
                }],
                'type': 'ClusterIP'
            }
        }
        
        # Horizontal Pod Autoscaler
        hpa_manifest = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f'{model_name}-hpa'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': f'{model_name}-{version}'
                },
                'minReplicas': self.config['scaling']['min_replicas'],
                'maxReplicas': self.config['scaling']['max_replicas'],
                'metrics': [{
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': self.config['scaling']['target_cpu_utilization']
                        }
                    }
                }]
            }
        }
        
        return {
            'deployment': deployment_manifest,
            'service': service_manifest,
            'hpa': hpa_manifest,
            'package': deployment_package
        }
    
    def _deploy_to_environment(self, manifests: Dict[str, Any], 
                             model_name: str, version: str) -> Dict[str, Any]:
        """Deploy manifests to target environment."""
        
        # Build Docker image
        image_tag = self._build_docker_image(manifests['package'], model_name, version)
        
        # Apply Kubernetes manifests
        deployment_id = self._apply_kubernetes_manifests(manifests, model_name, version)
        
        # Get service endpoints
        endpoints = self._get_service_endpoints(model_name)
        
        return {
            'deployment_id': deployment_id,
            'image_tag': image_tag,
            'endpoints': endpoints,
            'status': 'deployed'
        }
    
    def _build_docker_image(self, package: Dict[str, Any], 
                           model_name: str, version: str) -> str:
        """Build Docker image for the model."""
        try:
            client = docker.from_env()
            
            image_tag = f"{model_name}:{version}"
            package_dir = package['package_dir']
            
            logger.info(f"Building Docker image: {image_tag}")
            
            # Build image
            image, build_logs = client.images.build(
                path=package_dir,
                tag=image_tag,
                rm=True,
                forcerm=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
            
            logger.info(f"Successfully built image: {image_tag}")
            return image_tag
            
        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            raise DeploymentError(f"Failed to build Docker image: {str(e)}")
    
    def _apply_kubernetes_manifests(self, manifests: Dict[str, Any], 
                                   model_name: str, version: str) -> str:
        """Apply Kubernetes manifests."""
        try:
            # Create temporary manifest files
            manifest_dir = Path(tempfile.mkdtemp())
            
            # Write manifests to files
            deployment_file = manifest_dir / 'deployment.yaml'
            service_file = manifest_dir / 'service.yaml'
            hpa_file = manifest_dir / 'hpa.yaml'
            
            with open(deployment_file, 'w') as f:
                yaml.dump(manifests['deployment'], f)
            
            with open(service_file, 'w') as f:
                yaml.dump(manifests['service'], f)
            
            with open(hpa_file, 'w') as f:
                yaml.dump(manifests['hpa'], f)
            
            # Apply manifests using kubectl
            for manifest_file in [deployment_file, service_file, hpa_file]:
                result = subprocess.run(
                    ['kubectl', 'apply', '-f', str(manifest_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Applied manifest: {manifest_file.name}")
                logger.debug(result.stdout)
            
            # Cleanup
            shutil.rmtree(manifest_dir, ignore_errors=True)
            
            deployment_id = f"{model_name}-{version}-{int(datetime.utcnow().timestamp())}"
            return deployment_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"kubectl apply failed: {e.stderr}")
            raise DeploymentError(f"Failed to apply Kubernetes manifests: {e.stderr}")
        except Exception as e:
            logger.error(f"Manifest application failed: {e}")
            raise DeploymentError(f"Failed to apply manifests: {str(e)}")
    
    def _get_service_endpoints(self, model_name: str) -> List[str]:
        """Get service endpoints."""
        try:
            # Get service information using kubectl
            result = subprocess.run(
                ['kubectl', 'get', 'service', f'{model_name}-service', '-o', 'json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            service_info = json.loads(result.stdout)
            
            # Extract endpoints
            endpoints = []
            if 'status' in service_info and 'loadBalancer' in service_info['status']:
                ingress = service_info['status']['loadBalancer'].get('ingress', [])
                for ing in ingress:
                    if 'ip' in ing:
                        endpoints.append(f"http://{ing['ip']}")
                    elif 'hostname' in ing:
                        endpoints.append(f"http://{ing['hostname']}")
            
            # Fallback to cluster IP
            if not endpoints:
                cluster_ip = service_info['spec'].get('clusterIP')
                if cluster_ip:
                    endpoints.append(f"http://{cluster_ip}")
            
            return endpoints
            
        except Exception as e:
            logger.warning(f"Failed to get service endpoints: {e}")
            return []
    
    def rollback_deployment(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Rollback deployment to previous version.
        
        Args:
            model_name: Model name
            version: Version to rollback to
            
        Returns:
            Rollback information
        """
        try:
            logger.info(f"Rolling back {model_name} to {version}")
            
            # Use kubectl rollout undo
            result = subprocess.run(
                ['kubectl', 'rollout', 'undo', f'deployment/{model_name}-{version}'],
                capture_output=True,
                text=True,
                check=True
            )
            
            rollback_info = {
                'model_name': model_name,
                'version': version,
                'rolled_back_at': datetime.utcnow().isoformat(),
                'status': 'rolled_back',
                'details': result.stdout
            }
            
            logger.info(f"Successfully rolled back {model_name}")
            return rollback_info
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise DeploymentError(f"Failed to rollback {model_name}: {str(e)}")
    
    def get_deployment_status(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Get deployment status.
        
        Args:
            model_name: Model name
            version: Model version
            
        Returns:
            Deployment status information
        """
        try:
            # Get deployment status
            result = subprocess.run(
                ['kubectl', 'get', 'deployment', f'{model_name}-{version}', '-o', 'json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            deployment_info = json.loads(result.stdout)
            
            status = {
                'name': model_name,
                'version': version,
                'replicas': deployment_info['spec']['replicas'],
                'available_replicas': deployment_info['status'].get('availableReplicas', 0),
                'ready_replicas': deployment_info['status'].get('readyReplicas', 0),
                'updated_replicas': deployment_info['status'].get('updatedReplicas', 0),
                'conditions': deployment_info['status'].get('conditions', []),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def delete_deployment(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Delete deployment.
        
        Args:
            model_name: Model name
            version: Model version
            
        Returns:
            Deletion information
        """
        try:
            logger.info(f"Deleting deployment {model_name}-{version}")
            
            # Delete resources
            resources = [
                f'deployment/{model_name}-{version}',
                f'service/{model_name}-service',
                f'hpa/{model_name}-hpa'
            ]
            
            for resource in resources:
                try:
                    subprocess.run(
                        ['kubectl', 'delete', resource],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    logger.info(f"Deleted {resource}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to delete {resource}: {e.stderr}")
            
            deletion_info = {
                'model_name': model_name,
                'version': version,
                'deleted_at': datetime.utcnow().isoformat(),
                'status': 'deleted'
            }
            
            return deletion_info
            
        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            raise DeploymentError(f"Failed to delete deployment: {str(e)}")


class ContainerDeployment:
    """Container-based deployment using Docker."""
    
    def __init__(self):
        """Initialize container deployment."""
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            raise DeploymentError(f"Failed to connect to Docker: {str(e)}")
    
    def deploy_container(self, model, model_name: str, port: int = 8000) -> Dict[str, Any]:
        """
        Deploy model as Docker container.
        
        Args:
            model: Trained model
            model_name: Model name
            port: Port to expose
            
        Returns:
            Container deployment information
        """
        try:
            # Create deployment package
            deployment = ProductionDeployment()
            package = deployment._create_deployment_package(model, model_name, "latest")
            
            # Build image
            image_tag = deployment._build_docker_image(package, model_name, "latest")
            
            # Run container
            container = self.docker_client.containers.run(
                image_tag,
                ports={'8000/tcp': port},
                detach=True,
                name=f"{model_name}-container",
                restart_policy={"Name": "unless-stopped"}
            )
            
            container_info = {
                'container_id': container.id,
                'container_name': container.name,
                'image_tag': image_tag,
                'port': port,
                'status': 'running',
                'endpoint': f"http://localhost:{port}",
                'deployed_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Container deployed: {container_info}")
            return container_info
            
        except Exception as e:
            logger.error(f"Container deployment failed: {e}")
            raise DeploymentError(f"Failed to deploy container: {str(e)}")
    
    def stop_container(self, container_name: str) -> Dict[str, Any]:
        """Stop running container."""
        try:
            container = self.docker_client.containers.get(container_name)
            container.stop()
            container.remove()
            
            return {
                'container_name': container_name,
                'status': 'stopped',
                'stopped_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            raise DeploymentError(f"Failed to stop container: {str(e)}")


class CloudDeployment:
    """Cloud platform deployment (AWS, GCP, Azure)."""
    
    def __init__(self, cloud_provider: str = 'aws', credentials: Optional[Dict] = None):
        """
        Initialize cloud deployment.
        
        Args:
            cloud_provider: Cloud provider ('aws', 'gcp', 'azure')
            credentials: Cloud credentials
        """
        self.cloud_provider = cloud_provider.lower()
        self.credentials = credentials or {}
        
    def deploy_to_cloud(self, model, model_name: str, 
                       cloud_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Deploy model to cloud platform.
        
        Args:
            model: Trained model
            model_name: Model name
            cloud_config: Cloud-specific configuration
            
        Returns:
            Cloud deployment information
        """
        if self.cloud_provider == 'aws':
            return self._deploy_to_aws(model, model_name, cloud_config)
        elif self.cloud_provider == 'gcp':
            return self._deploy_to_gcp(model, model_name, cloud_config)
        elif self.cloud_provider == 'azure':
            return self._deploy_to_azure(model, model_name, cloud_config)
        else:
            raise DeploymentError(f"Unsupported cloud provider: {self.cloud_provider}")
    
    def _deploy_to_aws(self, model, model_name: str, config: Optional[Dict]) -> Dict[str, Any]:
        """Deploy to AWS using SageMaker or ECS."""
        # This would integrate with boto3 for AWS deployment
        logger.info(f"Deploying {model_name} to AWS")
        
        # Placeholder for AWS deployment logic
        return {
            'provider': 'aws',
            'service': 'sagemaker',
            'endpoint': f"https://{model_name}.aws.com",
            'model_name': model_name,
            'deployed_at': datetime.utcnow().isoformat(),
            'status': 'deployed'
        }
    
    def _deploy_to_gcp(self, model, model_name: str, config: Optional[Dict]) -> Dict[str, Any]:
        """Deploy to Google Cloud Platform."""
        # This would integrate with google-cloud libraries
        logger.info(f"Deploying {model_name} to GCP")
        
        return {
            'provider': 'gcp',
            'service': 'ai-platform',
            'endpoint': f"https://{model_name}.gcp.com",
            'model_name': model_name,
            'deployed_at': datetime.utcnow().isoformat(),
            'status': 'deployed'
        }
    
    def _deploy_to_azure(self, model, model_name: str, config: Optional[Dict]) -> Dict[str, Any]:
        """Deploy to Microsoft Azure."""
        # This would integrate with azure-ml-sdk
        logger.info(f"Deploying {model_name} to Azure")
        
        return {
            'provider': 'azure',
            'service': 'ml-service',
            'endpoint': f"https://{model_name}.azure.com",
            'model_name': model_name,
            'deployed_at': datetime.utcnow().isoformat(),
            'status': 'deployed'
        }
