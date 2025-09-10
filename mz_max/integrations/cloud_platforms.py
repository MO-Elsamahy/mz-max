"""
Cloud Platform Integrations for MZ Max

This module provides seamless integration with major cloud platforms
including AWS, Google Cloud Platform, and Microsoft Azure.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage, bigquery, aiplatform
    from google.oauth2 import service_account
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from ..utils.logging import get_logger
from ..core.exceptions import IntegrationError

logger = get_logger(__name__)


class AWSIntegration:
    """
    Comprehensive AWS integration for MZ Max.
    
    Provides integration with S3, SageMaker, Lambda, ECR, and other AWS services.
    """
    
    def __init__(self, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'us-east-1'):
        """
        Initialize AWS integration.
        
        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region name
        """
        if not AWS_AVAILABLE:
            raise IntegrationError("AWS SDK not available. Install with: pip install boto3")
        
        self.region_name = region_name
        
        # Initialize AWS session
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # Use default credentials (IAM role, environment variables, etc.)
                self.session = boto3.Session(region_name=region_name)
            
            # Initialize service clients
            self.s3_client = self.session.client('s3')
            self.sagemaker_client = self.session.client('sagemaker')
            self.lambda_client = self.session.client('lambda')
            self.ecr_client = self.session.client('ecr')
            self.cloudwatch_client = self.session.client('cloudwatch')
            
            logger.info("AWS integration initialized successfully")
            
        except NoCredentialsError:
            raise IntegrationError("AWS credentials not found. Please configure AWS credentials.")
        except Exception as e:
            raise IntegrationError(f"Failed to initialize AWS integration: {str(e)}")
    
    def upload_model_to_s3(self, model_path: str, bucket_name: str, 
                          s3_key: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Upload model to S3.
        
        Args:
            model_path: Local path to model file
            bucket_name: S3 bucket name
            s3_key: S3 object key
            metadata: Optional metadata to attach
            
        Returns:
            Upload information
        """
        try:
            # Prepare metadata
            upload_metadata = {
                'uploaded_at': datetime.utcnow().isoformat(),
                'model_framework': 'mz_max',
                'file_size': str(Path(model_path).stat().st_size)
            }
            
            if metadata:
                upload_metadata.update(metadata)
            
            # Upload file
            self.s3_client.upload_file(
                model_path,
                bucket_name,
                s3_key,
                ExtraArgs={'Metadata': upload_metadata}
            )
            
            # Generate S3 URL
            s3_url = f"s3://{bucket_name}/{s3_key}"
            
            upload_info = {
                'bucket': bucket_name,
                'key': s3_key,
                'url': s3_url,
                'metadata': upload_metadata,
                'uploaded_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Model uploaded to S3: {s3_url}")
            return upload_info
            
        except ClientError as e:
            raise IntegrationError(f"Failed to upload model to S3: {str(e)}")
    
    def deploy_to_sagemaker(self, model_s3_path: str, model_name: str,
                           instance_type: str = 'ml.m5.large',
                           initial_instance_count: int = 1) -> Dict[str, Any]:
        """
        Deploy model to Amazon SageMaker.
        
        Args:
            model_s3_path: S3 path to model artifacts
            model_name: SageMaker model name
            instance_type: EC2 instance type for hosting
            initial_instance_count: Initial number of instances
            
        Returns:
            Deployment information
        """
        try:
            # Create SageMaker model
            model_response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': self._get_sagemaker_image(),
                    'ModelDataUrl': model_s3_path,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                    }
                },
                ExecutionRoleArn=self._get_sagemaker_role()
            )
            
            # Create endpoint configuration
            endpoint_config_name = f"{model_name}-config"
            config_response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': initial_instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1
                }]
            )
            
            # Create endpoint
            endpoint_name = f"{model_name}-endpoint"
            endpoint_response = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            deployment_info = {
                'model_name': model_name,
                'endpoint_name': endpoint_name,
                'endpoint_config_name': endpoint_config_name,
                'instance_type': instance_type,
                'initial_instance_count': initial_instance_count,
                'status': 'Creating',
                'deployed_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"SageMaker deployment initiated: {endpoint_name}")
            return deployment_info
            
        except ClientError as e:
            raise IntegrationError(f"Failed to deploy to SageMaker: {str(e)}")
    
    def create_lambda_function(self, function_name: str, model_s3_path: str,
                              runtime: str = 'python3.9') -> Dict[str, Any]:
        """
        Create AWS Lambda function for model inference.
        
        Args:
            function_name: Lambda function name
            model_s3_path: S3 path to model
            runtime: Lambda runtime
            
        Returns:
            Lambda function information
        """
        try:
            # Create deployment package
            deployment_package = self._create_lambda_package(model_s3_path)
            
            # Create Lambda function
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime=runtime,
                Role=self._get_lambda_role(),
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': deployment_package},
                Environment={
                    'Variables': {
                        'MODEL_S3_PATH': model_s3_path
                    }
                },
                Timeout=300,
                MemorySize=1024
            )
            
            function_info = {
                'function_name': function_name,
                'function_arn': response['FunctionArn'],
                'runtime': runtime,
                'model_s3_path': model_s3_path,
                'created_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Lambda function created: {function_name}")
            return function_info
            
        except ClientError as e:
            raise IntegrationError(f"Failed to create Lambda function: {str(e)}")
    
    def setup_cloudwatch_monitoring(self, resource_name: str, 
                                   metric_filters: List[Dict] = None) -> Dict[str, Any]:
        """
        Set up CloudWatch monitoring for ML resources.
        
        Args:
            resource_name: Name of resource to monitor
            metric_filters: Custom metric filters
            
        Returns:
            Monitoring setup information
        """
        try:
            # Create CloudWatch dashboard
            dashboard_name = f"{resource_name}-ml-dashboard"
            
            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "properties": {
                            "metrics": [
                                ["AWS/SageMaker", "Invocations", "EndpointName", resource_name],
                                ["AWS/SageMaker", "ModelLatency", "EndpointName", resource_name],
                                ["AWS/SageMaker", "InvocationsPerInstance", "EndpointName", resource_name]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.region_name,
                            "title": "Model Performance Metrics"
                        }
                    }
                ]
            }
            
            self.cloudwatch_client.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            # Set up alarms
            alarm_name = f"{resource_name}-high-latency"
            self.cloudwatch_client.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=2,
                MetricName='ModelLatency',
                Namespace='AWS/SageMaker',
                Period=300,
                Statistic='Average',
                Threshold=5000.0,  # 5 seconds
                ActionsEnabled=True,
                AlarmDescription='Alert when model latency is high',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': resource_name
                    }
                ]
            )
            
            monitoring_info = {
                'dashboard_name': dashboard_name,
                'alarm_name': alarm_name,
                'resource_name': resource_name,
                'setup_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"CloudWatch monitoring setup for: {resource_name}")
            return monitoring_info
            
        except ClientError as e:
            raise IntegrationError(f"Failed to setup CloudWatch monitoring: {str(e)}")
    
    def _get_sagemaker_image(self) -> str:
        """Get appropriate SageMaker container image."""
        # Return the appropriate SageMaker container image for the framework
        account_id = self.session.client('sts').get_caller_identity()['Account']
        return f"{account_id}.dkr.ecr.{self.region_name}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
    
    def _get_sagemaker_role(self) -> str:
        """Get SageMaker execution role ARN."""
        # This should be configured based on your AWS setup
        account_id = self.session.client('sts').get_caller_identity()['Account']
        return f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    def _get_lambda_role(self) -> str:
        """Get Lambda execution role ARN."""
        account_id = self.session.client('sts').get_caller_identity()['Account']
        return f"arn:aws:iam::{account_id}:role/LambdaExecutionRole"
    
    def _create_lambda_package(self, model_s3_path: str) -> bytes:
        """Create Lambda deployment package."""
        # This would create a zip file with the Lambda function code
        # For now, return a placeholder
        lambda_code = f'''
import json
import boto3
import joblib
from io import BytesIO

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        # Load model from S3
        model_bucket, model_key = "{model_s3_path}".replace("s3://", "").split("/", 1)
        
        # Download model
        model_obj = s3_client.get_object(Bucket=model_bucket, Key=model_key)
        model = joblib.load(BytesIO(model_obj['Body'].read()))
        
        # Get input data
        input_data = json.loads(event['body'])
        
        # Make prediction
        prediction = model.predict([input_data['features']])
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{
                'prediction': prediction.tolist()
            }})
        }}
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps({{
                'error': str(e)
            }})
        }}
        '''
        
        return lambda_code.encode('utf-8')


class GCPIntegration:
    """
    Google Cloud Platform integration for MZ Max.
    
    Provides integration with GCS, AI Platform, BigQuery, and other GCP services.
    """
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """
        Initialize GCP integration.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to service account credentials JSON
        """
        if not GCP_AVAILABLE:
            raise IntegrationError("GCP SDK not available. Install with: pip install google-cloud-storage google-cloud-aiplatform")
        
        self.project_id = project_id
        
        try:
            # Initialize credentials
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
            else:
                # Use default credentials
                credentials = None
            
            # Initialize service clients
            self.storage_client = storage.Client(project=project_id, credentials=credentials)
            self.bigquery_client = bigquery.Client(project=project_id, credentials=credentials)
            
            # Initialize AI Platform
            aiplatform.init(project=project_id, credentials=credentials)
            
            logger.info("GCP integration initialized successfully")
            
        except Exception as e:
            raise IntegrationError(f"Failed to initialize GCP integration: {str(e)}")
    
    def upload_model_to_gcs(self, model_path: str, bucket_name: str, 
                           blob_name: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Upload model to Google Cloud Storage.
        
        Args:
            model_path: Local path to model file
            bucket_name: GCS bucket name
            blob_name: GCS blob name
            metadata: Optional metadata
            
        Returns:
            Upload information
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Set metadata
            if metadata:
                blob.metadata = metadata
            
            # Upload file
            blob.upload_from_filename(model_path)
            
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            
            upload_info = {
                'bucket': bucket_name,
                'blob': blob_name,
                'uri': gcs_uri,
                'metadata': metadata,
                'uploaded_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Model uploaded to GCS: {gcs_uri}")
            return upload_info
            
        except Exception as e:
            raise IntegrationError(f"Failed to upload model to GCS: {str(e)}")
    
    def deploy_to_vertex_ai(self, model_gcs_path: str, model_display_name: str,
                           machine_type: str = 'n1-standard-4') -> Dict[str, Any]:
        """
        Deploy model to Vertex AI.
        
        Args:
            model_gcs_path: GCS path to model artifacts
            model_display_name: Display name for the model
            machine_type: Machine type for deployment
            
        Returns:
            Deployment information
        """
        try:
            # Upload model to Vertex AI Model Registry
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=model_gcs_path,
                serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest"
            )
            
            # Deploy model to endpoint
            endpoint = model.deploy(
                machine_type=machine_type,
                min_replica_count=1,
                max_replica_count=3
            )
            
            deployment_info = {
                'model_id': model.resource_name,
                'model_display_name': model_display_name,
                'endpoint_id': endpoint.resource_name,
                'machine_type': machine_type,
                'status': 'deployed',
                'deployed_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Vertex AI deployment completed: {model_display_name}")
            return deployment_info
            
        except Exception as e:
            raise IntegrationError(f"Failed to deploy to Vertex AI: {str(e)}")
    
    def setup_bigquery_integration(self, dataset_id: str, table_id: str) -> Dict[str, Any]:
        """
        Set up BigQuery integration for data and predictions.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            Integration setup information
        """
        try:
            # Create dataset if it doesn't exist
            dataset_ref = self.bigquery_client.dataset(dataset_id)
            
            try:
                dataset = self.bigquery_client.get_dataset(dataset_ref)
            except:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset = self.bigquery_client.create_dataset(dataset)
            
            # Create table schema for predictions
            schema = [
                bigquery.SchemaField("prediction_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("input_data", "JSON", mode="REQUIRED"),
                bigquery.SchemaField("prediction", "JSON", mode="REQUIRED"),
                bigquery.SchemaField("confidence", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            ]
            
            table_ref = dataset_ref.table(table_id)
            table = bigquery.Table(table_ref, schema=schema)
            
            try:
                table = self.bigquery_client.create_table(table)
            except:
                # Table already exists
                pass
            
            integration_info = {
                'project_id': self.project_id,
                'dataset_id': dataset_id,
                'table_id': table_id,
                'table_ref': f"{self.project_id}.{dataset_id}.{table_id}",
                'setup_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"BigQuery integration setup: {dataset_id}.{table_id}")
            return integration_info
            
        except Exception as e:
            raise IntegrationError(f"Failed to setup BigQuery integration: {str(e)}")


class AzureIntegration:
    """
    Microsoft Azure integration for MZ Max.
    
    Provides integration with Azure ML, Blob Storage, and other Azure services.
    """
    
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """
        Initialize Azure integration.
        
        Args:
            subscription_id: Azure subscription ID
            resource_group: Resource group name
            workspace_name: Azure ML workspace name
        """
        if not AZURE_AVAILABLE:
            raise IntegrationError("Azure SDK not available. Install with: pip install azure-ai-ml azure-storage-blob")
        
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        try:
            # Initialize Azure ML client
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            
            # Initialize Blob Storage client
            self.blob_service_client = BlobServiceClient(
                account_url="https://<account_name>.blob.core.windows.net",
                credential=credential
            )
            
            logger.info("Azure integration initialized successfully")
            
        except Exception as e:
            raise IntegrationError(f"Failed to initialize Azure integration: {str(e)}")
    
    def upload_model_to_blob(self, model_path: str, container_name: str, 
                            blob_name: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Upload model to Azure Blob Storage.
        
        Args:
            model_path: Local path to model file
            container_name: Blob container name
            blob_name: Blob name
            metadata: Optional metadata
            
        Returns:
            Upload information
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(model_path, "rb") as data:
                blob_client.upload_blob(data, metadata=metadata, overwrite=True)
            
            blob_url = blob_client.url
            
            upload_info = {
                'container': container_name,
                'blob': blob_name,
                'url': blob_url,
                'metadata': metadata,
                'uploaded_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Model uploaded to Azure Blob: {blob_url}")
            return upload_info
            
        except Exception as e:
            raise IntegrationError(f"Failed to upload model to Azure Blob: {str(e)}")
    
    def deploy_to_azure_ml(self, model_path: str, model_name: str,
                          instance_type: str = "Standard_DS3_v2") -> Dict[str, Any]:
        """
        Deploy model to Azure ML.
        
        Args:
            model_path: Path to model file
            model_name: Model name
            instance_type: Azure VM instance type
            
        Returns:
            Deployment information
        """
        try:
            # Register model
            from azure.ai.ml.entities import Model
            from azure.ai.ml.constants import AssetTypes
            
            model = Model(
                path=model_path,
                type=AssetTypes.CUSTOM_MODEL,
                name=model_name,
                description=f"MZ Max model: {model_name}"
            )
            
            registered_model = self.ml_client.models.create_or_update(model)
            
            # Create deployment (simplified)
            deployment_info = {
                'model_name': model_name,
                'model_id': registered_model.id,
                'instance_type': instance_type,
                'status': 'registered',
                'deployed_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Azure ML model registered: {model_name}")
            return deployment_info
            
        except Exception as e:
            raise IntegrationError(f"Failed to deploy to Azure ML: {str(e)}")
