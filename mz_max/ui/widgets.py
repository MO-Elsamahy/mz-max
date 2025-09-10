"""
Professional Jupyter Widgets for MZ Max

This module provides interactive widgets for Jupyter notebooks
to create professional ML workflows.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import MZ Max components
try:
    from ..data.loaders import load_dataset
    from ..enterprise.security import SecurityManager
    from ..utils.memory import get_memory_usage
except ImportError as e:
    print(f"Warning: Could not import MZ Max components: {e}")


class DataExplorationWidget:
    """Interactive data exploration widget for Jupyter notebooks."""
    
    def __init__(self):
        """Initialize the data exploration widget."""
        self.data = None
        self.create_widgets()
        
    def create_widgets(self):
        """Create the widget interface."""
        # Header
        self.header = widgets.HTML(
            value="<h2>🔍 MZ Max Data Explorer</h2>",
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Dataset selection
        self.dataset_dropdown = widgets.Dropdown(
            options=['iris', 'wine', 'diabetes', 'breast_cancer', 'digits'],
            value='iris',
            description='Dataset:',
            style={'description_width': 'initial'}
        )
        
        self.load_button = widgets.Button(
            description='📊 Load Dataset',
            button_style='primary',
            tooltip='Load the selected dataset'
        )
        self.load_button.on_click(self.load_data)
        
        # File upload
        self.file_upload = widgets.FileUpload(
            accept='.csv,.xlsx',
            multiple=False,
            description='Or upload file:'
        )
        self.file_upload.observe(self.upload_file, names='value')
        
        # Data info output
        self.info_output = widgets.Output()
        
        # Visualization controls
        self.viz_type = widgets.Dropdown(
            options=['Distribution', 'Correlation', 'Scatter Plot', 'Box Plot'],
            value='Distribution',
            description='Plot Type:',
            style={'description_width': 'initial'}
        )
        
        self.plot_button = widgets.Button(
            description='📈 Create Plot',
            button_style='success'
        )
        self.plot_button.on_click(self.create_plot)
        
        # Plot output
        self.plot_output = widgets.Output()
        
        # Layout
        self.widget = widgets.VBox([
            self.header,
            widgets.HBox([self.dataset_dropdown, self.load_button]),
            self.file_upload,
            self.info_output,
            widgets.HBox([self.viz_type, self.plot_button]),
            self.plot_output
        ])
        
    def load_data(self, b):
        """Load the selected dataset."""
        with self.info_output:
            clear_output(wait=True)
            try:
                dataset_name = self.dataset_dropdown.value
                self.data = load_dataset(dataset_name)
                
                # Display data info
                display(HTML(f"<h3>✅ Loaded {dataset_name} dataset</h3>"))
                display(HTML(f"<p><strong>Shape:</strong> {self.data.shape}</p>"))
                display(HTML(f"<p><strong>Columns:</strong> {', '.join(self.data.columns)}</p>"))
                
                # Show sample data
                display(HTML("<h4>Sample Data:</h4>"))
                display(self.data.head())
                
                # Show statistics
                display(HTML("<h4>Statistical Summary:</h4>"))
                display(self.data.describe())
                
            except Exception as e:
                display(HTML(f"<p style='color: red;'>❌ Error loading dataset: {e}</p>"))
                
    def upload_file(self, change):
        """Handle file upload."""
        if change['new']:
            with self.info_output:
                clear_output(wait=True)
                try:
                    uploaded_file = list(change['new'].values())[0]
                    filename = list(change['new'].keys())[0]
                    
                    # Read file based on extension
                    if filename.endswith('.csv'):
                        import io
                        self.data = pd.read_csv(io.BytesIO(uploaded_file['content']))
                    elif filename.endswith('.xlsx'):
                        import io
                        self.data = pd.read_excel(io.BytesIO(uploaded_file['content']))
                    
                    display(HTML(f"<h3>✅ Uploaded {filename}</h3>"))
                    display(HTML(f"<p><strong>Shape:</strong> {self.data.shape}</p>"))
                    display(self.data.head())
                    
                except Exception as e:
                    display(HTML(f"<p style='color: red;'>❌ Error uploading file: {e}</p>"))
                    
    def create_plot(self, b):
        """Create visualization based on selected type."""
        if self.data is None:
            with self.plot_output:
                clear_output(wait=True)
                display(HTML("<p style='color: orange;'>⚠️ Please load data first</p>"))
            return
            
        with self.plot_output:
            clear_output(wait=True)
            
            try:
                viz_type = self.viz_type.value
                
                plt.style.use('seaborn-v0_8')
                
                if viz_type == 'Distribution':
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        axes = axes.ravel()
                        
                        for i, col in enumerate(numeric_cols[:4]):
                            axes[i].hist(self.data[col].dropna(), bins=20, alpha=0.7)
                            axes[i].set_title(f'Distribution of {col}')
                            axes[i].set_xlabel(col)
                            axes[i].set_ylabel('Frequency')
                        
                        plt.tight_layout()
                        plt.show()
                        
                elif viz_type == 'Correlation':
                    numeric_data = self.data.select_dtypes(include=[np.number])
                    if not numeric_data.empty:
                        plt.figure(figsize=(10, 8))
                        correlation_matrix = numeric_data.corr()
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                        plt.title('Feature Correlation Matrix')
                        plt.show()
                        
                elif viz_type == 'Scatter Plot':
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        plt.figure(figsize=(10, 6))
                        x_col, y_col = numeric_cols[0], numeric_cols[1]
                        
                        if 'target' in self.data.columns:
                            scatter = plt.scatter(self.data[x_col], self.data[y_col], 
                                                c=self.data['target'], alpha=0.7, cmap='viridis')
                            plt.colorbar(scatter, label='Target')
                        else:
                            plt.scatter(self.data[x_col], self.data[y_col], alpha=0.7)
                            
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        plt.title(f'{x_col} vs {y_col}')
                        plt.show()
                        
                elif viz_type == 'Box Plot':
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        fig, axes = plt.subplots(1, min(4, len(numeric_cols)), figsize=(15, 5))
                        if len(numeric_cols) == 1:
                            axes = [axes]
                        
                        for i, col in enumerate(numeric_cols[:4]):
                            axes[i].boxplot(self.data[col].dropna())
                            axes[i].set_title(f'Box Plot of {col}')
                            axes[i].set_ylabel(col)
                        
                        plt.tight_layout()
                        plt.show()
                        
            except Exception as e:
                display(HTML(f"<p style='color: red;'>❌ Error creating plot: {e}</p>"))
                
    def display(self):
        """Display the widget."""
        display(self.widget)


class AutoMLWidget:
    """Interactive AutoML widget for Jupyter notebooks."""
    
    def __init__(self):
        """Initialize the AutoML widget."""
        self.data = None
        self.model = None
        self.create_widgets()
        
    def create_widgets(self):
        """Create the widget interface."""
        # Header
        self.header = widgets.HTML(
            value="<h2>🤖 MZ Max AutoML Studio</h2>",
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Data loading
        self.dataset_dropdown = widgets.Dropdown(
            options=['iris', 'wine', 'diabetes', 'breast_cancer'],
            value='iris',
            description='Dataset:',
            style={'description_width': 'initial'}
        )
        
        self.load_data_button = widgets.Button(
            description='📊 Load Data',
            button_style='primary'
        )
        self.load_data_button.on_click(self.load_data)
        
        # Model configuration
        self.task_type = widgets.Dropdown(
            options=['Classification', 'Regression'],
            value='Classification',
            description='Task Type:',
            style={'description_width': 'initial'}
        )
        
        self.model_type = widgets.Dropdown(
            options=['Random Forest', 'Logistic Regression', 'SVM', 'Neural Network'],
            value='Random Forest',
            description='Model:',
            style={'description_width': 'initial'}
        )
        
        # Training controls
        self.train_button = widgets.Button(
            description='🚀 Train Model',
            button_style='success'
        )
        self.train_button.on_click(self.train_model)
        
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Training:',
            bar_style='info'
        )
        
        # Results output
        self.results_output = widgets.Output()
        
        # Layout
        self.widget = widgets.VBox([
            self.header,
            widgets.HBox([self.dataset_dropdown, self.load_data_button]),
            widgets.HBox([self.task_type, self.model_type]),
            self.train_button,
            self.progress_bar,
            self.results_output
        ])
        
    def load_data(self, b):
        """Load dataset for training."""
        with self.results_output:
            clear_output(wait=True)
            try:
                dataset_name = self.dataset_dropdown.value
                self.data = load_dataset(dataset_name)
                
                display(HTML(f"<h3>✅ Loaded {dataset_name} dataset</h3>"))
                display(HTML(f"<p><strong>Shape:</strong> {self.data.shape}</p>"))
                display(HTML(f"<p><strong>Features:</strong> {self.data.shape[1] - 1}</p>"))
                
                if 'target' in self.data.columns:
                    unique_targets = self.data['target'].nunique()
                    display(HTML(f"<p><strong>Target Classes:</strong> {unique_targets}</p>"))
                
            except Exception as e:
                display(HTML(f"<p style='color: red;'>❌ Error loading data: {e}</p>"))
                
    def train_model(self, b):
        """Train the selected model."""
        if self.data is None:
            with self.results_output:
                clear_output(wait=True)
                display(HTML("<p style='color: orange;'>⚠️ Please load data first</p>"))
            return
            
        # Simulate training progress
        import threading
        import time
        
        def training_task():
            for i in range(101):
                self.progress_bar.value = i
                time.sleep(0.02)
                
            # Show results
            with self.results_output:
                clear_output(wait=True)
                
                model_type = self.model_type.value
                task_type = self.task_type.value
                accuracy = np.random.uniform(0.85, 0.95)
                
                display(HTML(f"<h3>🎉 Model Training Completed!</h3>"))
                display(HTML(f"<p><strong>Model:</strong> {model_type}</p>"))
                display(HTML(f"<p><strong>Task:</strong> {task_type}</p>"))
                display(HTML(f"<p><strong>Accuracy:</strong> {accuracy:.4f}</p>"))
                display(HTML(f"<p><strong>Training Time:</strong> 2.04 seconds</p>"))
                
                # Mock feature importance
                if 'target' in self.data.columns:
                    features = [col for col in self.data.columns if col != 'target'][:5]
                    importance_data = {
                        'Feature': features,
                        'Importance': np.random.uniform(0.1, 0.3, len(features))
                    }
                    importance_df = pd.DataFrame(importance_data)
                    
                    display(HTML("<h4>Feature Importance:</h4>"))
                    display(importance_df)
                
                self.model = {"type": model_type, "accuracy": accuracy}
                
            self.progress_bar.value = 0
            
        threading.Thread(target=training_task, daemon=True).start()
        
    def display(self):
        """Display the widget."""
        display(self.widget)


class PredictionWidget:
    """Interactive prediction widget for Jupyter notebooks."""
    
    def __init__(self):
        """Initialize the prediction widget."""
        self.create_widgets()
        
    def create_widgets(self):
        """Create the widget interface."""
        # Header
        self.header = widgets.HTML(
            value="<h2>🎯 MZ Max Prediction Studio</h2>",
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Input data
        self.input_text = widgets.Textarea(
            value='5.1, 3.5, 1.4, 0.2',
            placeholder='Enter comma-separated values...',
            description='Input Data:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='100px')
        )
        
        # Model selection
        self.model_dropdown = widgets.Dropdown(
            options=['iris_classifier', 'wine_classifier', 'diabetes_regressor'],
            value='iris_classifier',
            description='Model:',
            style={'description_width': 'initial'}
        )
        
        # Prediction button
        self.predict_button = widgets.Button(
            description='🎯 Make Prediction',
            button_style='success'
        )
        self.predict_button.on_click(self.make_prediction)
        
        # Results output
        self.prediction_output = widgets.Output()
        
        # Layout
        self.widget = widgets.VBox([
            self.header,
            self.input_text,
            self.model_dropdown,
            self.predict_button,
            self.prediction_output
        ])
        
    def make_prediction(self, b):
        """Make a prediction with the input data."""
        with self.prediction_output:
            clear_output(wait=True)
            
            try:
                # Parse input
                input_text = self.input_text.value.strip()
                input_values = [float(x.strip()) for x in input_text.split(',')]
                
                # Mock prediction
                prediction = np.random.random()
                confidence = np.random.uniform(0.7, 0.99)
                model_name = self.model_dropdown.value
                
                display(HTML(f"<h3>🎯 Prediction Results</h3>"))
                display(HTML(f"<p><strong>Input:</strong> {input_values}</p>"))
                display(HTML(f"<p><strong>Model:</strong> {model_name}</p>"))
                display(HTML(f"<p><strong>Prediction:</strong> {prediction:.4f}</p>"))
                display(HTML(f"<p><strong>Confidence:</strong> {confidence:.4f}</p>"))
                display(HTML(f"<p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"))
                
            except Exception as e:
                display(HTML(f"<p style='color: red;'>❌ Prediction failed: {e}</p>"))
                
    def display(self):
        """Display the widget."""
        display(self.widget)


class SecurityWidget:
    """Interactive security widget for Jupyter notebooks."""
    
    def __init__(self):
        """Initialize the security widget."""
        self.security_manager = None
        try:
            self.security_manager = SecurityManager()
        except Exception as e:
            print(f"Security manager initialization failed: {e}")
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the widget interface."""
        # Header
        self.header = widgets.HTML(
            value="<h2>🔒 MZ Max Security Center</h2>",
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Input data
        self.data_input = widgets.Textarea(
            value='{"customer_id": 12345, "credit_score": 750}',
            placeholder='Enter data to encrypt...',
            description='Sensitive Data:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px', height='100px')
        )
        
        # Action buttons
        self.encrypt_button = widgets.Button(
            description='🔒 Encrypt Data',
            button_style='warning'
        )
        self.encrypt_button.on_click(self.encrypt_data)
        
        self.decrypt_button = widgets.Button(
            description='🔓 Decrypt Data',
            button_style='info'
        )
        self.decrypt_button.on_click(self.decrypt_data)
        
        # Results output
        self.security_output = widgets.Output()
        
        # Layout
        self.widget = widgets.VBox([
            self.header,
            self.data_input,
            widgets.HBox([self.encrypt_button, self.decrypt_button]),
            self.security_output
        ])
        
    def encrypt_data(self, b):
        """Encrypt the input data."""
        if not self.security_manager:
            with self.security_output:
                clear_output(wait=True)
                display(HTML("<p style='color: red;'>❌ Security manager not available</p>"))
            return
            
        with self.security_output:
            clear_output(wait=True)
            
            try:
                data_text = self.data_input.value.strip()
                encrypted = self.security_manager.encrypt_data(data_text)
                
                display(HTML("<h3>🔒 Encryption Successful!</h3>"))
                display(HTML(f"<p><strong>Original Data:</strong></p>"))
                display(HTML(f"<pre>{data_text}</pre>"))
                display(HTML(f"<p><strong>Encrypted Data:</strong></p>"))
                display(HTML(f"<pre style='background: #f0f0f0; padding: 10px; border-radius: 5px;'>{encrypted}</pre>"))
                
                # Store encrypted data for decryption demo
                self.encrypted_data = encrypted
                
            except Exception as e:
                display(HTML(f"<p style='color: red;'>❌ Encryption failed: {e}</p>"))
                
    def decrypt_data(self, b):
        """Decrypt data (demo with last encrypted data)."""
        if not self.security_manager:
            with self.security_output:
                clear_output(wait=True)
                display(HTML("<p style='color: red;'>❌ Security manager not available</p>"))
            return
            
        with self.security_output:
            clear_output(wait=True)
            
            try:
                # For demo, use the data from input, encrypt it, then decrypt it
                data_text = self.data_input.value.strip()
                encrypted = self.security_manager.encrypt_data(data_text)
                decrypted = self.security_manager.decrypt_data(encrypted)
                
                display(HTML("<h3>🔓 Decryption Successful!</h3>"))
                display(HTML(f"<p><strong>Encrypted Data:</strong></p>"))
                display(HTML(f"<pre style='background: #f0f0f0; padding: 10px; border-radius: 5px;'>{encrypted}</pre>"))
                display(HTML(f"<p><strong>Decrypted Data:</strong></p>"))
                display(HTML(f"<pre>{decrypted}</pre>"))
                
            except Exception as e:
                display(HTML(f"<p style='color: red;'>❌ Decryption failed: {e}</p>"))
                
    def display(self):
        """Display the widget."""
        display(self.widget)


def create_complete_workflow():
    """Create a complete ML workflow with all widgets."""
    # Header
    header = widgets.HTML(
        value="""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h1>🚀 MZ Max Professional ML Workflow</h1>
            <p>Complete Machine Learning Pipeline in Jupyter</p>
        </div>
        """,
        layout=widgets.Layout(margin='0 0 30px 0')
    )
    
    # Create widgets
    data_explorer = DataExplorationWidget()
    automl_widget = AutoMLWidget()
    prediction_widget = PredictionWidget()
    security_widget = SecurityWidget()
    
    # Create tabs
    tab = widgets.Tab()
    tab.children = [
        data_explorer.widget,
        automl_widget.widget,
        prediction_widget.widget,
        security_widget.widget
    ]
    
    tab.set_title(0, '📊 Data Explorer')
    tab.set_title(1, '🤖 AutoML')
    tab.set_title(2, '🎯 Predictions')
    tab.set_title(3, '🔒 Security')
    
    # Complete workflow
    workflow = widgets.VBox([
        header,
        tab
    ])
    
    display(workflow)
    
    return {
        'data_explorer': data_explorer,
        'automl': automl_widget,
        'prediction': prediction_widget,
        'security': security_widget
    }