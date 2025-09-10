#!/usr/bin/env python3
"""
MZ Max - The Most Powerful ML/DL Python Package
A comprehensive machine learning and deep learning framework
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mz-max",
    version="1.0.0",
    author="MZ Development Team",
    author_email="elsamahy771@gmail.com",
    description="The most powerful Python package for Machine Learning and Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mzmax/mz-max",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core ML/DL frameworks
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tensorflow>=2.13.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        
        # Scientific computing
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.3.0",
        
        # Data processing
        "pillow>=9.0.0",
        "opencv-python>=4.7.0",
        "librosa>=0.10.0",
        "nltk>=3.8",
        "spacy>=3.6.0",
        
        # Visualization
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "bokeh>=3.2.0",
        
        # Optimization & AutoML
        "optuna>=3.2.0",
        "hyperopt>=0.2.7",
        "ray[tune]>=2.5.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        
        # Model interpretability
        "shap>=0.42.0",
        "lime>=0.2.0",
        "captum>=0.6.0",
        
        # Utilities
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "dask>=2023.6.0",
        "mlflow>=2.5.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        
        # Web and API
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "gradio>=3.35.0",
        "streamlit>=1.25.0",
        "jinja2>=3.1.0",
        
        # UI Components
        "ipywidgets>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.0",
            "jupyter>=1.0.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mzmax=mz_max.cli:main",
            "mzmax-dashboard=mz_max.ui.dashboard:create_dashboard",
            "mzmax-gui=mz_max.ui.gui:launch_gui",
            "mzmax-webapp=mz_max.ui.web_app:launch_web_app",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
