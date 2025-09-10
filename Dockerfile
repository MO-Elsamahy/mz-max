# Multi-stage Dockerfile for MZ Max
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mzmax

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY setup.py .
RUN pip install -e .[dev]

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R mzmax:mzmax /app
USER mzmax

# Expose ports for development
EXPOSE 8000 8501 8888

# Default command for development
CMD ["python", "-m", "mz_max.ui.dashboard"]

# Production stage
FROM base as production

# Copy only necessary files
COPY setup.py .
COPY mz_max/ ./mz_max/
COPY README.md .
COPY LICENSE .

# Install package
RUN pip install --no-cache-dir .

# Change ownership to non-root user
RUN chown -R mzmax:mzmax /app
USER mzmax

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "mz_max.ui.web_app"]

# Jupyter stage for data science workflows
FROM base as jupyter

# Install Jupyter and extensions
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    jupyter_contrib_nbextensions

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .[dev]

# Enable Jupyter extensions
RUN jupyter contrib nbextension install --system && \
    jupyter nbextension enable --py widgetsnbextension --system

# Change ownership
RUN chown -R mzmax:mzmax /app
USER mzmax

# Expose Jupyter port
EXPOSE 8888

# Jupyter command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy source code and install
COPY . .
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash mzmax && \
    chown -R mzmax:mzmax /app

USER mzmax

# Expose port
EXPOSE 8000

# GPU command
CMD ["python", "-m", "mz_max.ui.web_app"]
