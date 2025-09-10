---
layout: default
title: Installation Guide
description: Complete installation guide for MZ Max Professional
---

# Installation Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM
- **Storage**: 2GB free disk space
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Recommended Requirements
- **Python**: 3.11 (latest stable)
- **Memory**: 8GB+ RAM
- **Storage**: 5GB+ free disk space
- **GPU**: CUDA-compatible GPU (optional, for deep learning acceleration)

## Installation Methods

### Standard Installation (Recommended)

Install MZ Max Professional from PyPI:

```bash
pip install mz-max
```

### Development Installation

For contributors and developers:

```bash
git clone https://github.com/mzmax/mz-max.git
cd mz-max
pip install -e ".[dev]"
```

### Docker Installation

Run MZ Max Professional in a container:

```bash
# Pull the image
docker pull mzmax/mz-max:latest

# Run with UI ports exposed
docker run -p 8501:8501 -p 8000:8000 mzmax/mz-max:latest
```

### Conda Installation

Using conda-forge (coming soon):

```bash
conda install -c conda-forge mz-max
```

## Verification

Verify your installation:

```python
import mz_max as mz
print(f"MZ Max version: {mz.__version__}")

# Test basic functionality
data = mz.load_dataset('iris')
print(f"Loaded dataset with shape: {data.shape}")
```

## Optional Dependencies

### UI Components
```bash
# For web interfaces
pip install streamlit plotly

# For desktop GUI
pip install matplotlib seaborn

# For Jupyter widgets
pip install ipywidgets
```

### Deep Learning Frameworks
```bash
# PyTorch (CPU)
pip install torch torchvision

# PyTorch (GPU - CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# TensorFlow
pip install tensorflow
```

### Cloud Providers
```bash
# AWS
pip install boto3

# Google Cloud
pip install google-cloud-storage

# Azure
pip install azure-storage-blob
```

## Platform-Specific Instructions

### Windows

1. **Install Python** from [python.org](https://python.org) or Microsoft Store
2. **Install Visual C++ Build Tools** (for some dependencies)
3. **Install MZ Max**:
   ```cmd
   pip install mz-max
   ```

### macOS

1. **Install Python** using Homebrew:
   ```bash
   brew install python
   ```
2. **Install MZ Max**:
   ```bash
   pip install mz-max
   ```

### Linux (Ubuntu/Debian)

1. **Update system packages**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-dev
   ```
2. **Install MZ Max**:
   ```bash
   pip3 install mz-max
   ```

## GPU Support

### NVIDIA CUDA Setup

1. **Install CUDA Toolkit** from [NVIDIA website](https://developer.nvidia.com/cuda-toolkit)
2. **Install cuDNN** (for TensorFlow)
3. **Verify GPU availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   ```

### AMD ROCm Setup (Linux)

1. **Install ROCm** following [AMD's guide](https://rocmdocs.amd.com/)
2. **Install PyTorch for ROCm**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
   ```

## Troubleshooting

### Common Issues

**Import Error: No module named 'mz_max'**
```bash
# Ensure correct Python environment
python -m pip install mz-max

# Check installation
pip list | grep mz-max
```

**Permission Denied (Windows)**
```cmd
# Run as administrator or use --user flag
pip install --user mz-max
```

**Build Errors (Linux)**
```bash
# Install build dependencies
sudo apt install build-essential python3-dev

# Or use conda for problematic packages
conda install -c conda-forge scikit-learn
```

**Memory Issues**
```bash
# Install with no cache to reduce memory usage
pip install --no-cache-dir mz-max
```

### Getting Help

If you encounter installation issues:

1. **Check System Requirements** - Ensure your system meets minimum requirements
2. **Update pip** - `pip install --upgrade pip`
3. **Use Virtual Environment** - Isolate dependencies
4. **Check GitHub Issues** - Search for similar problems
5. **Contact Support** - elsamahy771@gmail.com

## Next Steps

After successful installation:

1. **Quick Start** - Follow the [Quick Start Guide](quickstart.html)
2. **Launch UI** - Try the professional interfaces
3. **Explore Examples** - Check out the [examples](examples/)
4. **Read Documentation** - Browse the [API Reference](api.html)

---

Ready to get started? Head to the [Quick Start Guide](quickstart.html) to begin your machine learning journey with MZ Max Professional.
