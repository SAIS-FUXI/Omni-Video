# 🚀 OmniVideo Complete Setup Guide

This comprehensive guide covers everything you need to set up and run OmniVideo for training and inference.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Model Organization](#model-organization)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)

## 🔧 Environment Setup

### Option 1: Using Conda (Recommended)

#### Create New Environment
```bash
# Create and activate new conda environment
conda create -n omnivideo python=3.10
conda activate omnivideo

# Install from our curated requirements
pip install -r project_requirements.txt

# Or recreate exact environment
conda env create -f environment.yml
```

### Option 2: Using Pip

#### Full Installation
```bash
# Create virtual environment
python -m venv omnivideo_env
source omnivideo_env/bin/activate  # Linux/Mac

# Install all dependencies
pip install -r pip_requirements.txt
```

## 📁 Model Organization

### Directory Structure
The models should be organized in the `omni_ckpts` directory as follows:

```
omni_ckpts/
├── wan/
│   └── wanxiang1_3b/          # WAN model checkpoints
├── adapter/
│   └── model.pt               # Adapter model checkpoint  
├── vision_head/
│   └── vision_head/           # Vision head checkpoints
├── transformer/
│   └── model.pt               # Transformer model checkpoint
├── ar_model/
│   └── checkpoint/      # AR model checkpoint
├── unconditioned_context/
│   └── context.pkl            # Unconditioned context for classifier-free guidance
└── special_tokens/
    └── tokens.pkl             # Special token embeddings
```

## 🛠 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd omini_video
```

### 2. Set Environment Variables
```bash
# For CUDA (adjust path as needed)
export CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Add to inference or training script
export PYTHONPATH="${PWD}:${PWD}/nets/third_party:${PYTHONPATH}"
```

### 3. Verify Installation
```bash
python -c "
import torch
import transformers
import deepspeed
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ DeepSpeed: {deepspeed.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
"
```

## 🚀 Quick Start

### Inference
```bash
# Run inference with sample data
bash tools/inference/inference.sh
```

### Training
```bash
# Quick training with sample data
bash finetune.sh
```

## 💻 Development Setup

### Option 1: Sample Data
We provide sample data for quick testing:
```bash
# Sample data already included in:
examples/finetune_data/
├── t2i_sample/     # Text-to-Image samples (4 files)
├── i2i_sample/     # Image-to-Image samples (4 files)
├── t2v_sample/     # Text-to-Video samples (4 files)
├── t2i_sample_paths.txt
├── i2i_sample_paths.txt
└── t2v_sample_paths.txt
```

### Option 2: Prepare new data from raw videos and prompts.
The detailed introduciton of data preparation can be found in tools/data_prepare/DATA_PREPARE.md

### Configuration Files
- **Training Config**: `configs/foster/omnivideo_mixed_task_1_3B.yaml`
- **Model Paths**: Configured to use `omni_ckpts/` directory
- **Sample Data**: Already configured in the YAML

## 🔧 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA 11.8+ support
- **Memory**: 32GB+ RAM recommended
- **Storage**: 50GB+ free space for models

### Software
- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU acceleration)
- **GCC**: 7.5+ (for compilation)