#!/bin/bash

set -e  # Exit on any error

ENV="local"
echo "=== OmniVideo Training Environment: $ENV ==="

# Set environment variables based on deployment
if [ "$ENV" = "local" ]; then
    MASTER_ADDR="localhost"
    MASTER_PORT=$(( 10000 + RANDOM % 50000 ))
    NNODES=1
    NODE_RANK=0
    NGPUS_PER_NODE=$(nvidia-smi -L | wc -l)
    echo "Detected $NGPUS_PER_NODE GPU(s)"
fi

# Base directory and Python path setup
BASE_DIR=$(pwd)
export PYTHONPATH="$BASE_DIR:$BASE_DIR/nets/third_party:${PYTHONPATH}"

# CUDA environment (adjust path as needed)
CUDA_HOME="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/yanghao/software/cuda-11.8"
if [ -d "$CUDA_HOME" ]; then
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
    echo "CUDA environment configured: $CUDA_HOME"
fi

# Configuration paths
CONFIG_NAME="./configs/foster/omnivideo_mixed_task_1_3B.yaml"
WAN_CKPT_DIR="./omni_ckpts/wan/wanxiang1_3b"
RESUME_FROM="./omni_ckpts/transformer/model.pt"

# Validate configuration files exist
if [ ! -f "$CONFIG_NAME" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG_NAME"
    exit 1
fi

if [ ! -d "$WAN_CKPT_DIR" ]; then
    echo "❌ Error: WAN checkpoint directory not found: $WAN_CKPT_DIR"
    exit 1
fi

echo "✅ Configuration validated"
echo "  Config: $CONFIG_NAME"
echo "  WAN checkpoint: $WAN_CKPT_DIR"

# Output directory with timestamp
OUTPUT_DIR="./output/mixed_tasks/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "✅ Output directory created: $OUTPUT_DIR"


# Execute training
echo "=== Starting Multi-Task Training ==="
echo "Configuration: $CONFIG_NAME"
echo "Output: $OUTPUT_DIR"

# Execute training
torchrun \
    --nproc_per_node="$NGPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    finetune_model.py \
    --config "$CONFIG_NAME" \
    --ckpt_dir "$WAN_CKPT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"}