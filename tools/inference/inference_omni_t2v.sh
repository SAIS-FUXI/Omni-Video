#!/bin/bash

# =============================================================================
# OmniVideo T2V Inference Script
# =============================================================================
# This script runs Text-to-Video generation using the unified generate_omni_x2v.py script.
# =============================================================================

# GPU settings
export CUDA_VISIBLE_DEVICES=0  # Set your GPU IDs
NGPUS_PER_NODE=1

# Setup Python path
CUR_WORKDIR=$(pwd)
export PYTHONPATH=$CUR_WORKDIR:$PYTHONPATH

# NCCL settings
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# Model Paths (UPDATE THESE TO YOUR PATHS)
# =============================================================================
# Checkpoint root directory.
# Expected structure:
#   ${CKPT_DIR}/high_noise_model/model.pt
#   ${CKPT_DIR}/low_noise_model/model.pt
CKPT_DIR="{path to your checkpoint directory}"

# Qwen3-VL model for online feature extraction
QWEN3VL_MODEL_PATH="{path to your Qwen3-VL model directory}"

# =============================================================================
# Input Data (JSONL format only)
# =============================================================================
# For T2V: Each line should have "prompt"
DATA_FILE="samples/t2v_example.jsonl"

# =============================================================================
# Generation Parameters
# =============================================================================
SAMPLE_SOLVER="unipc"
SAMPLE_STEPS=50
SAMPLE_GUIDE_SCALE=3.0
SAMPLE_SHIFT=5
BASE_SEED=1818

# Video settings
GEN_SIZE="832*480"
GEN_FRAME_NUM=81
GEN_SAMPLE_FPS=16
GEN_TASK="t2v-14B"

# =============================================================================
# Run Inference
# =============================================================================
echo "=========================================="
echo "Generating T2V task: $GEN_TASK with size $GEN_SIZE"
echo "=========================================="

torchrun \
    --nproc_per_node=$NGPUS_PER_NODE \
    --master_addr=localhost \
    --master_port=$((10000 + RANDOM % 50000)) \
    tools/inference/generate_omni_t2v.py \
    --task ${GEN_TASK} \
    --size ${GEN_SIZE} \
    --frame_num ${GEN_FRAME_NUM} \
    --sample_fps ${GEN_SAMPLE_FPS} \
    --sample_shift ${SAMPLE_SHIFT} \
    --sample_solver ${SAMPLE_SOLVER} \
    --sample_steps ${SAMPLE_STEPS} \
    --sample_guide_scale ${SAMPLE_GUIDE_SCALE} \
    --base_seed ${BASE_SEED} \
    --ckpt_dir ${CKPT_DIR} \
    --qwen3vl_model_path ${QWEN3VL_MODEL_PATH} \
    --prompt_file ${DATA_FILE}

echo "=========================================="
echo "Inference completed!"
echo "=========================================="
