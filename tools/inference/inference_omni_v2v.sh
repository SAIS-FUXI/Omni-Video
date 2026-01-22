#!/bin/bash

# =============================================================================
# OmniVideo Inference Script
# =============================================================================
# This script runs video generation inference using the OmniVideo model.
# 
# Usage:
#   1. Update the model paths below to point to your checkpoints
#   2. Update DATA_FILE to point to your JSONL prompt file
#   3. Run: bash tools/inference/inference_omni_e2e.sh
# =============================================================================

# GPU settings
export CUDA_VISIBLE_DEVICES=0
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
# (and any base assets required by the pipeline)
CKPT_DIR="/inspire/ssd/project/sais-mtm/public/data/ppu_data/pretrained_models/OmniVideo2-A14B"

# Qwen3-VL model for online feature extraction
QWEN3VL_MODEL_PATH="/inspire/ssd/project/sais-mtm/public/pretrained_models/Qwen3-VL-30B-A3B-Instruct"

# =============================================================================
# Input Data (JSONL format only)
# =============================================================================
# Each line should be a JSON object with keys:
#   - "edit_prompt" or "prompt": The editing instruction
#   - "source_clip_path" or "video_path_src": Path to source video (for v2v tasks)
#   - "sample_id" or "id": Optional unique identifier
DATA_FILE="samples/input_list.jsonl"

# =============================================================================
# Generation Parameters
# =============================================================================
SAMPLE_SOLVER="unipc"
SAMPLE_STEPS=40
SAMPLE_GUIDE_SCALE=3.0
SAMPLE_SHIFT=5
BASE_SEED=1818

# Video settings
GEN_SIZE="832*480"
GEN_FRAME_NUM=41
GEN_SAMPLE_FPS=8
GEN_TASK="v2v-A14B"

# =============================================================================
# Run Inference
# =============================================================================
echo "=========================================="
echo "Generating $GEN_TASK with size $GEN_SIZE"
echo "=========================================="

torchrun \
    --nproc_per_node=$NGPUS_PER_NODE \
    --master_addr=localhost \
    --master_port=$((10000 + RANDOM % 50000)) \
    tools/inference/generate_omni_e2e.py \
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
    --prompt_file ${DATA_FILE} \
    --qwen3vl_model_path ${QWEN3VL_MODEL_PATH}

echo "=========================================="
echo "Inference completed!"
echo "=========================================="
