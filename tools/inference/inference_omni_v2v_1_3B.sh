#!/bin/bash

# =============================================================================
# OmniVideo V2V Inference Script (1.3B, single checkpoint)
# =============================================================================
# Usage:
#   1) Set CKPT_DIR and optionally NEW_CHECKPOINT below
#   2) Set DATA_FILE to your JSONL prompt file
#   3) Run: bash tools/inference/inference_omni_v2v_1_3B.sh
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
# Model Paths
# =============================================================================
# Checkpoint root directory for OmniVideo assets (T5/VAE/tokenizer/etc.)
CKPT_DIR=""

# Qwen3-VL model for online feature extraction
QWEN3VL_MODEL_PATH=""

# =============================================================================
# Input Data (JSONL format only)
# =============================================================================
DATA_FILE="samples/input_list.jsonl"

# =============================================================================
# Generation Parameters
# =============================================================================

# Sampler
SAMPLE_SOLVER="unipc"
SAMPLE_STEPS=40
SAMPLE_GUIDE_SCALE=3.0
SAMPLE_SHIFT=5
BASE_SEED=1818
CLASSIFIER_FREE_RATIO=0.0

# Video settings
GEN_SIZE="832*480"
GEN_FRAME_NUM=41
GEN_SAMPLE_FPS=8
GEN_TASK="v2v-1.3B"
SAMPLING_RATE=3
SKIP_NUM=0

# Distributed / memory options
USE_USP=False
SP_SIZE=1
T5_FSDP=False
DIT_FSDP=False
MAX_CONTEXT_LEN=6272

# Qwen3-VL options (effective when QWEN3VL_MODEL_PATH is provided)
QWEN3VL_DTYPE="bf16"
QWEN3VL_DEVICE_MAP="auto"
QWEN3VL_TEMPERATURE=0.0
VIDEO_MAX_DURATION=5.0

echo "=========================================="
echo "Generating ${GEN_TASK} with size ${GEN_SIZE}"
echo "=========================================="

EXTRA_ARGS=()
if [ -n "${NEW_CHECKPOINT}" ]; then
    EXTRA_ARGS+=(--new_checkpoint "${NEW_CHECKPOINT}")
fi
if [ -n "${PROMPT}" ]; then
    EXTRA_ARGS+=(--prompt "${PROMPT}")
fi
if [ -n "${DATA_FILE}" ]; then
    EXTRA_ARGS+=(--prompt_file "${DATA_FILE}")
fi
if [ -n "${QWEN3VL_MODEL_PATH}" ]; then
    EXTRA_ARGS+=(--qwen3vl_model_path "${QWEN3VL_MODEL_PATH}")
fi

torchrun \
    --nproc_per_node=$NGPUS_PER_NODE \
    --master_addr=localhost \
    --master_port=$((10000 + RANDOM % 50000)) \
    tools/inference/generate_omni_v2v_1_3B.py \
    --task ${GEN_TASK} \
    --size ${GEN_SIZE} \
    --frame_num ${GEN_FRAME_NUM} \
    --sample_fps ${GEN_SAMPLE_FPS} \
    --sample_shift ${SAMPLE_SHIFT} \
    --sample_solver ${SAMPLE_SOLVER} \
    --sample_steps ${SAMPLE_STEPS} \
    --sample_guide_scale ${SAMPLE_GUIDE_SCALE} \
    --base_seed ${BASE_SEED} \
    --classifier_free_ratio ${CLASSIFIER_FREE_RATIO} \
    --ckpt_dir ${CKPT_DIR} \
    --sampling_rate ${SAMPLING_RATE} \
    --skip_num ${SKIP_NUM} \
    --use_usp ${USE_USP} \
    --sp_size ${SP_SIZE} \
    --t5_fsdp ${T5_FSDP} \
    --dit_fsdp ${DIT_FSDP} \
    --max_context_len ${MAX_CONTEXT_LEN} \
    --qwen3vl_dtype ${QWEN3VL_DTYPE} \
    --qwen3vl_device_map ${QWEN3VL_DEVICE_MAP} \
    --qwen3vl_temperature ${QWEN3VL_TEMPERATURE} \
    --video_max_duration ${VIDEO_MAX_DURATION} \
    "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Inference completed!"
echo "=========================================="

