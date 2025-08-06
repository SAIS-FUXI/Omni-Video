#!/bin/bash

set -e  # Exit on any error

# Base directory and Python path setup
BASE_DIR=$(pwd)
export PYTHONPATH="$BASE_DIR:$BASE_DIR/nets/third_party:${PYTHONPATH}"

DATA_FILE="{path to the vae feature file list}"
OUTPUT_DIR="{output directory}"

python -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=1 \
    --use_env \
    --master_port 1234 \
    tools/data_prepare/ar_feature_extract.py \
    --model-path omni_ckpts/ar_model/checkpoint \
    --data-file ${DATA_FILE} \
    --result-folder ${OUTPUT_DIR}