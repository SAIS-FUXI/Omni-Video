ENV="local"

# Set environment variables based on deployment
if [ "$ENV" = "local" ]; then
  MASTER_ADDR="localhost"
  MASTER_PORT=$(( 10000 + RANDOM % 50000 ))
  NNODES=1
  NODE_RANK=0
  NGPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi

# Base directory and Python path setup
BASE_DIR=$(pwd)
export PYTHONPATH="$BASE_DIR:$BASE_DIR/nets/third_party:${PYTHONPATH}"

CKPT_DIR="omni_ckpts/wan/wanxiang1_3b"  # Path to model checkpoints

# set the path to the input json file and output directory
VIDEO_LIST_PATH="{path to input json file}"
OUTPUT_DIR="{output directory}"

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$NGPUS_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  tools/data_prepare/vae_feature_extract.py \
  --task "t2v-1.3B" \
  --ckpt_dir ${CKPT_DIR} \
  --video_list_path ${VIDEO_LIST_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --frame_num 17 \
  --sampling_rate 2 \
  --skip_num 0 \
  --target_size "352,640"
