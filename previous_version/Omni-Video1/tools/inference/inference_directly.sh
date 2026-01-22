#!/bin/bash
set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"

# Set Python path
export PYTHONPATH="${PROJECT_ROOT}/nets/third_party:${PYTHONPATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Distributed training settings
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export NNODES="1"
export NODE_RANK="0"
NGPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l || echo "1")

echo "=== OmniVideo Simplified Multi-Task Generation ==="
echo "Using ${NGPUS_PER_NODE} GPU(s)"

# Validate model directory
MODELS_DIR="${PROJECT_ROOT}/omni_ckpts"
if [[ ! -d "${MODELS_DIR}" ]]; then
    echo "❌ Error: Models directory not found: ${MODELS_DIR}"
    exit 1
fi
echo "✅ Using models from: ${MODELS_DIR}"

#==============================================================================
# SIMPLIFIED COMMAND TEMPLATE
#==============================================================================

BASE_CMD=(
    torchrun
    --nproc_per_node="$NGPUS_PER_NODE"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    --nnodes="$NNODES"
    --node_rank="$NODE_RANK"
    "${PROJECT_ROOT}/tools/inference/generate.py"
    --sample_solver "unipc"
    --adapter_in_channels "1152"
    --adapter_out_channels "4096"
    --adapter_query_length "256"
    --use_visual_context_adapter "true"
    --visual_context_adapter_patch_size "1,4,4"
    --use_visual_as_input false
    --condition_mode "full"
    --max_context_len "2560"
    --ar_model_num_video_frames "8"
    --ar_conv_mode "llama_3"
    --sampling_rate "3"
    --skip_num "1"
    --unconditioned_context_length "2560"
    --classifier_free_ratio "0.0"
)

#==============================================================================
# TEXT-TO-VIDEO (T2V) TASKS
#==============================================================================

# echo "=== Running Text-to-Video (T2V) Tasks ==="

# # T2V Task 1
# echo "Running T2V Task 1: Sunset video..."
# "${BASE_CMD[@]}" \
#     --task "t2v" \
#     --prompt "A beautiful sunset over the ocean with gentle waves" \
#     --size "832*480" \
#     --frame_num "81" \
#     --sample_steps "50" \
#     --sample_fps "16" \
#     --sample_guide_scale "5.0" \
#     --base_seed "42"

# # T2V Task 2  
# echo "Running T2V Task 2: Cat playing..."
# "${BASE_CMD[@]}" \
#     --task "t2v" \
#     --prompt "A cat playing with a ball of yarn in slow motion" \
#     --size "832*480" \
#     --frame_num "81" \
#     --sample_steps "50" \
#     --sample_fps "16" \
#     --sample_guide_scale "5.0" \
#     --base_seed "123"

#==============================================================================
# TEXT-TO-IMAGE (T2I) TASKS
#==============================================================================

echo "=== Running Text-to-Image (T2I) Tasks ==="

# T2I Task 1
"${BASE_CMD[@]}" \
    --task "t2i" \
    --prompt "A black and white photograph of a serene mountain lake surrounded by trees. The mountain can be seen in the distance, towering over the lake. The reflection of the mountain and its snowy peak can be seen clearly in the calm waters of the lake." \
    --size "480*480" \
    --frame_num "1" \
    --sample_steps "50" \
    --sample_guide_scale "5.0" \
    --base_seed "1818"
    
# T2I Task 2
"${BASE_CMD[@]}" \
    --task "t2i" \
    --prompt "Antarctic icebergs sculpted by polar winds. Azure caverns glow in monumental ice cliffs floating through silver seas where penguins porpoise through waves." \
    --size "480*480" \
    --frame_num "1" \
    --sample_steps "50" \
    --sample_guide_scale "5.0" \
    --base_seed "1818"

# T2I Task 3
"${BASE_CMD[@]}" \
    --task "t2i" \
    --prompt "A futuristic image featuring a woman with glowing yellow eyes. She is wearing a black and gold outfit, which is adorned with intricate patterns and lights. The woman has long hair, and her eyes are prominently glowing yellow. She is the main focus of the scene." \
    --size "480*480" \
    --frame_num "1" \
    --sample_steps "50" \
    --sample_guide_scale "5.0" \
    --base_seed "1818"

# T2I Task 4
"${BASE_CMD[@]}" \
    --task "t2i" \
    --prompt "An image that captures the essence of retro chic. At the center of the frame is a woman exuding a sense of confidence and style." \
    --size "480*480" \
    --frame_num "1" \
    --sample_steps "50" \
    --sample_guide_scale "5.0" \
    --base_seed "1818"

#==============================================================================
# IMAGE-TO-IMAGE (I2I) TASKS
#==============================================================================

echo "=== Running Image-to-Image (I2I) Tasks ==="

# I2I Task 1
echo "Running I2I Task 1: Transform to painting..."
"${BASE_CMD[@]}" \
    --task "i2i" \
    --prompt "add more logs to the fire" \
    --size "480*480" \
    --frame_num "1" \
    --sample_steps "40" \
    --sample_guide_scale "3.0" \
    --base_seed "666" \
    --src_file_path "./examples/i2i_source_1.jpg"

#==============================================================================
# VIDEO-TO-VIDEO (V2V) TASKS
#==============================================================================

echo "=== Running Video-to-Video (V2V) Tasks ==="

# V2V Task 1
"${BASE_CMD[@]}" \
    --task "v2v" \
    --prompt "add a hot air balloon floating above the clouds" \
    --size "640*352" \
    --frame_num 17 \
    --sample_fps 8 \
    --sample_steps 40 \
    --sample_guide_scale 3.0 \
    --sampling_rate 2 \
    --base_seed "1818" \
    --src_file_path "examples/source_0.mp4"

# V2V Task 2
"${BASE_CMD[@]}" \
    --task "v2v" \
    --prompt "Make the Corgi black and white" \
    --size "640*352" \
    --frame_num 17 \
    --sample_fps 8 \
    --sample_steps 40 \
    --sample_guide_scale 3.0 \
    --sampling_rate 2 \
    --base_seed "1573" \
    --src_file_path "examples/source_1.mp4"

echo "=== All Generation Tasks Completed ==="
echo "Check the ./output directory for generated files."
echo ""
echo "🎉 Simplified multi-task generation finished!"
