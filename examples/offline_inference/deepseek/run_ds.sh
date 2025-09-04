#!/bin/bash

# Default model path
FP8_MODEL_PATH="/mnt/weka/data/pytorch/DeepSeek-R1/"
# FP8_MODEL_PATH="/mnt/weka/llm/Qwen3-30B-A3B-FP8/"
# FP8_MODEL_PATH="/software/users/yiliu4/HF_HOME/Qwen/Qwen3-30B-A3B"
# FP8_MODEL_PATH="/software/users/yiliu4/HF_HOME/Qwen/Qwen3-32B"
# FP8_MODEL_PATH="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207"
# Default options
USE_SCALAR_FORMAT=false
ENABLE_PATCHING=false
ENABLE_MEASURE=false
# Parse command-line arguments
for arg in "$@"; do
    case $arg in
        --use-scalar)
            USE_SCALAR_FORMAT=true
            shift
            ;;
        --patching)
            ENABLE_PATCHING=true
            shift
            ;;
        --m)
            ENABLE_MEASURE=true
            shift
            ;;
        *)
            # Ignore unknown arguments
            ;;
    esac
done

# Set quantization configuration based on --use-scalar
if [ "$USE_SCALAR_FORMAT" = true ]; then
    export QUANT_CONFIG="./quant_configs/inc_quant_config_scalar.json"
    scale_format="scalar"
else
    export QUANT_CONFIG="./quant_configs/inc_quant_config_const.json"
    scale_format="const"
fi

if [ "$ENABLE_MEASURE" = true ]; then
    export QUANT_CONFIG="./quant_configs/inc_measure.json"
    echo "Measurement mode enabled."

fi
# Enable patching if --patching is set
if [ "$ENABLE_PATCHING" = true ]; then
    export RUNTIME_SCALE_PATCHING=1
else
    export RUNTIME_SCALE_PATCHING=0
fi

# export CALC_SCALE_WITH_CGUID=1
# export VLLM_DISABLE_MARK_SCALES_AS_CONST=true
export VLLM_SKIP_WARMUP=true


WORLD_SIZE=8
# WORLD_SIZE=1

echo "Using model path: ${FP8_MODEL_PATH}"
echo "Using quantization config: ${QUANT_CONFIG}"
echo "Using scale format: ${scale_format}"
echo "Using patching: ${ENABLE_PATCHING}"
echo "World size: ${WORLD_SIZE}"
echo "RUNTIME_SCALE_PATCHING: ${RUNTIME_SCALE_PATCHING}"

# Logging setup
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="inc_quant.${scale_format}.${timestamp}.log"

# Run the Python script

PT_HPU_LAZY_MODE=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_EXPONENTIAL_BUCKETING=false \
PT_HPU_LAZY_MODE=1 \
    python deepseek_example.py \
    --model "${FP8_MODEL_PATH}" \
    --tokenizer "${FP8_MODEL_PATH}" \
    --osl 32 \
    --max_num_seqs 1 \
    --tp_size "${WORLD_SIZE}" \
    --ep_size "${WORLD_SIZE}" \
    --max_model_len 2048 2>&1 | tee "${LOG_FILE}"
    
    
    # --dummy \
    
    

# def update_tensor_shape(tensor):
#     import torch
#     # if the tensor numel is 1, create a pure tensor
#     if tensor.numel() == 1:
#         return torch.tensor(tensor.item())
#     return tensor

# def scale_to_cpu(scale_tensor):
#     scale_tensor = update_tensor_shape(scale_tensor)
#     return scale_tensor.to("cpu")
