#!/bin/bash

# Default model path
FP8_MODEL_PATH="/mnt/weka/data/pytorch/DeepSeek-R1/"

# Default options
USE_SCALAR_FORMAT=false
ENABLE_PATCHING=false

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

# Enable patching if --patching is set
if [ "$ENABLE_PATCHING" = true ]; then
    export RUNTIME_SCALE_PATCH=1
else
    export RUNTIME_SCALE_PATCH=0
fi

export CALC_SCALE_WITH_CGUID=1
export VLLM_SKIP_WARMUP=true
export VLLM_DISABLE_MARK_SCALES_AS_CONST=true

WORLD_SIZE=8

echo "Using model path: ${FP8_MODEL_PATH}"
echo "Using quantization config: ${QUANT_CONFIG}"
echo "Using scale format: ${scale_format}"
echo "Using patching: ${ENABLE_PATCHING}"
echo "World size: ${WORLD_SIZE}"

# Logging setup
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="inc_quant.${scale_format}.${timestamp}.log"

# Run the Python script
PT_HPU_LAZY_MODE=1 \
    python deepseek_example.py \
    --model "${FP8_MODEL_PATH}" \
    --tokenizer "${FP8_MODEL_PATH}" \
    --osl 32 \
    --max_num_seqs 1 \
    --tp_size "${WORLD_SIZE}" \
    --ep_size "${WORLD_SIZE}" \
    --dummy \
    --max_model_len 8192 2>&1 | tee "${LOG_FILE}"