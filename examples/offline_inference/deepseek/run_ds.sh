#!/bin/bash

# Default model path
FP8_MODEL_PATH="/mnt/weka/data/pytorch/DeepSeek-R1/"
FP8_MODEL_PATH="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207"
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
    export RUNTIME_SCALE_PATCHING=1
else
    export RUNTIME_SCALE_PATCHING=0
fi

export CALC_SCALE_WITH_CGUID=0
export VLLM_SKIP_WARMUP=true
# export VLLM_DISABLE_MARK_SCALES_AS_CONST=true

WORLD_SIZE=8
export VLLM_GRAPH_RESERVED_MEM=0.4
echo "Using model path: ${FP8_MODEL_PATH}"
echo "Using quantization config: ${QUANT_CONFIG}"
echo "Using scale format: ${scale_format}"
echo "Using patching: ${ENABLE_PATCHING}"
echo "World size: ${WORLD_SIZE}"
echo "RUNTIME_SCALE_PATCHING: ${RUNTIME_SCALE_PATCHING}"

# Logging setup
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="inc_quant.${scale_format}.${timestamp}.log"

# # Run the Python script
# PT_HPU_LAZY_MODE=1 \
#     python deepseek_example.py \
#     --model "${FP8_MODEL_PATH}" \
#     --tokenizer "${FP8_MODEL_PATH}" \
#     --osl 32 \
#     --max_num_seqs 32 \
#     --tp_size "${WORLD_SIZE}" \
#     --ep_size "${WORLD_SIZE}" \
#     --max_model_len 4096 2>&1 | tee "${LOG_FILE}"



env | grep VLLM

max_model_len=4096
max_num_seqs=32
max_num_batched_tokens=4096

export VLLM_HPU_MARK_SCALES_AS_CONST=false
export VLLM_LOGGING_LEVEL=DEBUG
export PT_HPU_LAZY_MODE=1

export VLLM_EP_SIZE=8
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_DEVICES="ALL"
# os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
# os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
# os.environ["VLLM_EP_SIZE"] = f"{args.ep_size}"
# os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"
# os.environ["PT_HPU_WEIGHT_SHARING"] = "0"


python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8688 \
    --block-size 128 \
    --model $FP8_MODEL_PATH \
    --tokenizer $FP8_MODEL_PATH \
    --device hpu \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --trust-remote-code  \
    --max-model-len $max_model_len \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens  \
    --use-padding-aware-scheduling \
    --use-v2-block-manager \
    --distributed_executor_backend mp \
    --gpu_memory_utilization 0.85 \
    --disable-log-requests \
    --enable-reasoning \
    --enable_expert_parallel \
    --reasoning-parser deepseek_r1  2>&1 | tee "${LOG_FILE}"
