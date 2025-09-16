#! /bin/bash

pkill -9 python

DEFAULT_MODEL_PATH="/mnt/disk3/DeepSeek-R1-G2"
DEFAULT_MODEL_PATH="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207"
DEFAULT_MODEL_PATH="/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air/"
DEFAULT_MODEL_PATH="/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air-FP8/"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-3.2-1B-Instruct-W-FP8-PCS-A-FP8"
DEFAULT_MODEL_PATH="/mnt/disk8/RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Qwen3-30B-A3B-W-FP8-PCS-A-FP8"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Qwen3-30B-A3B-W-FP8-PCS-A-FP8-G2"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Qwen3-30B-A3B-W-FP8-PCS-A-FP8-G2-2nd"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Yi30/Qwen3-15B-A2B-Base-W8-A8-PCS-912-G2-2nd"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Qwen3-15B-A2B-Base-W8-A8-PCS-calib512-913"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Qwen3-15B-A2B-Base-W8-A8-PCS-calib512-913-G2"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-3.3-70B-Instruct-FP8_STATIC-0915"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-3.3-70B-Instruct-FP8_STATIC-0915-G2"
DEFAULT_MODEL_PATH="/mnt/disk8/Qwen/Qwen3-8B/"
DEFAULT_MODEL_PATH="/mnt/disk8/meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_MODEL_PATH="/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air-FP8-G2"
# DEFAULT_MODEL_PATH=/mnt/disk5/Qwen3-30B-A3B-FP8-G2/
# DEFAULT_MODEL_PATH="/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air-FP8-G2/"
# DEFAULT_MODEL_PATH="/mnt/disk6/yiliu4/deepseek-ai/DeepSeek-R1-0528"
FP8_MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

# export CALC_SCALE_WITH_CGUID=1
# export QUANT_CONFIG="./quant_configs/dynamic_quant_config.json"
# export QUANT_CONFIG="./quant_configs/unit_quant_config.json"
export VLLM_SKIP_WARMUP=true

WORLD_SIZE=1
WORLD_SIZE=8

# if 1, set VLLM_NUM_LAYERS to 4
if [ $WORLD_SIZE -eq 1 ]; then
    export VLLM_NUM_LAYERS=4
else
    unset VLLM_NUM_LAYERS
fi

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="dynamic_quant.inc.${timestamp}.log"

export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=1
export PT_HPU_METRICS_GC_DETAILS=1

export VLLM_DISABLE_MARK_SCALES_AS_CONST=1

export VLLM_HPU_CONVERT_TO_FP8UZ=0
# VLLM_HPU_FORCE_CHANNEL_FP8=1 \

# export VLLM_HPU_CONVERT_TO_FP8UZ=1
# export VLLM_HPU_FORCE_CHANNEL_FP8=0

export QUANT_CONFIG=./quant_configs/inc_measure.json
# export QUANT_CONFIG=./quant_configs/unit_quant_config.json 
# export  QUANT_CONFIG="./quant_configs/inc_quant.json"
# export  QUANT_CONFIG="./quant_configs/inc_quant_naive.json"
# export  QUANT_CONFIG="./quant_configs/inc_quant_fp8kv.json"
# VLLM_HPU_CONVERT_TO_FP8UZ=1 \
# VLLM_SUPPORT_MOE_CHUNK=true \

PT_HPU_LAZY_MODE=1 \
    python deepseek_example.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --tp_size $WORLD_SIZE \
    --ep_size $WORLD_SIZE \
    --dataset pile \
    --nprompts 512 \
    --max_model_len 512 2>&1 | tee $LOG_FILE
    # --fp8_inc \
    
