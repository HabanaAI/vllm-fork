#! /usr/bin

DEFAULT_MODEL_PATH="/mnt/disk3/DeepSeek-R1-G2"
DEFAULT_MODEL_PATH="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207"
DEFAULT_MODEL_PATH="/mnt/weka/data/pytorch/DeepSeek-R1/"
FP8_MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

export CALC_SCALE_WITH_CGUID=1
# export QUANT_CONFIG="./quant_configs/dynamic_quant_config.json"
export QUANT_CONFIG="./quant_configs/unit_quant_config.json"
export VLLM_SKIP_WARMUP=true

WORLD_SIZE=8
# WORLD_SIZE=1

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="dynamic_quant.inc.${timestamp}.log"

PT_HPU_LAZY_MODE=1 \
    python deepseek_example.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --tp_size $WORLD_SIZE \
    --ep_size $WORLD_SIZE \
    --dummy \
    --max_model_len 2048 2>&1 | tee $LOG_FILE
