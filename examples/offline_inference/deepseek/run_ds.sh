#! /usr/bin

DEFAULT_MODEL_PATH="/mnt/disk3/DeepSeek-R1-G2"
DEFAULT_MODEL_PATH="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207"
FP8_MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="dynamic_quant.inc.${timestamp}.log"

export QUANT_CONFIG="./quant_configs/dynamic_quant_config.json"
export VLLM_SKIP_WARMUP=true

WORLD_SIZE=8
# WORLD_SIZE=1

PT_HPU_LAZY_MODE=1  \
python deepseek_example.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --tp_size $WORLD_SIZE \
    --ep_size $WORLD_SIZE \
    --max_model_len 2048 2>&1 | tee $LOG_FILE
