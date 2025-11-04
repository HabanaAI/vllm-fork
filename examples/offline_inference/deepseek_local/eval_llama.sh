DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-3.3-70B-Instruct-FP8_STATIC-0915-G2"
DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-4-Scout-17B-16E-Instruct-FP8_STATIC-916-G2"
DEFAULT_MODEL_PATH="/mnt/disk5/Yi30/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916-G2/"
DEFAULT_MODEL_PATH="/mnt/disk5/meta-llama/Llama-4-Maverick-17B-128E-Instruct"
DEFAULT_MODEL_PATH="/mnt/disk5/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-G2"
# The Acc is good on G3, and have been uploaded to HF
# We need to update the weight scale shape from [out_feats] -> [out_feats, 1]
DEFAULT_MODEL_PATH=/software/users/yiliu7/HF_HOME/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916
DEFAULT_MODEL_PATH=/software/users/yiliu7/HF_HOME/Yi30/Llama-4-Maverick-17B-128E-Instruct-FP8_STATIC-916-Updated-Shape/
# DEFAULT_MODEL_PATH=/software/users/yiliu7/Yi30/Llama-4-Scout-17B-16E-Instruct-FP8_STATIC-916
# !!!!! The mlp.gate was quantized 
# DEFAULT_MODEL_PATH="/software/users/yiliu7/HF_HOME/Yi30/Qwen3-30B-A3B-W-FP8-PCS-A-FP8"
# DEFAULT_MODEL_PATH="/mnt/disk8/Yi30/Llama-4-Scout-17B-16E-Instruct-FP8_STATIC-916"
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


# export VLLM_HPU_FORCE_CHANNEL_FP8=0


# ==-------------------------------------------------------------------------==
# Run basic
# ==-------------------------------------------------------------------------==
VLLM_SUPPORT_MOE_CHUNK=true \
PT_HPU_LAZY_MODE=1 \
    python deepseek_example.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --tp_size $WORLD_SIZE \
    --ep_size $WORLD_SIZE \
    --max_model_len 512 2>&1 | tee $LOG_FILE
    # --fp8_inc \
