#!/bin/bash
QUANT_CONFIG_FILE="./quant_configs/inc_unit_scale.json"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="quant.pile.512.${timestamp}.log"



echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"



model_path="/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-20b-bf16"

# model_path="/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-120b-bf16"

tp_size=1
ep_size=1
basename=$(basename $model_path)

export VLLM_BUILD=1.23.0.248
export QUANT_CONFIG=./quant_configs/inc_unit_scale.json
# export QUANT_CONFIG=./quant_configs/inc_quant.json
# export QUANT_CONFIG=./quant_configs/inc_measure.json

nprompts=512
nprompts=4
tp_size=1
ep_size=1
echo "QUANT_CONFIG=${QUANT_CONFIG}"
# HABANA_LOGS=/software/users/yiliu4/gpt-oss/20b-oom/habana-logs LOG_LEVEL_DATA_TYPE=0 LOG_LEVEL_PASS_MANAGER=0 LOG_LEVEL_QUANT=0 LOG_LEVEL_PT_STATS=0 LOG_LEVEL_INC=0 \
# HABANA_LOGS=/software/users/yiliu4/gpt-oss/20b-oom/habana-logs-full LOG_LEVEL_ALL=0 \
VLLM_ENABLE_FUSED_MOE_WITH_BIAS=1 \
VLLM_DISABLE_MARK_SCALES_AS_CONST=1 \
VLLM_LOGGING_LEVEL=DEBUG \
PT_HPU_LAZY_MODE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=0 \
VLLM_SKIP_WARMUP=true  python run_example_tp.py \
    --tp_size $tp_size \
    --ep_size $ep_size \
    --model $model_path \
    --osl 32 \
    --max_num_seqs 1 \
    --max_model_len 8192 \
    --tokenizer $model_path \
    --fp8_kv_cache \
    --nprompts $nprompts 2>&1 | tee $LOG_FILE



