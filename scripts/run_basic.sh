#!/bin/bash
QUANT_CONFIG_FILE="./quant_configs/inc_unit_scale.json"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="quant.pile.512.${timestamp}.log"

# remove ./scripts/nc_workspace_measure_kvache if needed
if [ -e ./scripts/nc_workspace_measure_kvache ]; then
    echo "The directory ./scripts/nc_workspace_measure_kvache already exists, removing it..."
    rm -rf ./scripts/nc_workspace_measure_kvache
fi


echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"



echo "Start INC calibration with model ${FP8_MODEL_PATH}, log file ${LOG_FILE}"

model_path="/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-20b-bf16"

export QUANT_CONFIG=./quant_configs/inc_unit_scale.json
export QUANT_CONFIG=./quant_configs/inc_measure.json
# export QUANT_CONFIG=./quant_configs/inc_quant.json
export VLLM_BUILD=1.23.0.248
# QUANT_CONFIG=${QUANT_CONFIG_FILE} \
# VLLM_BUILD=1.23.0.248 \
tp_size=8
ep_size=8
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
    --dataset pile \
    --max_num_seqs 1 \
    --max_model_len 2048 \
    --tokenizer $model_path \
    --nprompts 512 2>&1 | tee $LOG_FILE
# VLLM_SKIP_WARMUP=true  python basic.py --tp_size $tp_size --ep_size $ep_size 