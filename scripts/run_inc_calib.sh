#!/bin/bash

# Default values
WORLD_SIZE=1
NUM_PROMPTS=16
FP8_MODEL_PATH="/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-20b-bf16"
FP8_MODEL_PATH="/mnt/disk5/lmsys/gpt-oss-20b-bf16"

# Help function to display usage information
function show_help() {
    echo "Usage: bash run_inc_calib.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --model <model_path>    Path to the FP8 model for calibration"
    echo "  --wd <world_size>       Number of devices (default: 8)"
    echo "  --nprompts <num_prompts> Number of prompts for calibration (default: 512)"
    echo "  --help                  Display this help message"
    echo
    echo "Examples:"
    echo "  bash run_inc_calib.sh --wd 8 --nprompts 512 --model /path/to/model"
    echo "  bash run_inc_calib.sh --help"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            FP8_MODEL_PATH=$2
            shift 2
            ;;
        --wd)
            WORLD_SIZE=$2
            shift 2
            ;;
        --nprompts)
            NUM_PROMPTS=$2
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done



QUANT_CONFIG_FILE="quant_configs/inc_measure_with_fp8kv_config.json"
QUANT_CONFIG_FILE="quant_configs/inc_quant.json"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="prepare.pile.${NUM_PROMPTS}.${timestamp}.log"


# remove ./scripts/nc_workspace_measure_kvcache if needed
if [ -e ./scripts/nc_workspace_measure_kvcache ]; then
    echo "The directory ./scripts/nc_workspace_measure_kvcache already exists, removing it..."
    rm -rf ./scripts/nc_workspace_measure_kvcache
fi


echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NUM_PROMPTS: $NUM_PROMPTS"
echo "FP8_MODEL_PATH: $FP8_MODEL_PATH"

echo "Start INC calibration with model ${FP8_MODEL_PATH}, log file ${LOG_FILE}"


# VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# VLLM_REQUANT_FP8_INC=1 \
# VLLM_MOE_N_SLICE=1 \

# PT_HPU_LAZY_MODE=1 \
# VLLM_PROMPT_BS_BUCKET_MIN=1 \
# VLLM_PROMPT_BS_BUCKET_MAX=1 \
# VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
# VLLM_PROMPT_SEQ_BUCKET_STEP=512 \
# VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
# VLLM_DECODE_BS_BUCKET_MIN=1 \
# VLLM_DECODE_BS_BUCKET_MAX=1 \

VLLM_DISABLE_MARK_SCALES_AS_CONST=0 INC_PT_ONLY=1 PT_HPU_WEIGHT_SHARING="0" \
PT_HPU_LAZY_MODE=1  EXPERIMENTAL_WEIGHT_SHARING=0  OPENAI_API_BASE=http://localhost:8080/v1  \
OPENAI_API_KEY=secret_abcdefg  PT_HPU_ENABLE_FUSED_SDPA_SINK=1  PT_HPU_ENABLE_LAZY_COLLECTIVES=true  \
PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128  PT_HPU_SDPA_QKV_SLICE_MODE_FWD=1  VLLM_PROMPT_USE_FUSEDSDPA=1 \
PT_HPU_GPT_MOE_WT_INTERLEAVED=0  VLLM_DECODE_BLOCK_BUCKET_MAX=4224  VLLM_DECODE_BLOCK_BUCKET_MIN=2048  \
VLLM_DECODE_BS_BUCKET_MAX=128  VLLM_DECODE_BS_BUCKET_MIN=128  VLLM_DECODE_BS_BUCKET_STEP=128  VLLM_ENGINE_ITERATION_TIMEOUT_S=3600  \
VLLM_EXPONENTIAL_BUCKETING=false  VLLM_PROMPT_BS_BUCKET_MAX=16  VLLM_PROMPT_SEQ_BUCKET_MAX=2048  VLLM_PROMPT_SEQ_BUCKET_MIN=2048  \
VLLM_RPC_TIMEOUT=600000  VLLM_SKIP_WARMUP=false \
QUANT_CONFIG=${QUANT_CONFIG_FILE} \
    python run_example_tp.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --max_model_len 2048 \
    --osl 32 \
    --tp_size $WORLD_SIZE >&1 | tee $LOG_FILE