#!/bin/bash

# Default values
WORLD_SIZE=8
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



QUANT_CONFIG_FILE="./scripts/quant_configs/inc_measure_with_fp8kv_config.json"
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

PT_HPU_LAZY_MODE=1 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=1 \
VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
VLLM_PROMPT_SEQ_BUCKET_STEP=512 \
VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
VLLM_DECODE_BS_BUCKET_MIN=1 \
VLLM_DECODE_BS_BUCKET_MAX=1 \
QUANT_CONFIG=${QUANT_CONFIG_FILE} \
    python scripts/run_example_tp.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --nprompts ${NUM_PROMPTS} \
    --max_model_len 2048 \
    --tp_size $WORLD_SIZE \
    --ep_size 1 \
    --dataset pile 2>&1 | tee $LOG_FILE