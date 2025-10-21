#!/bin/bash

# Default values
WORLD_SIZE=8
FP8_MODEL_PATH="/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air-FP8-G2"


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



QUANT_CONFIG_FILE="$(realpath ./scripts/quant_configs/inc_quant_config.json)"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="prepare.pile.${NUM_PROMPTS}.${timestamp}.log"





echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NUM_PROMPTS: $NUM_PROMPTS"
echo "FP8_MODEL_PATH: $FP8_MODEL_PATH"


VLLM_HPU_CONVERT_TO_FP8UZ=0 \
PT_HPU_LAZY_MODE=1 \
VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
VLLM_ENABLE_RUNTIME_DEQUANT=1 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=1 \
VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
VLLM_PROMPT_SEQ_BUCKET_STEP=512 \
VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
VLLM_DECODE_BS_BUCKET_MIN=1 \
VLLM_DECODE_BS_BUCKET_MAX=1 \
VLLM_REQUANT_FP8_INC=1 \
VLLM_MOE_N_SLICE=1 \
VLLM_DISABLE_MARK_SCALES_AS_CONST=1 \
QUANT_CONFIG=${QUANT_CONFIG_FILE} \
    python scripts/run_example_tp.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --tp_size ${WORLD_SIZE} \
    --ep_size ${WORLD_SIZE} \
    --fp8_kv_cache \
    --max_model_len 2048  2>&1 | tee $LOG_FILE