FP8_MODEL_PATH="/mnt/disk3/DeepSeek-R1-G2-INC"

QUANT_CONFIG_FILE="./scripts/quant_configs/inc_measure_with_fp8kv_config.json"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="prepare.pile.512.${timestamp}.log"

#!/bin/bash

# Default value for WORLD_SIZE
WORLD_SIZE=8
NUM_PROMPTS=512
FP8_MODEL_PATH=""

# Help function to display usage information
function show_help() {
    echo "Usage: bash run_inc_calib.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --wd <world_size>       Number of processes/devices for distributed computation (default: 8)"
    echo "  --nprompts <num_prompts> Number of prompts for calibration (default: 512)"
    echo "  --model <model_path>    Path to the FP8 model for calibration"
    echo "  --help                  Display this help message"
    echo
    echo "Examples:"
    echo "  bash run_inc_calib.sh --wd 16 --nprompts 512 --model /path/to/model"
    echo "  bash run_inc_calib.sh --help"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wd)
            WORLD_SIZE=$2
            shift 2
            ;;
        --nprompts)
            NUM_PROMPTS=$2
            shift 2
            ;;
        --model)
            FP8_MODEL_PATH=$2
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

# Print the parsed values for debugging
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NUM_PROMPTS: $NUM_PROMPTS"
echo "FP8_MODEL_PATH: $FP8_MODEL_PATH"

# remove ./scripts/nc_workspace_measure_kvcache if needed
if [ -e ./scripts/nc_workspace_measure_kvache ]; then
    echo "The directory ./scripts/nc_workspace_measure_kvache already exists, removing it..."
    rm -rf ./scripts/nc_workspace_measure_kvache
fi


echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"



echo "Start INC calibration with model ${FP8_MODEL_PATH}, log file ${LOG_FILE}"

export RAY_DEDUP_LOGS=0
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1

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
QUANT_CONFIG=${QUANT_CONFIG_FILE} \
    python scripts/run_example_tp.py \
    --model ${FP8_MODEL_PATH} \
    --tokenizer ${FP8_MODEL_PATH} \
    --osl 32 \
    --max_num_seqs 1 \
    --nprompts 512 \
    --max_model_len 2048 \
    --dataset pile 2>&1 | tee $LOG_FILE
