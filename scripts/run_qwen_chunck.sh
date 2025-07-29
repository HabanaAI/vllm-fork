WORLD_SIZE=1


export VLLM_DISABLE_MARK_SCALES_AS_CONST=1
QUANT_CONFIG_FILE="/mnt/disk3/yiliu4/vllm-fork/scripts/quant_configs/inc_uint_scale.json"



QUANT_CONFIG=${QUANT_CONFIG_FILE} \
PT_HPU_LAZY_MODE=1 \
python run_example_tp_qwen_bk.py \
    --osl 32 \
    --max_num_seqs 1 \
    --max_model_len 2048 \
    --tp_size $WORLD_SIZE \
    --ep_size $WORLD_SIZE


# DEFAULT_MODEL_PATH="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207"
# # DEFAULT_MODEL_PATH="/mnt/disk7/yiliu4/DeepSeek-R1-0528-G2-2nd"
# # DEFAULT_MODEL_PATH=/mnt/disk9/hf_models/Kimi-K2-Instruct-G2/
# # DEFAULT_MODEL_PATH=/mnt/disk8/yiliu7/moonshotai/Kimi-K2-Instruct
# DEFAULT_MODEL_PATH=/mnt/disk5/Kimi-K2-Instruct-G2/

# FP8_MODEL_PATH=$DEFAULT_MODEL_PATH

# QUANT_CONFIG_FILE="/mnt/disk3/yiliu4/vllm-fork/scripts/quant_configs/inc_measure_with_fp8kv_config.json"
# timestamp=$(date +%Y%m%d_%H%M%S)
# LOG_FILE="prepare.pile.512.${timestamp}.log"

# # remove ./scripts/nc_workspace_measure_kvache if needed
# if [ -e ./scripts/nc_workspace_measure_kvache ]; then
#     echo "The directory ./scripts/nc_workspace_measure_kvache already exists, removing it..."
#     rm -rf ./scripts/nc_workspace_measure_kvache
# fi


# echo "============ QUANT_CONFIG file content ==============="
# cat ${QUANT_CONFIG_FILE}
# echo "======================================================"



# echo "Start INC calibration with model ${FP8_MODEL_PATH}, log file ${LOG_FILE}"

# # Usage of this script:
# # bash ./scripts/run_inc_calib.sh --wd 16 --prompts 512


# # Default value for WORLD_SIZE
# WORLD_SIZE=8
# NUM_PROMPTS=512
# # Parse command-line arguments
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         # model
#         --model)
#             FP8_MODEL_PATH=$2
#             shift 2
#             ;;
#         --wd)
#             WORLD_SIZE=$2
#             shift 2
#             ;;
#         --prompts)
#             NUM_PROMPTS=$2
#             shift 2
#             ;;
#         *)
#             echo "Unknown argument: $1"
#             exit 1
#             ;;
#     esac
# done

# export RAY_DEDUP_LOGS=0

# export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
# VLLM_REQUANT_FP8_INC=1 \
# QUANT_CONFIG=${QUANT_CONFIG_FILE} \
# VLLM_LOGGING_LEVEL=DEBUG \
# PT_HPU_LAZY_MODE=1 \
# VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# VLLM_PROMPT_BS_BUCKET_MIN=1 \
# VLLM_PROMPT_BS_BUCKET_MAX=1 \
# VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
# VLLM_PROMPT_SEQ_BUCKET_STEP=512 \
# VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
# VLLM_DECODE_BS_BUCKET_MIN=1 \
# VLLM_DECODE_BS_BUCKET_MAX=1 \
# VLLM_MOE_N_SLICE=1 \
#     python scripts/run_example_tp.py \
#     --model ${FP8_MODEL_PATH} \
#     --tokenizer ${FP8_MODEL_PATH} \
#     --osl 32 \
#     --max_num_seqs 1 \
#     --nprompts $NUM_PROMPTS \
#     --max_model_len 2048 \
#     --tp_size $WORLD_SIZE \
#     --ep_size $WORLD_SIZE \
#     --dataset pile 2>&1 | tee $LOG_FILE
# Translate and improve on the page with DeepL
# Click the icons to open the advanced mode window. Make informed edits and compare versions.