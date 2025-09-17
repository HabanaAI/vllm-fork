#! /bin/bash

# set -x

# -----------------------------------------------------------------------------------
## HOW TO USE
#
# 1. Add or modify a model profile in the `set_model_config` function below.
#    Each profile now includes its own performance and bucketing settings.
# 2. Run the script using the profile name:
#    bash start_vllm.sh -m gemma-27b
#
# 3. To override a setting from a profile, use the command-line flags:
#    bash start_vllm.sh -m gemma-27b -l 8192
#
# -----------------------------------------------------------------------------------
pip install datasets
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
# This function contains all model-specific configurations.
set_model_config() {
    local model_profile=$1
    echo "Applying configuration profile: $model_profile"

    # Default platform step size for decode batching calculations
    local default_decode_bs_step=8

    case "$model_profile" in
    "gemma-27b")
        # --- Core Settings ---
<<<<<<< Updated upstream
        model_path="/software/data/pytorch/huggingface/hub/gemma-3-27b-it"
        tp_size=2; block_size=128; max_model_len=16384; max_num_seqs=128
=======
        model_path="/mnt/weka/llm/gemma-3-27b-it"
        tp_size=4; block_size=128; max_model_len=16384; max_num_seqs=128
>>>>>>> Stashed changes
        gpu_mem_util=0.98; limit_mm_per_prompt=150
        EXTRA_VLLM_ARGS=(--limit-mm-per-prompt "image=${limit_mm_per_prompt}" --data-parallel-size "2")

        # --- Bucketing Settings ---
        prompt_bs_min=1; prompt_bs_step=8; prompt_bs_max=$(( max_num_seqs > 64 ? 64 : max_num_seqs ))
        prompt_seq_min=128; prompt_seq_step=128; prompt_seq_max=4096
        decode_bs_min=1; decode_bs_step=$(( max_num_seqs > default_decode_bs_step ? default_decode_bs_step : max_num_seqs )); decode_bs_max=$(( max_num_seqs > 128 ? max_num_seqs : 128 ))
        decode_block_min=128; decode_block_step=4; decode_block_max=1024
        ;;

    "gemma-27b-orig")
        # --- Core Settings ---
        model_path="/workdir/vllm-fork/models/gemma-3-27b-it"
        tp_size=8; block_size=128; max_model_len=16384; max_num_seqs=128
        gpu_mem_util=0.98; limit_mm_per_prompt=150
        EXTRA_VLLM_ARGS=(--limit-mm-per-prompt "image=${limit_mm_per_prompt}")

        # --- Bucketing Settings ---
        prompt_bs_min=1; prompt_bs_step=8; prompt_bs_max=$(( max_num_seqs > 64 ? 64 : max_num_seqs ))
        prompt_seq_min=128; prompt_seq_step=128; prompt_seq_max=4096
        decode_bs_min=1; decode_bs_step=$(( max_num_seqs > default_decode_bs_step ? default_decode_bs_step : max_num_seqs )); decode_bs_max=$(( max_num_seqs > 128 ? max_num_seqs : 128 ))
        decode_block_min=128; decode_block_step=4; decode_block_max=1024
        ;;

    "g318b")
        # --- Core Settings ---
        model_path="/workdir/vllm-fork/models/granite-3.1-8b-instruct"
        tp_size=1; block_size=128; max_model_len=16384; max_num_seqs=128
        gpu_mem_util=0.98
        EXTRA_VLLM_ARGS=()

        # --- Bucketing Settings ---
        prompt_bs_min=1; prompt_bs_step=8; prompt_bs_max=$(( max_num_seqs > 64 ? 64 : max_num_seqs ))
        prompt_seq_min=128; prompt_seq_step=128; prompt_seq_max=4096
        decode_bs_min=1; decode_bs_step=$(( max_num_seqs > default_decode_bs_step ? default_decode_bs_step : max_num_seqs )); decode_bs_max=$(( max_num_seqs > 128 ? max_num_seqs : 128 ))
        decode_block_min=128; decode_block_step=4; decode_block_max=1024
        ;;

    "qwen2.5-vl-3b")
        # --- Core Settings ---
        model_path="/workdir/vllm-fork/models/Qwen2.5-VL-3B-Instruct"
        tp_size=1; block_size=128; max_model_len=16384; max_num_seqs=256
        gpu_mem_util=0.90; limit_mm_per_prompt=150
        EXTRA_VLLM_ARGS=(--limit-mm-per-prompt "image=${limit_mm_per_prompt}")

        # --- Bucketing Settings ---
        prompt_bs_min=1; prompt_bs_step=16; prompt_bs_max=128
        prompt_seq_min=256; prompt_seq_step=256; prompt_seq_max=8192
        decode_bs_min=1; decode_bs_step=$(( max_num_seqs > default_decode_bs_step ? default_decode_bs_step : max_num_seqs )); decode_bs_max=256
        decode_block_min=64; decode_block_step=8; decode_block_max=512
        ;;

    "llama4-scout")
        # --- Core Settings ---
        model_path="/workdir/vllm-fork/models/Llama-4-Scout-17B-16E-Instruct"
        tp_size=2; block_size=128; max_model_len=8192; max_num_seqs=64
        gpu_mem_util=0.80; limit_mm_per_prompt=10
        EXTRA_VLLM_ARGS=(--limit-mm-per-prompt "image=${limit_mm_per_prompt}")

        # --- Bucketing Settings ---
        prompt_bs_min=1; prompt_bs_step=4; prompt_bs_max=32
        prompt_seq_min=512; prompt_seq_step=512; prompt_seq_max=4096
        decode_bs_min=1; decode_bs_step=$(( max_num_seqs > default_decode_bs_step ? default_decode_bs_step : max_num_seqs )); decode_bs_max=64
        decode_block_min=128; decode_block_step=16; decode_block_max=1024
        ;;

    "phi4-reasoning-plus")
        # --- Core Settings ---
        model_path="/workdir/vllm-fork/models/Phi-4-reasoning-plus"
        tp_size=1; block_size=128; max_model_len=32768; max_num_seqs=64
        gpu_mem_util=0.80; limit_mm_per_prompt=10
        EXTRA_VLLM_ARGS=(--enable-reasoning --reasoning-parser deepseek_r1)

        # --- Bucketing Settings ---
        prompt_bs_min=1; prompt_bs_step=4; prompt_bs_max=32
        prompt_seq_min=512; prompt_seq_step=512; prompt_seq_max=4096
        decode_bs_min=1; decode_bs_step=$(( max_num_seqs > default_decode_bs_step ? default_decode_bs_step : max_num_seqs )); decode_bs_max=64
        decode_block_min=128; decode_block_step=16; decode_block_max=1024
        ;;

    *)
        echo "Error: Unknown model profile '$model_profile'."
        Help
        exit 1
        ;;
    esac

    # Automatically set cache path based on model path
    warmup_cache_path="$model_path/.recipecache"

    # Export all the bucketing variables for vLLM to use
    export VLLM_PROMPT_BS_BUCKET_MIN=${prompt_bs_min}
    export VLLM_PROMPT_BS_BUCKET_STEP=${prompt_bs_step}
    export VLLM_PROMPT_BS_BUCKET_MAX=${prompt_bs_max}
    export VLLM_PROMPT_SEQ_BUCKET_MIN=${prompt_seq_min}
    export VLLM_PROMPT_SEQ_BUCKET_STEP=${prompt_seq_step}
    export VLLM_PROMPT_SEQ_BUCKET_MAX=${prompt_seq_max}
    export VLLM_DECODE_BS_BUCKET_MIN=${decode_bs_min}
    export VLLM_DECODE_BS_BUCKET_STEP=${decode_bs_step}
    export VLLM_DECODE_BS_BUCKET_MAX=${decode_bs_max}
    export VLLM_DECODE_BLOCK_BUCKET_MIN=${decode_block_min}
    export VLLM_DECODE_BLOCK_BUCKET_STEP=${decode_block_step}
    export VLLM_DECODE_BLOCK_BUCKET_MAX=${decode_block_max}
}

Help() {
    echo "Start vllm server for a huggingface model on Gaudi."
    echo
    echo "Syntax: bash start_vllm.sh -m <model_profile> [options]"
    echo "options:"
    echo "m  Model profile name to use (e.g., gemma-27b, llama3-70b). Required."
    echo "w  Override model weights path from profile."
    echo "u  URL of the server (default: 0.0.0.0)."
    echo "p  Port for the server (default: 8688)."
    echo "l  Override max_model_len from profile."
    echo "b  Override max_num_seqs (batch size) from profile."
    echo "z  Override tensor_parallel_size from profile."
    echo "c  Override HPU recipe cache path."
    echo "s  Skip warmup (default: false)."
    echo "t  Enable profiling (default: false)."
    echo "h  Display this Help info."
    echo
}

# --- Default values ---
host="0.0.0.0"
vllm_port=8688
skip_warmup=false
profile_enabled=false
model_profile=""

# --- Argument Parsing ---
while getopts "hm:w:u:p:l:b:z:c:st" flag; do
    case ${flag} in
    m) model_profile=$OPTARG ;;
    esac
done
OPTIND=1

if [ -z "$model_profile" ]; then
    echo "Error: Model profile is required. Please use the -m option."
    Help
    exit 1
fi

# Load the configuration from the selected profile
set_model_config "$model_profile"

# Now, parse all arguments again to allow overrides
while getopts "hm:w:u:p:l:b:z:st" flag; do
    case ${flag} in
    h) Help; exit ;;
    m) ;; # Already handled
    w) model_path=$OPTARG ;;
    u) host=$OPTARG ;;
    p) vllm_port=$OPTARG ;;
    l) max_model_len=$OPTARG ;;
    b) max_num_seqs=$OPTARG ;;
    z) tp_size=$OPTARG ;;
    s) skip_warmup=true ;;
    t) profile_enabled=true ;;
    \?) echo "Error: Invalid option"; Help; exit 1 ;;
    esac
done
#     c) warmup_cache_path=$OPTARG ;;
# --- Environment and System Setup ---
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

<<<<<<< Updated upstream
#if [ -n "$warmup_cache_path" ]; then
#    echo "HPU recipe cache path: $warmup_cache_path"
#    mkdir -p "${warmup_cache_path}"
#
#    if [ -z "$(ls -A "$warmup_cache_path")" ]; then
#        echo "Cache directory is empty. Running with RECIPE_CACHE_DELETE=True (rebuild)."
#        export PT_HPU_RECIPE_CACHE_CONFIG=${warmup_cache_path},True,16384
#    else
#        echo "Cache directory already has recipes. Running with RECIPE_CACHE_DELETE=False (reuse)."
#        export PT_HPU_RECIPE_CACHE_CONFIG=${warmup_cache_path},False,16384
#    fi
#fi
=======
# if [ -n "$warmup_cache_path" ]; then
#     echo "HPU recipe cache path: $warmup_cache_path"
#     mkdir -p "${warmup_cache_path}"

#     if [ -z "$(ls -A "$warmup_cache_path")" ]; then
#         echo "Cache directory is empty. Running with RECIPE_CACHE_DELETE=True (rebuild)."
#         export PT_HPU_RECIPE_CACHE_CONFIG=${warmup_cache_path},True,16384
#     else
#         echo "Cache directory already has recipes. Running with RECIPE_CACHE_DELETE=False (reuse)."
#         export PT_HPU_RECIPE_CACHE_CONFIG=${warmup_cache_path},False,16384
#     fi
# fi
>>>>>>> Stashed changes
if [ "$skip_warmup" = "true" ]; then
    echo "VLLM_SKIP_WARMUP is set to True"
    export VLLM_SKIP_WARMUP=True
fi
if [ "$profile_enabled" = "true" ]; then
    echo "Enabling profiling..."
    export VLLM_TORCH_PROFILER_DIR="$BASH_DIR/profiler_dir_gemma"
    # hl-prof-config --use-template profile_api --hw-trace off
    hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on
    export HABANA_PROFILE=1
fi

ray stop --force

# --- Key Habana/vLLM Environment Variables ---
export HABANA_VISIBLE_DEVICES="0,2,3,5"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
# export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export PT_HPU_LAZY_MODE=1
export VLLM_DELAYED_SAMPLING="true"
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0
export VLLM_MLA_DISABLE_REQUANTIZATION=0
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_ENABLE_ASYNC_MM_PREPROCESS=1
export VLLM_MAX_CONCURRENT_PREPROC=25
max_num_batched_tokens=$max_model_len
export VLLM_GPU_MEMORY_UTILIZATION=$gpu_mem_util
export VLLM_USE_V1=0
export VLLM_RPC_TIMEOUT=3000000
export VLLM_MERGED_PREFILL="false"

echo "--- Launching vLLM Server with the following settings ---"
echo "Model Profile: $model_profile"
echo "Model Path: $model_path"
echo "Tensor Parallel Size: $tp_size, Max Model Length: $max_model_len, Max Num Seqs: $max_num_seqs"
echo "GPU Memory Utilization: $gpu_mem_util"
echo "Extra vLLM Args: ${EXTRA_VLLM_ARGS[*]}"
echo "Bucketing variables have been set."
echo "---------------------------------------------------------"
if [ ${#EXTRA_VLLM_ARGS[@]} -gt 0 ]; then
    VLLM_SERVER_ARGS+=("${EXTRA_VLLM_ARGS[@]}")
fi
# env | grep VLLM # Uncomment to debug all VLLM environment variables

export VLLM_USE_V1=0
export EXPERIMENTAL_WEIGHT_SHARING=0
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_SDPA_QKV_SLICE_MODE_FWD=1
export ENABLE_SCALAR_LIKE_FULL_FUSION=1
export ENABLE_FUSION_BEFORE_NORM=true
export FUSER_ENABLE_LOW_UTILIZATION=true
export VLLM_ENABLE_ASYNC_MM_PREPROCESS=1
export VLLM_MAX_CONCURRENT_PREPROC=25
export VLLM_HPU_FORCE_MARK_STEP=false
export VLLM_DETOKENIZE_ON_OPENAI_SERVER=true
export VLLM_EXPONENTIAL_BUCKETING=False
export VLLM_MULTIMODAL_BUCKETS="1,4,6,8"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_SUSPEND_PROMPT=1
export VLLM_GRAPH_RESERVED_MEM=0.8
export VLLM_GRAPH_PROMPT_RATIO=0.3
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=4
export VLLM_PROMPT_SEQ_BUCKET_MIN=8192
export VLLM_PROMPT_SEQ_BUCKET_STEP=1024
export VLLM_PROMPT_SEQ_BUCKET_MAX=10240
export VLLM_DECODE_BLOCK_BUCKET_MIN=72
export VLLM_DECODE_BLOCK_BUCKET_STEP=128
export VLLM_DECODE_BLOCK_BUCKET_MAX=1920

# VLLM_USE_V1=0 EXPERIMENTAL_WEIGHT_SHARING=0 PT_HPU_LAZY_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=true  PT_HPU_SDPA_QKV_SLICE_MODE_FWD=1 ENABLE_SCALAR_LIKE_FULL_FUSION=1 ENABLE_FUSION_BEFORE_NORM=true FUSER_ENABLE_LOW_UTILIZATION=true VLLM_ENABLE_ASYNC_MM_PREPROCESS=1 VLLM_MAX_CONCURRENT_PREPROC=25 VLLM_HPU_FORCE_MARK_STEP=false VLLM_DETOKENIZE_ON_OPENAI_SERVER=true VLLM_EXPONENTIAL_BUCKETING=False VLLM_MULTIMODAL_BUCKETS="1,4,6,8" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_SUSPEND_PROMPT=1 VLLM_GRAPH_RESERVED_MEM=0.8 VLLM_GRAPH_PROMPT_RATIO=0.3 VLLM_PROMPT_BS_BUCKET_MIN=1 VLLM_PROMPT_BS_BUCKET_STEP=1 VLLM_PROMPT_BS_BUCKET_MAX=4 VLLM_PROMPT_SEQ_BUCKET_MIN=8192 VLLM_PROMPT_SEQ_BUCKET_STEP=1024 VLLM_PROMPT_SEQ_BUCKET_MAX=10240 VLLM_DECODE_BLOCK_BUCKET_MIN=72 VLLM_DECODE_BLOCK_BUCKET_STEP=128 VLLM_DECODE_BLOCK_BUCKET_MAX=1920
# --- Launch vLLM Server ---
set -x
python3 -m vllm.entrypoints.openai.api_server \
    --host "$host" \
    --port "$vllm_port" \
    --model "$model_path" \
    --tensor-parallel-size "$tp_size" \
    --block-size "$block_size" \
    --max-model-len "$max_model_len" \
    --max-num-seqs "$max_num_seqs" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --gpu-memory-utilization "$gpu_mem_util" \
    --use-padding-aware-scheduling \
    --device hpu \
    --dtype bfloat16 \
    --trust-remote-code \
    --use-v2-block-manager \
    --disable-log-requests \
    "${EXTRA_VLLM_ARGS[@]}" 2>&1 | tee vllm_serving.log &
# PT_HPU_RECIPE_CACHE_CONFIG=/tmp/gemma27,False,1024,False PT_HPU_RECIPE_CACHE_CONFIG="/ws/vllm-fork/recipe,False,4096"
# VLLM_USE_V1=0 EXPERIMENTAL_WEIGHT_SHARING=0 PT_HPU_LAZY_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=true  PT_HPU_SDPA_QKV_SLICE_MODE_FWD=1 ENABLE_SCALAR_LIKE_FULL_FUSION=1 ENABLE_FUSION_BEFORE_NORM=true FUSER_ENABLE_LOW_UTILIZATION=true VLLM_ENABLE_ASYNC_MM_PREPROCESS=1 VLLM_MAX_CONCURRENT_PREPROC=25 VLLM_HPU_FORCE_MARK_STEP=false VLLM_DETOKENIZE_ON_OPENAI_SERVER=true VLLM_EXPONENTIAL_BUCKETING=False VLLM_MULTIMODAL_BUCKETS="1,4,6,8" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_SUSPEND_PROMPT=1 VLLM_GRAPH_RESERVED_MEM=0.8 VLLM_GRAPH_PROMPT_RATIO=0.3 VLLM_PROMPT_BS_BUCKET_MIN=1 VLLM_PROMPT_BS_BUCKET_STEP=1 VLLM_PROMPT_BS_BUCKET_MAX=4 VLLM_PROMPT_SEQ_BUCKET_MIN=8192 VLLM_PROMPT_SEQ_BUCKET_STEP=1024 VLLM_PROMPT_SEQ_BUCKET_MAX=10240 VLLM_DECODE_BLOCK_BUCKET_MIN=72 VLLM_DECODE_BLOCK_BUCKET_STEP=128 VLLM_DECODE_BLOCK_BUCKET_MAX=1920 numactl --cpunodebind=0 --membind=0 python3 -m vllm.entrypoints.openai.api_server --host "$host" --port "$vllm_port" --block-size 128 --model "$model_path" --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 20480 --max-num-batched-tokens 20480 --gpu_memory_utilization 0.7 --limit-mm-per-prompt image=6 --max-num-prefill-seqs 4 --max-num-seqs 64 --split_qkv --disable-log-requests --disable-log-stats --max-num-batched-tokens 40960 2>&1 | tee log_27.txt &
 
set +x
pid=$(($! - 1))

    # --max-num-prefill-seqs 8 \

echo "Waiting for server to start..."
until grep -q "Started server process" vllm_serving.log; do
    sleep 5s
done
echo "Server is ready. PID: ${pid}"

    # --distributed-executor-backend ray \
# Optional: run a benchmark script after launch 
    #     --tensor-parallel-size "8" \
./online-multi-image-benchmark.sh 2>&1 | tee client.log
