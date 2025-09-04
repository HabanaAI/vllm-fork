#! /bin/bash

# set -x

original_env=( $(env) )

# get device name
device=$(hl-smi -Q name -f csv | tail -n 1)
available_modules=($(hl-smi -Q module_id -f csv | tail -n +2 | sort | tr '\n' ' '))

# HPU specific constants
BLOCK_SIZE=128
PREFERED_BATCHED_TOKENS=8192
PREFERED_NUM_SEQS=128
PREFERED_SEQ_LEN_TO_CAPTURE=8192
DATA_TYPE="bfloat16"

# ceiling an integer with specified base
ceil() {
    local num=$1
    local base=$2
    if ! [[ $num =~ ^[0-9]+$ ]] || ! [[ $base =~ ^[0-9]+$ ]]; then
        echo "Error: Both arguments must be integers."
        return 1
    fi
    echo $(( (num + base - 1) / base * base ))
}

# ceil_div of two integers
ceil_div() {
    local num=$1
    local div=$2
    if ! [[ $num =~ ^[0-9]+$ ]] || ! [[ $div =~ ^[0-9]+$ ]]; then
        echo "Error: Both arguments must be integers."
        return 1
    fi
    echo $(( (num + div - 1) / div ))
}

# get min of two integers
min() {
    local a=$1
    local b=$2
    if ! [[ $a =~ ^-?[0-9]+$ ]] || ! [[ $b =~ ^-?[0-9]+$ ]]; then
        echo "Error: Both arguments must be integers."
        return 1
    fi
    if [ "$a" -lt "$b" ]; then
        echo "$a"
    else
        echo "$b"
    fi
}

# get max of two integers
max() {
    local a=$1
    local b=$2
    if ! [[ $a =~ ^-?[0-9]+$ ]] || ! [[ $b =~ ^-?[0-9]+$ ]]; then
        echo "Error: Both arguments must be integers."
        return 1
    fi
    if [ "$a" -gt "$b" ]; then
        echo "$a"
    else
        echo "$b"
    fi
}

# set up common environment variables for vllm
set_common_env(){
    # pytorch bridge
    if [ "$num_hpu" -gt 1 ]; then
        export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
    fi

    # memory usage tuning
    export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-"0.9"}
    export VLLM_GRAPH_RESERVED_MEM=${VLLM_GRAPH_RESERVED_MEM:-"0.2"}
    export VLLM_GRAPH_PROMPT_RATIO=${VLLM_GRAPH_PROMPT_RATIO:-"0.8"}

    # performance tuning
    export VLLM_DELAYED_SAMPLING=${VLLM_DELAYED_SAMPLING:-"true"}
    export VLLM_ZERO_PADDING=${VLLM_ZERO_PADDING:-"true"}

    # MoE specific
    export VLLM_EP_SIZE=${VLLM_EP_SIZE:-"${num_hpu}"}
    export VLLM_DYNAMIC_MOE_MIN_TOKENS=${VLLM_DYNAMIC_MOE_MIN_TOKENS:-"256"}
    export VLLM_DYNAMIC_MOE_MIN_EXPERTS_SINGLEHPU=${VLLM_DYNAMIC_MOE_MIN_EXPERTS_SINGLEHPU:-"32"}

    # profiler
    export VLLM_PROFILER_ENABLED=${VLLM_PROFILER_ENABLED:-"false"}
    export VLLM_ENGINE_PROFILER_ENABLED=${VLLM_ENGINE_PROFILER_ENABLED:-"false"}
    export VLLM_ENGINE_PROFILER_WARMUP_STEPS=${VLLM_ENGINE_PROFILER_WARMUP_STEPS:-"0"}
    export VLLM_ENGINE_PROFILER_STEPS=${VLLM_ENGINE_PROFILER_STEPS:-"1"}
    export VLLM_ENGINE_PROFILER_REPEAT=${VLLM_ENGINE_PROFILER_REPEAT:-"1"}

    # network
    default_host_ip=$( hostname -I | awk '{print $1}' )
    default_ifname=$( ip -br addr show to ${default_host_ip} | awk '{print $1}' )
    export VLLM_HOST_IP=${VLLM_HOST_IP:-"${default_host_ip}"}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"${default_ifname}"}
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-"${default_ifname}"}

    # misc
    export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-"spawn"}
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"true"}
    export RAY_IGNORE_UNHANDLED_ERRORS=${RAY_IGNORE_UNHANDLED_ERRORS:-"1"}
    export VLLM_RAY_DISABLE_LOG_TO_DRIVER=${VLLM_RAY_DISABLE_LOG_TO_DRIVER:-"1"}
}

# set up numactl for the specified module IDs
set_numactl(){
    if [ "$module_ids" != "None" ]; then
        # Check if module_ids is a comma-separated list of integers
        if [[ $module_ids =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            IFS="," read -r -a MODULES <<< "$module_ids"
        else
            echo "The specified module IDs should be a comma-separated list of integers instead of $module_ids."
            return
        fi
    else
        echo no modules specified, skip numactl
        return
    fi

    HL_TOPO="hl-smi topo -c -N"
    NODE_MEM=($( echo -e "$($HL_TOPO | grep "^[$(IFS="|" ; echo "${MODULES[*]}")]" | awk '{print $4}' | uniq)" ))
    NODE_CPUS=($( echo -e "$($HL_TOPO | grep "^[$(IFS="|" ; echo "${MODULES[*]}")]" | awk '{print $2}' | uniq | sed 's/,//g')" ))

    if [ "${#NODE_MEM[@]}" -gt 1 ] || [ "${#NODE_CPUS[@]}" -gt 1 ];then
        echo "The specified modules are not on the same NUMA node, skip numactl"
        return
    fi
    NUM_HPU_PER_NODE=$($HL_TOPO | grep -c "${NODE_CPUS[0]}")

    CPUS_LOW=$(echo "${NODE_CPUS[0]}" | cut -d '-' -f 1)
    CPUS_UP=$(echo "${NODE_CPUS[0]}" | cut -d '-' -f 2)
    NUM_CPU_PER_HPU=$(echo "($CPUS_UP-$CPUS_LOW+1)/$NUM_HPU_PER_NODE" | bc)

    CORES=()
    for MODULE in "${MODULES[@]}"; do
        MODULE_IDX=$(echo "$MODULE % $NUM_HPU_PER_NODE" | bc)
        CORE_LOW=$(echo "$CPUS_LOW + ($NUM_CPU_PER_HPU * $MODULE_IDX)" | bc)
        CORE_UP=$(echo "$CORE_LOW + $NUM_CPU_PER_HPU - 1" | bc)
        CORES+=("$CORE_LOW-$CORE_UP")
    done
    CORES_STR=$(IFS="," ; echo "${CORES[*]}")

    NUMA_CTL="numactl -C $CORES_STR -p ${NODE_MEM[0]}"
    echo "using '$NUMA_CTL' for module id: $module_ids"
}

set_module_ids(){
    if [[ $module_ids =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        IFS="," read -r -a MODULES <<< "$module_ids"
        # check if the length of module_ids is equal to num_hpu
        if [ ${#MODULES[@]} -ne "$num_hpu" ]; then
            echo "The number of module IDs should be equal to the number of HPUs."
            exit
        fi
        if [ "$num_hpu" -gt 1 ]; then
            export HABANA_VISIBLE_MODULES=$module_ids
        else
            export HLS_MODULE_ID=$module_ids
        fi

        # set up numactl based on module ids
        set_numactl
    elif [ "$module_ids" == "None" ]; then
        echo "No module IDs specified, skip numactl"
        NUMA_CTL=""
        export HABANA_VISIBLE_MODULES=$(IFS="," ; echo "${available_modules[*]}")
    else
        echo "The specified module IDs should be a comma-separated list of integers instead of $module_ids."
        exit
    fi
}

set_dtype(){
    case "$dtype" in
        "bfloat16" | "float16")
            echo Running with dtype="$dtype" ;;
        "fp8")
            echo Running with dtype="$dtype"
            export QUANT_CONFIG=${QUANT_CONFIG:-"$BASH_DIR/quantization/${model_name_lower}/maxabs_quant_g2.json"}
            export PT_HPU_LAZY_MODE=1   # INC must run in lazy mode
            ;;
        "awq")
            echo Running with AWQ
            extra_params+=(--quantization awq_hpu)
            ;;
        "gptq")
            echo Running with GPTQ
            extra_params+=(--quantization gptq_hpu)
            ;;
        *)
            echo Invalid dtype: "$dtype"
            exit
            ;;
    esac
}

# set up linear bucketing based on input/output range and max_num_batched_tokens
set_linear_bucketing(){
    export VLLM_EXPONENTIAL_BUCKETING=false

    max_num_batched_tokens=${max_num_batched_tokens:-8192}
    max_num_seqs=${max_num_seqs:-128}
    input_min=${input_min:-1024}
    input_max=${input_max:-1024}
    output_max=${output_max:-2048}
    block_size=${BLOCK_SIZE:-128}

    prompt_bs_step=1
    prompt_bs_min=1
    prompt_bs_max=$(( $max_num_batched_tokens / $(ceil $input_min $block_size) ))
    prompt_bs_max=$( ceil $prompt_bs_max $prompt_bs_step )
    prompt_bs_max=$( min $prompt_bs_max $max_num_seqs ) # consider max_num_prefill_seqs
    prompt_bs_max=$( max $prompt_bs_max 1 )
    export VLLM_PROMPT_BS_BUCKET_MIN=${VLLM_PROMPT_BS_BUCKET_MIN:-$prompt_bs_min}
    export VLLM_PROMPT_BS_BUCKET_STEP=${VLLM_PROMPT_BS_BUCKET_STEP:-$prompt_bs_step}
    export VLLM_PROMPT_BS_BUCKET_MAX=${VLLM_PROMPT_BS_BUCKET_MAX:-$prompt_bs_max}

    prompt_seq_step=$block_size
    prompt_seq_min=$( ceil $input_min $prompt_seq_step )
    prompt_seq_max=$( ceil $input_max $prompt_seq_step )
    export VLLM_PROMPT_SEQ_BUCKET_MIN=${VLLM_PROMPT_SEQ_BUCKET_MIN:-$prompt_seq_min}
    export VLLM_PROMPT_SEQ_BUCKET_STEP=${VLLM_PROMPT_SEQ_BUCKET_STEP:-$prompt_seq_step}
    export VLLM_PROMPT_SEQ_BUCKET_MAX=${VLLM_PROMPT_SEQ_BUCKET_MAX:-$prompt_seq_max}

    decode_bs_step=$( ceil_div $max_num_seqs 16 )
    decode_bs_step=$( min $decode_bs_step 16 )
    decode_bs_min=1
    decode_bs_max=$( ceil $max_num_seqs $decode_bs_step )
    export VLLM_DECODE_BS_BUCKET_MIN=${VLLM_DECODE_BS_BUCKET_MIN:-$decode_bs_min}
    export VLLM_DECODE_BS_BUCKET_STEP=${VLLM_DECODE_BS_BUCKET_STEP:-$decode_bs_step}
    export VLLM_DECODE_BS_BUCKET_MAX=${VLLM_DECODE_BS_BUCKET_MAX:-$decode_bs_max}

    decode_block_step=$block_size
    decode_block_min=$( ceil_div $input_min $block_size )
    decode_block_min=$( ceil_div $decode_block_min $decode_block_step )
    max_context_len=$(( $input_max + $output_max ))
    max_context_blocks=$(( $(ceil_div $max_context_len $block_size) + 1 )) # todo: check if must +1
    decode_block_max=$(( $max_context_blocks * $decode_bs_max))
    decode_block_max=$( ceil $decode_block_max $decode_block_step )
    export VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN:-$decode_block_min}
    export VLLM_DECODE_BLOCK_BUCKET_STEP=${VLLM_DECODE_BLOCK_BUCKET_STEP:-$decode_block_step}
    export VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX:-$decode_block_max}
}

set_perf_tuning(){
    if [ "$cache_path" != "" ]; then
        echo "HPU recipe cache will be saved to $cache_path"
        export PT_HPU_RECIPE_CACHE_CONFIG=${cache_path},false,16384
        mkdir -p "${cache_path}"
    fi

    if [ "$skip_warmup" == "true" ]; then
        echo "VLLM_SKIP_WARMUP is set to true"
        export VLLM_SKIP_WARMUP=true
    fi

    if [ "$profile" == "true" ]; then
        echo "VLLM_PROFILER_ENABLED is set to true"
        export VLLM_PROFILER_ENABLED=true
        export VLLM_PROFILE_FILE=${case_name}_profile.json
    else
        extra_params+=("--disable-log-requests")
    fi

    if [ "$disable_zero_padding" == "true" ]; then
        echo "VLLM_ZERO_PADDING is disabled"
        export VLLM_ZERO_PADDING=false
    else
        echo "VLLM_ZERO_PADDING is enabled"
        export VLLM_ZERO_PADDING=true
    fi

    if [ "$disable_fsdpa" == "true" ]; then
        echo "VLLM_PROMPT_USE_FUSEDSDPA is disabled"
        export VLLM_PROMPT_USE_FUSEDSDPA=false
    else
        echo "VLLM_PROMPT_USE_FUSEDSDPA is enabled"
        export VLLM_PROMPT_USE_FUSEDSDPA=true
    fi

    # VLLM_FP32_SOFTMAX=true by default for model_type=qwen2* models.
    # set VLLM_FP32_SOFTMAX=false for models without accuracy issue.
    if [[ $model_name_lower == *"deepseek-r1-distill-qwen-14b"* \
        || $model_name_lower == *"deepseek-r1-distill-qwen-32b"* \
        || $model_name_lower == *"deepseek-r1-distill-llama-8b"* \
        || $model_name_lower == *"deepseek-r1-distill-llama-70b"* \
        || $model_name_lower == *"qwen3-8b"* \
        || $model_name_lower == *"qwen3-14b"* \
        || $model_name_lower == *"qwen3-32b"* \
        || $model_name_lower == *"qwq-32b"* \
        || $model_name_lower == *"qwen3-30b-a3b"* \
        || $model_name_lower == *"qwen3-235b-a22b"* \
        ]]; then
        export VLLM_FP32_SOFTMAX=false
        echo "Set VLLM_FP32_SOFTMAX=false for $model_name"
    fi

    # check if 'experts' in the model's config.json
    # disable expert parallel for Llama-4-Scout-17B-16E-Instruct
    if [[ "$model_name_lower" != *"llama-4-scout-17b-16e-instruct"* ]]; then
        extra_params+=("--enable-expert-parallel")
    fi

    # TODO: add ray start process
    if [ "$num_hpu" -gt 8 ]; then
        dist_backend="ray"
    else
        dist_backend="mp"
    fi
}

set_config(){
    set_module_ids
    set_dtype
    if [ "$use_linear_bucketing" == "true" ]; then
        set_linear_bucketing
    fi
    set_perf_tuning

    new_env=( $(env) )
    # report out the changed env
    echo "Changed environment variables:"
    comm -13 <(printf "%s\n" "${original_env[@]}" | sort) <(printf "%s\n" "${new_env[@]}" | sort)
}
