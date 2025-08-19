#! /bin/bash

# set -x

# get device name
device=$(hl-smi -Q name -f csv | tail -n 1)

# set up common environment variables for vllm
set_common_env(){
    # pytorch bridge
    export PT_HPU_WEIGHT_SHARING=${PT_HPU_WEIGHT_SHARING:-"0"}
    export PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-"1"}
    if [ "$num_hpu" -gt 1 ]; then
        export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
    fi

    # memory usage tuning
    export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-"0.9"}
    export VLLM_GRAPH_RESERVED_MEM=${VLLM_GRAPH_RESERVED_MEM:-"0.2"}
    export VLLM_GRAPH_PROMPT_RATIO=${VLLM_GRAPH_PROMPT_RATIO:-"0.8"}
    export VLLM_MAX_SEQ_LEN_TO_CAPTURE=${VLLM_MAX_SEQ_LEN_TO_CAPTURE:-"8192"}

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

    NUMA_CTL="numactl -C $CORES_STR -m ${NODE_MEM[0]}"
    echo "using '$NUMA_CTL' for module id: $module_ids"
}

# set up bucketing based on input/output range and max_num_batched_tokens
set_bucketing(){
    max_num_batched_tokens=${max_num_batched_tokens:-8192}
    max_num_seqs=${max_num_seqs:-128}
    input_min=${input_min:-1024}
    input_max=${input_max:-1024}
    output_max=${output_max:-2048}
    block_size=${block_size:-128}

    prompt_bs_step=1
    prompt_bs_min=1
    prompt_bs_max=$(( $max_num_batched_tokens / $input_min ))
    # prompt_bs_max = min(prompt_bs_max, max_num_seqs)
    prompt_bs_max=$(( $prompt_bs_max > $max_num_seqs ? $max_num_seqs : $prompt_bs_max ))
    # prompt_bs_max = CEILING.MATH(prompt_bs_max, prompt_bs_step)
    prompt_bs_max=$(( ($prompt_bs_max + $prompt_bs_step - 1) / $prompt_bs_step * $prompt_bs_step ))    
    export VLLM_PROMPT_BS_BUCKET_MIN=${VLLM_PROMPT_BS_BUCKET_MIN:-$prompt_bs_min}
    export VLLM_PROMPT_BS_BUCKET_STEP=${VLLM_PROMPT_BS_BUCKET_STEP:-$prompt_bs_step}
    export VLLM_PROMPT_BS_BUCKET_MAX=${VLLM_PROMPT_BS_BUCKET_MAX:-$prompt_bs_max}

    prompt_seq_step=$block_size
    # prompt_seq_min = CEILING.MATH(input_min, prompt_seq_step)
    prompt_seq_min=$(( ($input_min + $prompt_seq_step -1) / $prompt_seq_step * $prompt_seq_step ))
    # prompt_seq_max = CEILING.MATH(input_max, prompt_seq_step) + prompt_seq_step
    prompt_seq_max=$(( (($input_max + $prompt_seq_step -1) / $prompt_seq_step + 1) * $prompt_seq_step ))
    export VLLM_PROMPT_SEQ_BUCKET_MIN=${VLLM_PROMPT_SEQ_BUCKET_MIN:-$prompt_seq_min}
    export VLLM_PROMPT_SEQ_BUCKET_STEP=${VLLM_PROMPT_SEQ_BUCKET_STEP:-$prompt_seq_step}
    export VLLM_PROMPT_SEQ_BUCKET_MAX=${VLLM_PROMPT_SEQ_BUCKET_MAX:-$prompt_seq_max}

    # decode_bs_step = ROUNDUP(max_num_seqs / 16, 0)
    decode_bs_step=$(( ($max_num_seqs + 15) / 16 ))
    # decode_bs_step = min(decode_bs_step, 16)
    decode_bs_step=$(( $decode_bs_step > 16 ? 16 : $decode_bs_step ))
    decode_bs_min=1
    # decode_bs_max = CEILING.MATH(max_num_seqs, decode_bs_step)
    decode_bs_max=$(( ($max_num_seqs + $decode_bs_step -1) / $decode_bs_step * $decode_bs_step ))
    export VLLM_DECODE_BS_BUCKET_MIN=${VLLM_DECODE_BS_BUCKET_MIN:-$decode_bs_min}
    export VLLM_DECODE_BS_BUCKET_STEP=${VLLM_DECODE_BS_BUCKET_STEP:-$decode_bs_step}
    export VLLM_DECODE_BS_BUCKET_MAX=${VLLM_DECODE_BS_BUCKET_MAX:-$decode_bs_max}

    decode_block_step=$block_size
    # decode_block_min = ROUNDUP(input_min / block_size, 0)
    decode_block_min=$(( ($input_min + $block_size - 1) / $block_size ))
    # decode_block_min = CEILING.MATH(decode_block_min, decode_block_step)
    decode_block_min=$(( ($decode_block_min + $decode_block_step -1) / $decode_block_step * $decode_block_step ))
    # decode_block_max = (ROUNDUP((input_max + output_max) / block_size, 0) + 1) * decode_bs_max
    decode_block_max=$(( (($input_max + $output_max + $block_size -1) / $block_size + 1) * $decode_bs_max))
    # decode_block_max = (CEILING.MATH(decode_block_max, decode_block_step)
    decode_block_max=$(( ($decode_block_max + $decode_block_step -1) / $decode_block_step * $decode_block_step ))
    export VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN:-$decode_block_min}
    export VLLM_DECODE_BLOCK_BUCKET_STEP=${VLLM_DECODE_BLOCK_BUCKET_STEP:-$decode_block_step}
    export VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX:-$decode_block_max}
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
            export PT_HPU_WEIGHT_SHARING=0
            export VLLM_DISABLE_MARK_SCALES_AS_CONST=true
            kv_cache_dtype_arg=(--kv-cache-dtype fp8_inc)
            weights_load_device_arg=""
            if [[ "${model_name_lower}" == *"deepseek-r1-distill-llama-8b"* ]]; then
                kv_cache_dtype_arg=(--kv-cache-dtype auto)
            fi
            if [[ "${model_name_lower}" == *"qwen3-235b-a22b"* ]]; then
                kv_cache_dtype_arg=(--kv-cache-dtype auto)
                weights_load_device_arg=(--weights-load-device cpu)
            fi
            if [[ "${model_name_lower}" == *"qwen3"* ]]; then
                # qwen3 models that using fp8 attention and kv-cache
                if [[ $model_name_lower == *"qwen3-32b"* \
                    || $model_name_lower == *"qwen3-30b-a3b"* \
                    ]]; then
                    kv_cache_dtype_arg=(--kv-cache-dtype fp8_inc)
                else
                    kv_cache_dtype_arg=(--kv-cache-dtype auto)
                fi
            elif [[ $model_name_lower == *"deepseek-r1-distill-qwen-7b"* \
                || $model_name_lower == *"qwen2-7b-instruct"* \
                || $model_name_lower == *"qwen2.5-7b-instruct"* ]]; then
                kv_cache_dtype_arg=(--kv-cache-dtype auto)
            fi

            echo Using "${kv_cache_dtype_arg[@]}" for $model_name
            QUANT_ARGS=(--quantization inc ${kv_cache_dtype_arg[@]} ${weights_load_device_arg[@]})
            dtype="bfloat16"
            ;;
        "awq")
            echo Running with AWQ
            QUANT_ARGS=(--quantization awq_hpu)
            dtype="bfloat16"
            ;;
        "gptq")
            echo Running with GPTQ
            QUANT_ARGS=(--quantization gptq_hpu)
            dtype="bfloat16"
            ;;
        *)
            echo Invalid dtype: "$dtype"
            exit
            ;;
    esac
}

set_perf_tuning(){
    if [ "$cache_path" != "" ]; then
        echo "HPU recipe cache will be saved to $cache_path"
        export PT_HPU_RECIPE_CACHE_CONFIG=${cache_path},false,16384
        mkdir -p "${cache_path}"
    fi

    if [ "$skip_warmup" = "true" ]; then
        echo "VLLM_SKIP_WARMUP is set to True"
        export VLLM_SKIP_WARMUP=True
    fi

    if [ "$profile" = "true" ]; then
        echo "VLLM_PROFILER_ENABLED is set to True"
        export VLLM_PROFILER_ENABLED=True
        export VLLM_PROFILE_FILE=${case_name}_profile.json
    fi

    if [ "$disable_zero_padding" = "true" ]; then
        echo "VLLM_ZERO_PADDING is disabled"
        export VLLM_ZERO_PADDING=false
    else
        echo "VLLM_ZERO_PADDING is enabled"
        export VLLM_ZERO_PADDING=true
    fi

    if [ "$disable_fsdpa" = "true" ]; then
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
        echo Set VLLM_FP32_SOFTMAX=false for $model_name
    fi

    VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-"0.9"}
    VLLM_MAX_SEQ_LEN_TO_CAPTURE=${VLLM_MAX_SEQ_LEN_TO_CAPTURE:-"8192"}

    if [[ "$model_name_lower" == *"llama-4-scout-17b-16e-instruct"* ]]; then
        # disable expert parallel for Llama-4-Scout-17B-16E-Instruct
        ENABLE_EXPERT_PARALLEL=""
    else
        ENABLE_EXPERT_PARALLEL="--enable-expert-parallel"
    fi

    if [ "$num_hpu" -gt 8 ]; then
        dist_backend="ray"
    else
        dist_backend="mp"
    fi
}

set_config(){
    set_module_ids
    set_dtype
    set_common_env
    set_bucketing
    set_perf_tuning
}
