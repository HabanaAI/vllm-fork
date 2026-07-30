#! /bin/bash

# set -x

original_env=( $(env) )

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

# set up numactl for the selected module IDs
set_numactl(){
    if [ "$module_ids" != "None" ]; then
        # Check if module_ids is a comma-separated list of integers
        if [[ $module_ids =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            IFS="," read -r -a selected_modules <<< "$module_ids"
        else
            echo "The selected module IDs should be a comma-separated list of integers instead of $module_ids."
            return
        fi
    else
        echo no modules selected, skip numactl
        return
    fi

    hl_topo_cmd="hl-smi topo -c -N"
    memory_nodes=($( echo -e "$($hl_topo_cmd | grep "^[$(IFS="|" ; echo "${selected_modules[*]}")]" | awk '{print $4}' | uniq)" ))
    cpu_nodes=($( echo -e "$($hl_topo_cmd | grep "^[$(IFS="|" ; echo "${selected_modules[*]}")]" | awk '{print $2}' | uniq | sed 's/,//g')" ))

    if [ "${#memory_nodes[@]}" -gt 1 ] || [ "${#cpu_nodes[@]}" -gt 1 ];then
        echo "The selected modules are not on the same NUMA node, skip numactl"
        return
    fi
    memory_node=${memory_nodes[0]}
    cpu_node=${cpu_nodes[0]}
    num_hpu_per_node=$($hl_topo_cmd | grep -c "${cpu_node}")

    cpus_lower=$(echo "${cpu_node}" | cut -d '-' -f 1)
    cpus_upper=$(echo "${cpu_node}" | cut -d '-' -f 2)
    num_cpu_per_hpu=$(echo "($cpus_upper-$cpus_lower+1)/$num_hpu_per_node" | bc)

    selected_cores=()
    for module_id in "${selected_modules[@]}"; do
        local_idx=$(echo "$module_id % $num_hpu_per_node" | bc)
        core_lower=$(echo "$cpus_lower + ($num_cpu_per_hpu * $local_idx)" | bc)
        core_upper=$(echo "$core_lower + $num_cpu_per_hpu - 1" | bc)
        selected_cores+=("$core_lower-$core_upper")
    done
    core_ids=$(IFS="," ; echo "${selected_cores[*]}")

    NUMA_CTL_CMD="numactl -C $core_ids -p ${memory_node}"
    echo "using '$NUMA_CTL_CMD' for module id: $module_ids"
}

set_module_ids(){
    module_to_index=()
    all_modules=()
    while IFS=',' read -r index module_id; do
        [[ $index == "index" ]] && continue
        index=$(echo $index | xargs)
        module_id=$(echo $module_id | xargs)
        all_modules+=("$module_id")
        module_to_index[$module_id]=$index
    done < <(hl-smi -Q index,module_id -f csv)

    # sort all_modules
    mapfile -t all_modules < <(printf "%s\n" "${all_modules[@]}" | sort -n)

    used_modules=()
    available_modules=()
    for module_id in "${all_modules[@]}"; do
        module_index=${module_to_index[$module_id]}
        # check if the device is in-use
        if [ -n "$(lsof /dev/accel/accel_controlD$module_index)" ]; then
            used_modules+=("$module_id")
        else
            available_modules+=("$module_id")
        fi
    done

    if [ ${#used_modules[@]} -eq 0 ]; then
        echo available modules: ${available_modules[*]}
    else
        echo all modules: ${all_modules[*]}
        echo modules in-use: ${used_modules[*]}
        echo available modules: ${available_modules[*]}
    fi

    if [[ $module_ids =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        IFS="," read -r -a selected_modules <<< "$module_ids"
        # check if the length of module_ids is equal to num_hpu
        if [ ${#selected_modules[@]} -ne "$num_hpu" ]; then
            echo "The number of module IDs should be equal to the number of HPUs."
            exit
        fi
        # make sure all the selected module_ids are in available_modules
        for module_id in "${selected_modules[@]}"; do
            if [[ ! " ${available_modules[*]} " =~ " $module_id " ]]; then
                echo "The selected module ID $module_id is not available. Available module IDs are: ${available_modules[*]}"
                exit
            fi
        done
        if [ "$num_hpu" -gt 1 ]; then
            export HABANA_VISIBLE_MODULES=$module_ids
        else
            export HLS_MODULE_ID=$module_ids
        fi

        # set up numactl based on module ids
        set_numactl
    elif [ "$module_ids" == "None" ]; then
        echo "No module IDs selected, skip numactl"
        NUMA_CTL_CMD=""
        # if HABANA_VISIBLE_MODULES is not set or is 'all', use all available modules
        if [ -z "$HABANA_VISIBLE_MODULES" ] || [ "$HABANA_VISIBLE_MODULES" == "all" ]; then
            export HABANA_VISIBLE_MODULES=$(IFS="," ; echo "${available_modules[*]}")
        else
            # make sure all the visible module_ids are in available_modules
            IFS="," read -r -a visible_modules <<< "$HABANA_VISIBLE_MODULES"
            for module_id in "${visible_modules[@]}"; do
                if [[ ! " ${available_modules[*]} " =~ " $module_id " ]]; then
                    echo "The visible module ID $module_id in HABANA_VISIBLE_MODULES is not available."
                    echo "Available module IDs are: ${available_modules[*]}"
                    exit
                fi
            done
        fi
    else
        echo "The selected module IDs should be a comma-separated list of integers instead of $module_ids."
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

set_partial_warmup(){
    if [ "$partial_warmup" = "true" ]; then
        echo "VLLM_SKIP_LAZY_WARMUP is set to True"
        export VLLM_SKIP_LAZY_WARMUP=true
        export VLLM_PROMPT_SEQ_BUCKET_MAX=$max_warmup_seq
        export VLLM_DECODE_BS_BUCKET_MAX=$max_warmup_bs
        export VLLM_DECODE_BLOCK_BUCKET_MAX=$(( ( ($max_warmup_seq + $output_max) * $max_warmup_bs  / $block_size + $decode_block_step - 1 ) / $decode_block_step * $decode_block_step ))
    fi
}

set_config(){
    set_module_ids
    set_dtype
    set_common_env
    set_bucketing
    set_perf_tuning
    set_partial_warmup

    new_env=( $(env) )
    # report out the changed env
    changed_env=$(comm -13 <(printf "%s\n" "${original_env[@]}" | sort) <(printf "%s\n" "${new_env[@]}" | sort))
}
