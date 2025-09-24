#! /bin/bash

# set -x

original_env=( $(env) )

# HPU specific constants
DEVICE_NAME=$(hl-smi -Q name -f csv | tail -n 1)
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
    export PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-"1"}   # change to '0' to use torch.compile
    if [ "$num_hpu" -gt 1 ]; then
        export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
    fi

    # performance tuning
    export VLLM_GRAPH_RESERVED_MEM=${VLLM_GRAPH_RESERVED_MEM:-"0.1"}
    export VLLM_DELAYED_SAMPLING=${VLLM_DELAYED_SAMPLING:-"true"}
    export VLLM_ZERO_PADDING=${VLLM_ZERO_PADDING:-"true"}
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"true"}

    # network
    default_host_ip=${host:-$(hostname -I | awk '{print $1}')}
    default_ifname=$( ip -br addr show to ${default_host_ip} | awk '{print $1}' )
    export VLLM_HOST_IP=${VLLM_HOST_IP:-"${default_host_ip}"}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"${default_ifname}"}
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-"${default_ifname}"}
}

# set max_num_batched_tokens and max_model_len based on input/output ranges
set_length(){
    max_num_batched_tokens=$(( $input_max + $output_max ))
    if [ "$max_num_batched_tokens" -lt $PREFERED_BATCHED_TOKENS ]; then
        max_num_batched_tokens=$PREFERED_BATCHED_TOKENS
    fi
    # Ceiling max_num_batched_tokens to a multiple of BLOCK_SIZE
    max_num_batched_tokens=$( ceil $max_num_batched_tokens $BLOCK_SIZE )

    if [ "$max_num_prefill_seqs" == "" ]; then
        # Ceiling input_min to a multiple of BLOCK_SIZE
        input_min_ceil=$( ceil $input_min $BLOCK_SIZE )
        max_num_prefill_seqs=$(( $max_num_batched_tokens / $input_min_ceil ))
        max_num_prefill_seqs=$( max $max_num_prefill_seqs 1 )
    fi

    max_model_len=$(( $input_max + $output_max ))
    # Ceiling max_model_len to a multiple of BLOCK_SIZE
    max_model_len=$( ceil $max_model_len $BLOCK_SIZE )

    if [ "$input_min" == "$input_max" ]; then
        disable_zero_padding=true
    fi
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
            quant_config=${quant_config:-"$BASH_DIR/quantization/${model_name_lower}/maxabs_quant_g2.json"}
            export QUANT_CONFIG=${QUANT_CONFIG:-"$quant_config"}
            export PT_HPU_WEIGHT_SHARING=0
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
    prompt_bs_max=$(( $PREFERED_SEQ_LEN_TO_CAPTURE / $(ceil $input_min $block_size) ))
    prompt_bs_max=$( ceil $prompt_bs_max $prompt_bs_step )
    prompt_bs_max=$( min $prompt_bs_max $max_num_seqs )
    if [ -n "$max_num_prefill_seqs" ]; then
        prompt_bs_max=$( min $prompt_bs_max $max_num_prefill_seqs )
    fi
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
    decode_block_min=$( max $decode_block_min $decode_block_step )
    max_context_len=$(( $input_max + $output_max ))
    max_context_blocks=$( ceil_div $max_context_len $block_size )
    decode_block_max=$(( $max_context_blocks * $decode_bs_max ))
    decode_block_max=$( ceil $decode_block_max $decode_block_step )
    export VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN:-$decode_block_min}
    export VLLM_DECODE_BLOCK_BUCKET_STEP=${VLLM_DECODE_BLOCK_BUCKET_STEP:-$decode_block_step}
    export VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX:-$decode_block_max}
}

set_perf_tuning(){
    if [ "$cache_path" != "" ]; then
        echo "HPU recipe cache will be saved to $cache_path"
        export PT_HPU_RECIPE_CACHE_CONFIG=${cache_path},false,40960
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

    # VLLM_FP32_SOFTMAX=false by default, set to true for models with accuracy issues.
    if [[ $model_name_lower == *"deepseek-r1-distill-qwen-7b"* \
            || $model_name_lower == *"qwen2-7b-instruct"* \
            || $model_name_lower == *"qwen2.5-7b-instruct"* ]]; then
        export VLLM_FP32_SOFTMAX=true
        echo "Set VLLM_FP32_SOFTMAX=true for $model_name"
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
    set_common_env
    set_length
    set_module_ids
    set_dtype
    if [ "$use_linear_bucketing" == "true" ]; then
        set_linear_bucketing
    fi
    set_perf_tuning

    new_env=( $(env) )
    # report out the changed env
    changed_env=$(comm -13 <(printf "%s\n" "${original_env[@]}" | sort) <(printf "%s\n" "${new_env[@]}" | sort))
}
