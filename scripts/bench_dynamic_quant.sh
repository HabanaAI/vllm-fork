#!/bin/bash
# https://raw.githubusercontent.com/HabanaAI/vllm-fork/refs/heads/dev/qwen3/scripts/01-benchmark-online-235B-tp8.sh
#########################################################
# vLLM Benchmark Script for Qwen3
# 
# This script runs a vLLM server with specific configurations
# and benchmarks it using the sonnet dataset.
#########################################################

#===========================================================
# CONFIGURATION PARAMETERS
#===========================================================

# bash bench_dynamic_quant.sh --model_path /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/  --tp_size 8 --use_inc 1
# bash bench_dynamic_quant.sh --model_path /mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/  --tp_size 8


# ./bench_dynamic_quant.sh --model_path /mnt/weka/data/pytorch/Qwen/Qwen3-30B-A3B --tp_size 8

# ./bench_dynamic_quant.sh --model_path /mnt/weka/llm/Qwen3-30B-A3B-FP8/ --tp_size 8
# ./bench_dynamic_quant.sh --model_path /mnt/weka/data/pytorch/DeepSeek-R1/ --tp_size 8

# model_path="/mnt/weka/data/pytorch/Qwen/Qwen3-30B-A3B"
# model_path="/mnt/weka/data/pytorch/DeepSeek-R1/"

export no_proxy=localhost,127.0.0.1
if [ $# -gt 0 ] && [ "$1" == "--model_path" ]; then
    model=$2
else
    model="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/"
fi

if [ $# -eq 4 ] && [ "$3" == "--tp_size" ]; then
    tp_size=$4
else
    tp_size=8
fi

# use inc or not
if [ $# -eq 6 ] && [ "$5" == "--use_inc" ]; then
    use_inc=$6
else
    use_inc=0
fi

if [ $use_inc -eq 1 ]; then
    echo "Using inc fork of vLLM"
    export QUANT_CONFIG=inc_dynamic_quant_config.json
    export VLLM_HPU_FORCE_CHANNEL_FP8=0
    export CALC_SCALE_WITH_CGUID=1
    export PT_HPU_RECIPE_CACHE_CONFIG=~/gc_recipes/inc_dynamic,false,16384
    export VLLM_DISABLE_MARK_SCALES_AS_CONST=1
else
    echo "Using original vLLM"
    export   VLLM_HPU_FORCE_CHANNEL_FP8=1
fi

model_name=$(basename ${model})

# Model Configuration
tokenizer=$model

# Hardware Configuration
moe_n_slice=1         # MoE groups
gpu_utils=0.65        # GPU memory utilization

# Request Configuration
max_model_len=9216    # Max model len
request_rate="inf"    # Request rate (inf = unlimited)
multi_step=1          # Number of scheduler steps


#===========================================================
# START the LOOP
#===========================================================

tp_parallel=$tp_size
req_in_out_list=(32_1024_1024 )
#req_in_out_list=(192_5120_1024)

for req_in_out in "${req_in_out_list[@]}"; do
    # Token Length Configuration
    bs=$(echo "$req_in_out" | awk -F'_' '{ print $1 }')
    in_len=$(echo "$req_in_out" | awk -F'_' '{ print $2 }')
    out_len=$(echo "$req_in_out" | awk -F'_' '{ print $3 }')

    num_prompts=$((bs * 3)) 
    # Expert parallelism size
    ep_size=${tp_parallel}

    #===========================================================
    # DERIVED PARAMETERS
    #===========================================================

    # Calculate and align total length
    # Calculate aligned lengths for buckets
    in_len_aligned=$(((in_len + 127) / 128 * 128))
    prompt_seq_max=$((in_len * 1500 / 1000))
    prompt_seq_max=$(((prompt_seq_max + 127) / 128 * 128))

    total_len=$((prompt_seq_max + out_len))
    if [ $((total_len % 128)) -ne 0 ]; then
        echo 'Rounding up total length to multiple of 128'
        total_len=$(((total_len / 128 + 1) * 128))
    fi

    total_len_aligned=$(((total_len + 127) / 128 * 128))

    decode_total_len=$((total_len + 128))
    decode_total_len_aligned=$(((decode_total_len + 127) / 128 * 128))
    
    echo "Configuration:"
    echo "- Batch Size: ${bs}"
    echo "- Input Length: ${in_len}"
    echo "- Output Length: ${out_len}"
    echo "- Total Length: ${total_len}"
    echo "- Total Length Aligned: ${total_len_aligned}"
    echo "- Input Length Aligned: ${in_len_aligned}"
    echo "- Prompt Sequence Max: ${prompt_seq_max}"
    echo "- Decode Total Length: ${decode_total_len}"
    echo "- Decode Total Length Aligned: ${decode_total_len_aligned}"

    
    # Calculate bucket sizes
    VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len_aligned * bs / 128))
    VLLM_DECODE_BLOCK_BUCKET_MIN=$(((VLLM_DECODE_BLOCK_BUCKET_MIN + 127) / 128 * 128))
    VLLM_DECODE_BLOCK_BUCKET_MAX=$((decode_total_len_aligned * bs / 128))
    VLLM_DECODE_BLOCK_BUCKET_MAX=$(((VLLM_DECODE_BLOCK_BUCKET_MAX + 127) / 128 * 128))

    #===========================================================
    # LOG CONFIGURATION
    #===========================================================

    # Create a descriptive log name based on parameters
    timestamp=$(date +%Y%m%d-%H%M%S)
    log_name="${model_name}-gaudi2-tp${tp_parallel}-ep${ep_size}-moe${moe_n_slice}-ms${multi_step}_np${num_prompts}_rr${request_rate}_bs${bs}_i${in_len}_o${out_len}_len${total_len}_${timestamp}"

    # Create log directory
    mkdir -p benchmark_logs

    #===========================================================
    # START vLLM SERVER
    #===========================================================

    echo "Starting vLLM server with the following configuration:"
    echo "- Model: ${model_name}"
    echo "- Tensor Parallel Size: ${tp_parallel}"
    echo "- Expert Parallel Size: ${ep_size}"
    echo "- Batch Size: ${bs}"
    echo "- Input Length: ${in_len}"
    echo "- Output Length: ${out_len}"
    echo "- Total Length: ${total_len}"
    
    VLLM_LOGGING_LEVEL=DEBUG \
    PT_HPU_LAZY_MODE=1 \
    VLLM_PROMPT_BS_BUCKET_MIN=1 \
    VLLM_PROMPT_BS_BUCKET_MAX=8 \
    VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len_aligned} \
    VLLM_PROMPT_SEQ_BUCKET_MAX=${prompt_seq_max} \
    VLLM_DECODE_BS_BUCKET_MIN=${bs} \
    VLLM_DECODE_BS_BUCKET_MAX=${bs} \
    VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
    VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
    VLLM_DECODE_BLOCK_BUCKET_STEP=128 \
    VLLM_DELAYED_SAMPLING=true \
    HABANA_VISIBLE_DEVICES="ALL" \
    VLLM_EP_SIZE=${ep_size} \
    PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
    PT_HPU_WEIGHT_SHARING=0 \
    python3 -m vllm.entrypoints.openai.api_server \
        --port 18080 \
        --model ${model} \
        --load-format safetensors \
        --config-format hf \
        --tensor-parallel-size ${tp_parallel} \
        --max-num-seqs ${bs} \
        --disable-log-requests \
        --dtype bfloat16 \
        --use-v2-block-manager \
        --use-padding-aware-scheduling \
        --num_scheduler_steps ${multi_step} \
        --max-model-len $((total_len_aligned)) \
        --max-num-batched-tokens $((total_len_aligned * 4)) \
        --distributed_executor_backend ray \
        --gpu_memory_utilization ${gpu_utils} \
        --enable-expert-parallel \
        2>&1 | tee benchmark_logs/${log_name}_serving.log &
    pid=$(($!-1))
        #  --trust-remote-code false    --enforce-eager \

    # Wait for server to start
    n=0
    ready=false
    until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
        n=$((n+1))
        if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
            break
        fi
        sleep 5s
    done
    sleep 10s
    echo "Server started with PID: ${pid}"

    #===========================================================
    # RUN BENCHMARK
    #===========================================================

    echo "Starting benchmark with random dataset"
    max_concurrency_client=${bs}
    start_time=$(date +%s)

    
    python3 ../benchmarks/benchmark_serving.py \
        --backend vllm \
        --model ${model} \
        --tokenizer ${tokenizer} \
        --dataset-name random \
        --request-rate ${request_rate} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --ignore-eos \
        --num-prompts ${num_prompts} \
        --port 18080 \
        --random-input-len ${in_len} \
        --random-output-len ${out_len} \
        --max-concurrency ${max_concurrency_client} \
        --append-result \
        --save-result 2>&1 | tee benchmark_logs/${log_name}_benchmark.log


    end_time=$(date +%s)
    echo "Benchmark completed in $((end_time - start_time)) seconds"

    # Clean up
    echo "Stopping vLLM server"
    kill ${pid}
    echo "Script execution completed"
    sleep 10
done