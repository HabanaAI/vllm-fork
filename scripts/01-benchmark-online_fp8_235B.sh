#!/bin/bash
#########################################################
# vLLM Benchmark Script for Qwen3
# 
# This script runs a vLLM server with specific configurations
# and benchmarks it using the sonnet dataset.
#########################################################

#===========================================================
# CONFIGURATION PARAMETERS
#===========================================================


if [ $# -gt 0 ] && [ "$1" == "--model_path" ]; then
    model=$2
else
    model="/mnt/weka/llm/qwen3_pre_release/Qwen3-30B-A3B-250425/"
fi

if [ $# -gt 4 ] && [ "$3" == "--tp_size" ]; then
    tp_size=$4
else
    tp_size=1
fi

if [ $# -eq 6 ] && [ "$5" == "--ep_size" ]; then
    ep_size=$6
else
    ep_size=1
fi

model_name=$(basename ${model})

# Model Configuration
tokenizer=$model

# Hardware Configuration
tp_parallel=${tp_size}         # Tensor parallelism size
moe_n_slice=1         # MoE groups
gpu_utils=0.95        # GPU memory utilization

# Request Configuration
req_in_out_list=(192_5120_1024)
#req_in_out_list=(512_1024_1024 192_5120_1024)
for VLLM_FP32_SOFTMAX in 0 1; do
    for req_in_out in "${req_in_out_list[@]}"; do
        con=$(echo "$req_in_out" | awk -F'_' '{ print $1 }')
        isl=$(echo "$req_in_out" | awk -F'_' '{ print $2 }')
        osl=$(echo "$req_in_out" | awk -F'_' '{ print $3 }')

        bs=${con}                 # Batch size
        num_prompts=$((con * 5))       # Number of prompts for benchmarking
        request_rate="inf"    # Request rate (inf = unlimited)
        multi_step=1          # Number of scheduler steps

        # Token Length Configuration
        in_len=${isl}           # Input length
        out_len=${osl}          # Output length

        #===========================================================
        # DERIVED PARAMETERS
        #===========================================================

        # Calculate and align total length
        total_len=$((in_len + out_len))
        if [ $((total_len % 128)) -ne 0 ]; then
            echo 'Rounding up total length to multiple of 128'
            total_len=$(((total_len / 128 + 1) * 128))
        fi

        # Calculate aligned lengths for buckets
        in_len_aligned=$(((in_len + 127) / 128 * 128))
        prompt_seq_max=$(((in_len + 128 + 127) / 128 * 128))
        total_len_aligned=$(((total_len + 127) / 128 * 128))

        decode_total_len=$((total_len + 128))
        decode_total_len_aligned=$(((decode_total_len + 127) / 128 * 128))

        # Calculate bucket sizes
        VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len_aligned * bs / 128))
        VLLM_DECODE_BLOCK_BUCKET_MIN=$(((VLLM_DECODE_BLOCK_BUCKET_MIN + 127) / 128 * 128))
        VLLM_DECODE_BLOCK_BUCKET_MAX=$((decode_total_len_aligned * bs / 128))
        VLLM_DECODE_BLOCK_BUCKET_MAX=$(((VLLM_DECODE_BLOCK_BUCKET_MAX + 127) / 128 * 128))

        #===========================================================
        # LOG CONFIGURATION
        #===========================================================

        # Create a descriptive log name based on parameters
        log_name="FP8-fp32softmax${VLLM_FP32_SOFTMAX}-Qwen3-gaudi3-tp${tp_parallel}-ep${ep_size}-moe${moe_n_slice}-ms${multi_step}_np${num_prompts}_rr${request_rate}_bs${bs}_i${in_len}_o${out_len}_len${total_len}"

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

        # Environment variables for vLLM vllm-fork-qwen3/vllm/engine/arg_utils.py
        echo "* VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len_aligned}"
        echo "* VLLM_PROMPT_SEQ_BUCKET_MAX=${prompt_seq_max}"
        echo "* VLLM_DECODE_BS_BUCKET_MIN=${bs}"
        echo "* VLLM_DECODE_BS_BUCKET_MAX=${bs}"
        echo "* VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN}"
        echo "* VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX}"

        # export HABANA_LOGS=./habana_logs
        # echo $HABANA_LOGS
        # export LOG_LEVEL_ALL=3
        # export LOG_LEVEL_ALL_PT=3
        # export ENABLE_EXPERIMENTAL_FLAGS=true
        # export RUN_TPC_FUSER=false
        # ENABLE_EXPERIMENTAL_FLAGS=1 RUN_TPC_FUSER=0 \
        # PT_HPUGRAPH_DISABLE_TENSOR_CACHE=0 VLLM_SKIP_WARMUP=true \
        VLLM_FP32_SOFTMAX=${VLLM_FP32_SOFTMAX} \
        PT_HPU_LAZY_MODE=1 \
        VLLM_PROMPT_BS_BUCKET_MIN=1 \
        VLLM_PROMPT_BS_BUCKET_MAX=8 \
        VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len_aligned} \
        VLLM_PROMPT_SEQ_BUCKET_MAX=${prompt_seq_max} \
        VLLM_DECODE_BS_BUCKET_MIN=${bs} \
        VLLM_DECODE_BS_BUCKET_MAX=${bs} \
        VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
        VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
        VLLM_DECODE_BLOCK_BUCKET_STEP=256 \
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
            --max-model-len $(( total_len_aligned * 2 )) \
            --max-num-batched-tokens $(( in_len_aligned * 4 )) \
            --distributed_executor_backend ray \
            --gpu_memory_utilization ${gpu_utils} \
            --quantization=inc \
            --enable-expert-parallel \
            2>&1 | tee benchmark_logs/${log_name}_serving.log &
        pid=$(($!-1))
            # --enable-expert-parallel \ --trust-remote-code false    --enforce-eager \

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

        echo "Starting benchmark with Sonnet dataset"
        max_concurrency_client=1024
        start_time=$(date +%s)

        python3 ../benchmarks/benchmark_serving.py \
            --backend vllm \
            --model ${model} \
            --tokenizer ${tokenizer} \
            --dataset-name sonnet \
            --dataset-path ../benchmarks/sonnet.txt \
            --request-rate ${request_rate} \
            --percentile-metrics ttft,tpot,itl,e2el \
            --ignore-eos \
            --num-prompts ${num_prompts} \
            --port 18080 \
            --sonnet-input-len ${in_len} \
            --sonnet-output-len ${out_len} \
            --sonnet-prefix-len 100 \
            --max-concurrency ${max_concurrency_client} \
            --save-result 2>&1 | tee benchmark_logs/${log_name}_benchmark.log

        end_time=$(date +%s)
        echo "Benchmark completed in $((end_time - start_time)) seconds"

        # Clean up
        echo "Stopping vLLM server"
        kill ${pid}
        echo "Script execution completed"
    done
done
