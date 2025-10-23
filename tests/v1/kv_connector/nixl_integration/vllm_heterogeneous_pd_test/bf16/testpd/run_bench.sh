#!/bin/bash

#MODEL="/software/data/pytorch/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/"
MODEL="/mnt/weka/data/llm-d-models-pv/Meta-Llama-3.1-8B-Instruct/"
PORT=1295
ISL=8192
OSL=1024
CONC=32 #(4 8 16 32 64)
RANDOM_RANGE_RATIO=0.8
NUM_PROMPT=256

for i in "${!CONC[@]}"; do
    c=${CONC[$i]}
    ts=$(date +"%Y%m%d_%H%M%S")

    # python3 /yyoon/github/production-stack/nixl/vllm-fork/benchmarks/benchmark_serving.py \
    python3 /tmp/bench_serving/benchmark_serving.py \
    --model $MODEL \
    --backend vllm \
    --base-url "http://localhost:$PORT" \
    --dataset-name random \
    --random-input-len $ISL --random-output-len $OSL \
    --random-range-ratio $RANDOM_RANGE_RATIO \
    --num-prompts=$NUM_PROMPT --max-concurrency=$c \
    --request-rate inf --ignore-eos \
    --save-result --percentile-metrics ttft,tpot,itl,e2el \
    --result-filename log_c32_8k1k_256.txt   --result-dir ./

    sleep 3
done




