#!/bin/bash

# VLLM_CONTIGUOUS_PA=false VLLM_SKIP_WARMUP=true python3 -m \
#     vllm.entrypoints.openai.mm_api_server \
#     --models mistralai/Mistral-7B-Instruct-v0.3 meta-llama/Llama-3.1-8B-Instruct \
#     --port 8080 --device hpu --dtype bfloat16 \
#     --gpu-memory-utilization=0.3 --use-v2-block-manager --max-model-len 4096 2>&1 > multi_models.log &


bs=128
in_len=1024
out_len=1024


python benchmarks/benchmark_serving.py \
		--backend vllm \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--dataset-name sonnet \
		--dataset-path benchmarks/sonnet.txt \
		--request-rate 512 \
		--num-prompts ${bs} \
		--port 8080 \
		--sonnet-input-len ${in_len} \
		--sonnet-output-len ${out_len} \
		--sonnet-prefix-len 100 \
		--save-result > mistral-sonnet-1.log 2>&1 &

python benchmarks/benchmark_serving.py \
		--backend vllm \
		--model meta-llama/Llama-3.1-8B-Instruct \
		--dataset-name sonnet \
		--dataset-path benchmarks/sonnet.txt \
		--request-rate 512 \
		--num-prompts ${bs} \
		--port 8080 \
		--sonnet-input-len ${in_len} \
		--sonnet-output-len ${out_len} \
		--sonnet-prefix-len 100 \
		--save-result > llama-sonnet-1.log 2>&1 &