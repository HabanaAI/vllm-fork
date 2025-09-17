#!/bin/bash
model_path="${model_path:-/mnt/weka/llm/gemma-3-27b-it}"
# model_path="${model_path:-/mnt/weka/llm/Qwen2.5-VL-3B-Instruct}"
vllm_port="${vllm_port:-8688}"


# python3 /software/stanley/benchmark-config/vllm-fork-async-mm/benchmarks/benchmark_serving.py \
# 	--backend openai-chat \
# 	--endpoint /chat/completions \
# 	--base-url http://127.0.0.1:8688/v1 \
# 	--model /mnt/weka/llm/gemma-3-27b-it \
# 	--request-rate "inf" \
# 	--percentile-metrics ttft,tpot,itl,e2el \
# 	--ignore-eos \
# 	--num-prompt 16 \
# 	--port 8688 \
#     --dataset-path lmms-lab/LLaVA-OneVision-Data \
#     --dataset-name hf \
#     --hf-subset "chart2text(cauldron)" \
#     --hf-split train \
# 	--max-concurrency 128 \
# 	--save-result 


# python3 /software/stanley/benchmark-config/vllm-fork/benchmarks/benchmark_serving.py \
# 	--backend openai-chat \
# 	--endpoint /chat/completions \
# 	--base-url http://127.0.0.1:8688/v1 \
# 	--model /mnt/weka/llm/gemma-3-27b-it \
# 	--request-rate "inf" \
# 	--percentile-metrics ttft,tpot,itl,e2el \
# 	--ignore-eos \
# 	--num-prompt 100 \
# 	--port 8688 \
# 	--dataset-path MUIRBENCH/MUIRBENCH \
# 	--dataset-name hf \
# 	--max-concurrency 128 \
# 	--save-result \
# 	--limit-mm-per-prompt image=6 \
# 	--text-len 512 \
#     --hf-output-len 295
# 	# --save-detailed
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
#python3 /workdir/vllm-fork/benchmarks/benchmark_serving.py \
#        --backend openai-chat \
#	--endpoint /chat/completions \
#	--base-url http://127.0.0.1:8688/v1 \
#	--model "/software/data/pytorch/huggingface/hub/gemma-3-27b-it" \
#        --percentile-metrics ttft,tpot,itl,e2el \
#        --num-prompt 3 \
#	--port 8688 \
#        --dataset-path MUIRBENCH/MUIRBENCH \
#	--dataset-name hf \
#	--hf-output-len 295 \
#	--input-len 8192 \
#	--filtered-res 1024,1024 \
#        --max-concurrency 25 \
#	--limit-mm-per-prompt image=6 \
#	--ignore-eos \
#	--full
curl -X POST http://localhost:$vllm_port/start_profile
python3 /workdir/vllm-fork/benchmarks/benchmark_serving.py \
        --backend openai-chat \
	--endpoint /chat/completions \
	--base-url http://127.0.0.1:8688/v1 \
	--model "/software/data/pytorch/huggingface/hub/gemma-3-27b-it" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --num-prompt 3 \
	--port 8688 \
        --dataset-path MUIRBENCH/MUIRBENCH \
	--dataset-name hf \
	--hf-output-len 295 \
	--input-len 8192 \
	--filtered-res 1024,1024 \
        --max-concurrency 25 \
	--limit-mm-per-prompt image=6 \
	--ignore-eos \
	--full
curl -X POST http://localhost:$vllm_port/stop_profile
# # 	#--save-detailed
  #--model /software/data/pytorch/huggingface/hub/gemma-3-27b-it \

# curl -sS http://127.0.0.1:8688/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "/mnt/weka/llm/Phi-4-reasoning-plus",
#     "messages": [
#       {"role": "system", "content": "You are a helpful reasoning assistant."},
#       {"role": "user", "content": "How does vLLM work, and what is the difference with v0 and v1, and list all 100 optimizations of vLLM"}
#     ],
#     "max_tokens": 6000,
#     "temperature": 0.8,
#     "top_k": 50,
#     "top_p": 0.95
#   }'


#curl -X POST http://localhost:$vllm_port/start_profile
#python3 /software/stanley/benchmark-config/vllm-fork/benchmarks/benchmark_serving.py \
#  --backend openai-chat \
#  --endpoint /chat/completions \
#  --base-url http://127.0.0.1:8688/v1 \
#  --model /mnt/weka/llm/granite-3.1-8b-instruct \
#  --percentile-metrics ttft,tpot,itl,e2el \
#  --num-prompt 1 \
#  --port 8688 \
#  --dataset-name random \
#  --random-output-len 1024 \
#  --random-input-len 8192 \
#  --max-concurrency 25
#curl -X POST http://localhost:$vllm_port/stop_profile
#   --limit-mm-per-prompt image=6 \
#   --ignore-eos \
#   --full
