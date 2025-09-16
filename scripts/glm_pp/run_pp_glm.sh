#! /bin/bash

pkill -9 python

export MAX_MODEL_LEN=8192
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS_PER_PP_GROUP=16
export VLLM_FP32_SOFTMAX=false
export HABANA_VISIBLE_DEVICES=ALL
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_RAY_DISABLE_LOG_TO_DRIVER=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7
export VLLM_GPU_MEMORY_UTILIZATION=0.8
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0
export VLLM_EP_SIZE=4
export VLLM_PP_USE_CPU_COMS=1
# export PT_HPU_RECIPE_CACHE_CONFIG=/data/40960_cache,false,40960
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=2
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=128
export VLLM_PROMPT_SEQ_BUCKET_MAX=10240
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=4
export VLLM_DECODE_BS_BUCKET_MAX=16
export VLLM_DECODE_BLOCK_BUCKET_MIN=64
export VLLM_DECODE_BLOCK_BUCKET_STEP=64
export VLLM_DECODE_BLOCK_BUCKET_MAX=1280
export VLLM_SKIP_WARMUP=true
export VLLM_DELAYED_SAMPLING=false
model_path=/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air-FP8-G2
# model_path=/mnt/disk8/meta-llama/Llama-3.2-3B-Instruct
# model_path=/mnt/disk8/Qwen/Qwen3-30B-A3B
tp_size=4
pp_size=2
# export QUANT_CONFIG="./quant_configs/inc_quant.json"
export INC_ENABLE_TP_RANK_INFO=1
# export QUANT_CONFIG="./quant_configs/inc_quant_w2.json"
export QUANT_CONFIG="./quant_configs/inc_quant_w4.json"
# export QUANT_CONFIG="./quant_configs/inc_quant_expand_w4.json"


python3 -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8688 \
  --block-size 128 \
  --model $model_path \
  --device hpu \
  --tensor-parallel-size ${tp_size} \
  --pipeline-parallel-size ${pp_size} \
  --trust-remote-code \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS_PER_PP_GROUP} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --disable-log-requests \
  --use-padding-aware-scheduling \
  --use-v2-block-manager \
  --distributed_executor_backend mp \
  --enable-expert-parallel \
  --num-scheduler-steps 1 \
  --enable-expert-parallel \
  --gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
  --tool-call-parser glm45 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
