#!/bin/bash
export VLLM_PROFILER_ENABLED=1
# export VLLM_PT_PROFILE=decode_32_1_t
# hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on
hl-prof-config --use-template profile_api --hw-trace off
export VLLM_TORCH_PROFILER_DIR=./profiler_dir_internvl_v4
export HABANA_PROFILE=1
export HOST=localhost 
export PORT=12345 
export MODEL="/software/stanley/benchmark-config/vllm-Internvl/vllm-fork/InternVL3-14B" 
export PT_HPU_RECIPE_CACHE_CONFIG="/tmp/int/recipe,False,4096"
export VLLM_USE_V1=0 
export VLLM_MULTIMODAL_BUCKETS=1,13,26,52 
export PT_HPU_LAZY_MODE=1 
export ENABLE_SCALAR_LIKE_FULL_FUSION=1 
export VLLM_ENABLE_ASYNC_MM_PREPROCESS=1 
export VLLM_MAX_CONCURRENT_PREPROC=25 
export ENABLE_FUSION_BEFORE_NORM=true 
export FUSER_ENABLE_LOW_UTILIZATION=true 
export VLLM_DETOKENIZE_ON_OPENAI_SERVER=true 
export VLLM_EXPONENTIAL_BUCKETING=False 
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 
export VLLM_SUSPEND_PROMPT=1 
export VLLM_GRAPH_RESERVED_MEM=0.8 
export VLLM_PROMPT_BS_BUCKET_MIN=1  
export VLLM_PROMPT_BS_BUCKET_STEP=2 
export VLLM_PROMPT_BS_BUCKET_MAX=4 
export VLLM_PROMPT_SEQ_BUCKET_MIN=768 
export VLLM_PROMPT_SEQ_BUCKET_STEP=256 
export VLLM_PROMPT_SEQ_BUCKET_MAX=2432 
export VLLM_DECODE_BS_BUCKET_MIN=1  
export VLLM_DECODE_BS_BUCKET_STEP=8 
export VLLM_DECODE_BS_BUCKET_MAX=64 
export VLLM_DECODE_BLOCK_BUCKET_MIN=20 
export VLLM_DECODE_BLOCK_BUCKET_STEP=64 
export VLLM_DECODE_BLOCK_BUCKET_MAX=864 
export VLLM_SKIP_WARMUP=false
# numactl --cpunodebind=0 --membind=0 \
python3 -m vllm.entrypoints.openai.api_server --host $HOST --port $PORT \
 --block-size 128 --model  $MODEL \
 --dtype bfloat16 --tensor-parallel-size 1 \
 --max-model-len 19456 --max-num-batched-tokens 19456 \
 --gpu_memory_utilization 0.99 --limit-mm-per-prompt image=13 \
 --max-num-prefill-seqs 4 --max-num-seqs 64 --split_qkv \
 --disable-log-requests --disable-log-stats --trust-remote-code 2>&1 | tee log.txt