#!/bin/bash

export VLLM_MLA_DISABLE_REQUANTIZATION=1 
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"

export VLLM_EP_SIZE=1
export VLLM_SKIP_WARMUP=True
# export VLLM_LOGGING_LEVEL=DEBUG
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
# export PT_HPU_LAZY_MODE=0
export MAX_MODEL_LEN=8192
# export MODEL_PATH=/software/data/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/
export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

VLLM_USE_V1=0 VLLM_DP_RANK=0 VLLM_DP_SIZE=2 VLLM_DP_MASTER_IP=127.0.0.1 VLLM_DP_MASTER_PORT=25940 python -m vllm.entrypoints.openai.api_server --enable-expert-parallel --model $MODEL_PATH   --tensor-parallel-size 4 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.90 --max-num-seqs 32 --trust-remote-code --port 8801 &

VLLM_USE_V1=0 VLLM_DP_RANK=1 VLLM_DP_SIZE=2 VLLM_DP_MASTER_IP=127.0.0.1 VLLM_DP_MASTER_PORT=25940 python -m vllm.entrypoints.openai.api_server --enable-expert-parallel --model $MODEL_PATH   --tensor-parallel-size 4 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.90 --max-num-seqs 32 --trust-remote-code --port 8802  

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
