#!/bin/bash

# Source the original script content but run in background
export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/:/tmp/ucx-gaudi/install/lib/:/opt/amazon/openmpi/lib:/usr/lib/habanalabs

export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=10000000
CMD="hl-prof-config --use-template profile_api --hw-trace off"
eval "$CMD &"
export VLLM_USE_V1=1
export VLLM_SKIP_WARMUP=True
export PT_HPU_LAZY_MODE=1
export HABANA_PROFILE=1
#Enable full vLLM Profiler and instruct where to save the profiling:
export VLLM_PROFILER_ENABLED=1
export VLLM_TORCH_PROFILER_DIR=./

export VLLM_EXPONENTIAL_BUCKETING=False
export VLLM_PROMPT_SEQ_BUCKET_MIN=1024
export VLLM_PROMPT_SEQ_BUCKET_STEP=256
export VLLM_PROMPT_SEQ_BUCKET_MAX=1280
export VLLM_DECODE_BLOCK_BUCKET_MIN=10
export VLLM_DECODE_BLOCK_BUCKET_MAX=330
export VLLM_USE_PADDING_AWARE_SCHEDULING=1

unset https_proxy;
unset http_proxy;
unset HTTPS_PROXY;
unset HTTP_PROXY;

GPU_ID=0,1,2,3
SIDE_CHANNEL_PORT=9790
SIDE_CHANNEL_HOST="172.26.47.134" # dell03
# SIDE_CHANNEL_HOST="172.26.47.41" # dell01
# SIDE_CHANNEL_HOST="172.26.47.138" # super17
# model_name="/mnt/weka/data/llm-d-models-pv/Meta-Llama-3.1-8B-Instruct/"
model_name="/software/data/pytorch/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/"
PORT=9909
DECODER_TP_SIZE=2

# BASE_CMD="VLLM_SKIP_WARMUP=true UCX_TLS=ib,rc,gaudi_gdr PT_HPU_BLOCK_SIZE_FACTOR=8 PT_HPU_ENABLE_RESTORE_KV_LAYOUT=1 \
# 	PT_HPU_LAZY_MODE=1 PT_HPUGRAPH_DISABLE_TENSOR_CACHE=False \
#     NIXL_LOG_LEVEL=debug VLLM_LOGGING_LEVEL=INFO UCX_MEMTYPE_CACHE=0 VLLM_USE_V1=1 \
#         HABANA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_HOST=$SIDE_CHANNEL_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
#     --port $PORT \
    # --enforce-eager \
#     --gpu-memory-utilization 0.8 \
#     --tensor-parallel-size $DECODER_TP_SIZE \
#     --max-num-batched-tokens 99999 \
#     --no-enable-prefix-caching \
#     --block-size 128 \
#     --disable-log-requests \
#     --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"hpu\"}'"

BASE_CMD="VLLM_SKIP_WARMUP=true UCX_TLS=ib,rc  PT_HPU_BLOCK_SIZE_FACTOR=8 PT_HPU_ENABLE_RESTORE_KV_LAYOUT=1 \
    NIXL_LOG_LEVEL=debug VLLM_LOGGING_LEVEL=INFO UCX_MEMTYPE_CACHE=0 VLLM_USE_V1=1 \
        HABANA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_HOST=$SIDE_CHANNEL_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --max-num-batched-tokens 99999 \
    --disable-log-requests \
    --async_scheduling \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"cpu\"}'"

# Execute the command and save PID
eval "$BASE_CMD" > /tmp/run_decoder_hpu.log 2>&1 &
VLLM_PID=$!
echo $VLLM_PID > /tmp/run_decoder_hpu.pid
echo "VLLM server started with PID: $VLLM_PID"

# Wait for the process to finish
wait $VLLM_PID
