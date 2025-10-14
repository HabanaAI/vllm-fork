#!/bin/bash

# Source the original script content but run in background
export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/:/tmp/ucx-gaudi/install/lib/:/opt/amazon/openmpi/lib:/usr/lib/habanalabs

unset https_proxy;
unset http_proxy;
unset HTTPS_PROXY;
unset HTTP_PROXY;

GPU_ID="3"
SIDE_CHANNEL_PORT=9760
SIDE_CHANNEL_HOST="172.26.47.37" # sc09super16-nvd
model_name="/mnt/weka/data/llm-d-models-pv/Meta-Llama-3.1-8B-Instruct/"
PORT=9700
PREFILLER_TP_SIZE=1

BASE_CMD="UCX_TLS=ib,rc,cuda_copy NIXL_LOG_LEVEL=info VLLM_LOGGING_LEVEL=INFO UCX_MEMTYPE_CACHE=0 VLLM_USE_V1=1 \
        CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_HOST=$SIDE_CHANNEL_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --max-num-batched-tokens 99999 \
    --disable-log-requests \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"cuda\"}'"

# Execute the command and save PID
eval "$BASE_CMD" > /tmp/run_prefill_gpu.log 2>&1 &
VLLM_PID=$!
echo $VLLM_PID > /tmp/run_prefill_gpu.pid
echo "VLLM server started with PID: $VLLM_PID"

# Wait for the process to finish
wait $VLLM_PID
