export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/:/tmp/ucx-gaudi/install/lib/:/opt/amazon/openmpi/lib:/usr/lib/habanalabs
unset https_proxy;
unset http_proxy;
unset HTTPS_PROXY;
unset HTTP_PROXY;

GPU_ID=1
SIDE_CHANNEL_PORT=9771
SIDE_CHANNEL_HOST="localhost"
model_name="/mnt/weka/data/llm-d-models-pv/Meta-Llama-3.1-8B-Instruct/"
#model_name="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
PORT=1909
PREFILLER_TP_SIZE=1

#UCX_LOG_LEVEL=trace 
#RANK=1 
#    --num-gpu-blocks-override 19500 \
#    --block-size 16 \
BASE_CMD="UCX_TLS=ib,rc,gaudi_gdr PT_HPU_BLOCK_SIZE_FACTOR=8 PT_HPU_ENABLE_RESTORE_KV_LAYOUT=1 NIXL_LOG_LEVEL=debug VLLM_LOGGING_LEVEL=DEBUG UCX_MEMTYPE_CACHE=0 VLLM_USE_V1=1 \
	CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_HOST=$SIDE_CHANNEL_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --max-num-batched-tokens 99999 \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"hpu\"}' 2>&1 | tee decode.log "
 
eval "$BASE_CMD"


#NIXL_LOG_LEVEL=debug \
#RANK=1 UCX_LOG_LEVEL=trace \
#UCX_MEMTYPE_CACHE=0 VLLM_USE_V1=1 UCX_TLS=tcp \
#VLLM_ENABLE_V1_MULTIPROCESSING=1 \
#VLLM_WORKER_MULTIPROC_METHOD=spawn \
#VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
#	vllm serve /mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/ --port 8100 \
#	--enforce-eager --disable-log-requests --gpu-memory-utilization 0.8 \
#	--tensor-parallel-size 1 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"hpu"}' 2>&1 | tee prefill.log

