#! /bin/bash

PREFILL_PORT=8100
export MOONCAKE_CONFIG_PATH=./mooncake_p.json
MODEL_PATH=/software/data/disk10/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

# DO NOT change unless you fully undersand its purpose
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export PT_HPU_LAZY_MODE=1

export VLLM_MOE_N_SLICE=8
export VLLM_EP_SIZE=8

block_size=128
# DO NOT change ends...

# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.5
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_PROMPT_RATIO=0
export VLLM_MLA_DISABLE_REQUANTIZATION=0
export VLLM_DELAYED_SAMPLING="true"
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
#export VLLM_MOE_SLICE_LENGTH=20480

# params
max_model_len=8192
max_num_batched_tokens=8192
max_num_seqs=32
input_min=1
input_max=16384
output_max=16384


export VLLM_DECODE_BS_BUCKET_MIN=32
export VLLM_DECODE_BS_BUCKET_STEP=32
export VLLM_DECODE_BS_BUCKET_MAX=32
export VLLM_DECODE_BLOCK_BUCKET_MIN=2048
export VLLM_DECODE_BLOCK_BUCKET_STEP=2048
export VLLM_DECODE_BLOCK_BUCKET_MAX=2048
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_PROMPT_SEQ_BUCKET_MIN=1024
export VLLM_PROMPT_SEQ_BUCKET_STEP=1024
export VLLM_PROMPT_SEQ_BUCKET_MAX=8192
export VLLM_PROFILER_ENABLED=true
env | grep VLLM


python3 -m vllm.entrypoints.openai.api_server --port $PREFILL_PORT \
--block-size 128 \
--model $MODEL_PATH \
--device hpu \
--dtype bfloat16 \
--kv-cache-dtype fp8_inc \
--tensor-parallel-size 8 \
--trust-remote-code  \
--max-model-len $max_model_len \
--max-num-seqs $max_num_seqs \
--max-num-batched-tokens $max_num_batched_tokens  \
--use-padding-aware-scheduling \
--use-v2-block-manager \
--distributed_executor_backend ray \
--gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
--disable-log-requests \
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
