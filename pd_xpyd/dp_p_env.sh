#set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_bucket.sh
source "$BASH_DIR"/pd_env.sh

export VLLM_EP_SIZE=8

export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_PROMPT_RATIO=1
# params
model_len=16384
max_num_batched_tokens=16384
max_num_seqs=8
input_min=128
input_max=16384
output_max=16384

unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

set_bucketing

export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=1
export VLLM_DECODE_BLOCK_BUCKET_MIN=2
export VLLM_DECODE_BLOCK_BUCKET_STEP=1
export VLLM_DECODE_BLOCK_BUCKET_MAX=2

echo " environments are reseted "

env | grep VLLM_PROMPT_BS
env | grep VLLM_PROMPT_SEQ
env | grep VLLM_DECODE_BS
env | grep VLLM_DECODE_BLOCK

export VLLM_SKIP_PREFILL_SAMPLING=1

export VLLM_SKIP_WARMUP=True
export VLLM_DP_SIZE=1
export VLLM_USE_V1=0

#unset VLLM_SKIP_WARMUP
#export PT_HPU_RECIPE_CACHE_CONFIG=/workspace/ww33_inc_fp8_p,false,131072

if [ "$INC_FP8" -eq 1 ]; then
  export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp8ep8.json
fi

#python3 -m vllm.entrypoints.openai.api_server --model $model_path --port 8100 --max-model-len $model_len --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION -tp 16  --max-num-seqs $max_num_seqs --trust-remote-code --disable-async-output-proc --kv-cache-dtype fp8_inc --disable-log-requests --max-num-batched-tokens $max_num_batched_tokens --use-padding-aware-scheduling --use-v2-block-manager --distributed_executor_backend ray --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
