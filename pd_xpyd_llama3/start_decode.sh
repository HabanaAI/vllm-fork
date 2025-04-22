set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
# source "$BASH_DIR"/pd_bucket.sh

# source /pd_env.sh

export MOONCAKE_CONFIG_PATH=./mooncake.json 
export HABANA_VISIBLE_DEVICES="1"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
model_path=/workspace/Meta-Llama-3.1-8B-Instruct/


export VLLM_GPU_MEMORY_UTILIZATION=0.8
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_PROMPT_RATIO=1
# params
model_len=2048
max_num_batched_tokens=2048
max_num_seqs=1
input_min=128
input_max=1024
output_max=1024

unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

# set_bucketing

# export VLLM_DECODE_BS_BUCKET_MIN=1
# export VLLM_DECODE_BS_BUCKET_STEP=1
# export VLLM_DECODE_BS_BUCKET_MAX=1
# export VLLM_DECODE_BLOCK_BUCKET_MIN=2
# export VLLM_DECODE_BLOCK_BUCKET_STEP=1
# export VLLM_DECODE_BLOCK_BUCKET_MAX=2

echo " environments are reseted "

env | grep VLLM_PROMPT_BS
env | grep VLLM_PROMPT_SEQ
env | grep VLLM_DECODE_BS
env | grep VLLM_DECODE_BLOCK

export VLLM_SKIP_WARMUP=True
#export PT_HPU_RECIPE_CACHE_CONFIG=./_prefill_cache,false,16384

python3 -m vllm.entrypoints.openai.api_server --model $model_path --port 8200 --max-model-len $model_len --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION -tp 1  --max-num-seqs $max_num_seqs --trust-remote-code --disable-async-output-proc --disable-log-requests --max-num-batched-tokens $max_num_batched_tokens --use-padding-aware-scheduling --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
