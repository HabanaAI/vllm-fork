#!/bin/bash
set -x

# Usage: benchmark_server_param.sh NUM_NODES MAX_MODEL_LEN MAX_NUM_SEQS TP_SIZE PP_SIZE COMM_BACKEND [PP_LAYER_PARTITION]

NUM_NODES=$1
MAX_MODEL_LEN=$2
MAX_NUM_SEQS=$3
TP_SIZE=$4
PP_SIZE=$5
COMM_BACKEND=$6
PP_LAYER_PARTITION=${7:-}
KV_CACHE_DTYPE=${8:-auto}
HOST=${9:-127.0.0.1}
PORT=${10:-8688}
MODEL_PATH=${11:-${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}}

#hl-prof-config --use-template profile_api --hw-trace off
#export HABANA_PROFILE=1
#export VLLM_PROFILER_ENABLED=full
#export VLLM_TORCH_PROFILER_DIR=./logs/

# Environment settings
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"

export VLLM_SKIP_WARMUP=true
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_DELAYED_SAMPLING="false"

# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_GRAPH_PROMPT_RATIO=0

export VLLM_EP_SIZE=$TP_SIZE
if [ "$PP_SIZE" -gt 1 ]; then
  if [ -n "$PP_LAYER_PARTITION" ]; then
    export VLLM_PP_LAYER_PARTITION=$PP_LAYER_PARTITION
  else
    echo "Warning: PP_SIZE >1 but PP_LAYER_PARTITION not provided"
  fi
fi

if [ "$COMM_BACKEND" = "gloo" ]; then
  export VLLM_PP_USE_CPU_COMS=1
fi

if [ "$KV_CACHE_DTYPE" = "fp8_inc" ]; then
  export VLLM_USE_FP8_MATMUL="true"
  export VLLM_USE_SINGLE_TENSOR_CACHE="1"
fi

# Bucketing configuration
BLOCK_SIZE=128
export PT_HPU_RECIPE_CACHE_CONFIG="/data/${MAX_MODEL_LEN}_cache,false,${MAX_MODEL_LEN}"
MAX_NUM_BATCHED_TOKENS=$MAX_MODEL_LEN

prompt_bs_min=1
prompt_bs_step=$(( MAX_NUM_SEQS > 32 ? 32 : MAX_NUM_SEQS ))
prompt_bs_max=$(( MAX_NUM_SEQS > 64 ? 64 : MAX_NUM_SEQS ))
export VLLM_PROMPT_BS_BUCKET_MIN=${VLLM_PROMPT_BS_BUCKET_MIN:-$prompt_bs_min}
export VLLM_PROMPT_BS_BUCKET_STEP=${VLLM_PROMPT_BS_BUCKET_STEP:-$prompt_bs_step}
export VLLM_PROMPT_BS_BUCKET_MAX=${VLLM_PROMPT_BS_BUCKET_MAX:-$prompt_bs_max}

prompt_seq_min=128
prompt_seq_step=128
prompt_seq_max=$MAX_NUM_BATCHED_TOKENS
export VLLM_PROMPT_SEQ_BUCKET_MIN=${VLLM_PROMPT_SEQ_BUCKET_MIN:-$prompt_seq_min}
export VLLM_PROMPT_SEQ_BUCKET_STEP=${VLLM_PROMPT_SEQ_BUCKET_STEP:-$prompt_seq_step}
export VLLM_PROMPT_SEQ_BUCKET_MAX=${VLLM_PROMPT_SEQ_BUCKET_MAX:-$prompt_seq_max}

decode_bs_min=1
decode_bs_step=$(( MAX_NUM_SEQS > 32 ? 32 : MAX_NUM_SEQS ))
decode_bs_max=$MAX_NUM_SEQS
export VLLM_DECODE_BS_BUCKET_MIN=${VLLM_DECODE_BS_BUCKET_MIN:-$decode_bs_min}
export VLLM_DECODE_BS_BUCKET_STEP=${VLLM_DECODE_BS_BUCKET_STEP:-$decode_bs_step}
export VLLM_DECODE_BS_BUCKET_MAX=${VLLM_DECODE_BS_BUCKET_MAX:-$decode_bs_max}

decode_block_min=128
decode_block_step=128
decode_block_max=$(( ((MAX_NUM_SEQS * MAX_MODEL_LEN / BLOCK_SIZE) > 128) ? (MAX_NUM_SEQS * MAX_MODEL_LEN / BLOCK_SIZE) : 128 ))
export VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN:-$decode_block_min}
export VLLM_DECODE_BLOCK_BUCKET_STEP=${VLLM_DECODE_BLOCK_BUCKET_STEP:-$decode_block_step}
export VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX:-$decode_block_max}

echo "Environments set for ${NUM_NODES}-node server: MAX_MODEL_LEN=${MAX_MODEL_LEN}, MAX_NUM_SEQS=${MAX_NUM_SEQS}, TP_SIZE=${TP_SIZE}, PP_SIZE=${PP_SIZE}, COMM_BACKEND=${COMM_BACKEND}"
env | grep VLLM

python3 -m vllm.entrypoints.openai.api_server --host $HOST --port $PORT \
  --block-size $BLOCK_SIZE \
  --model $MODEL_PATH \
  --device hpu \
  --dtype bfloat16 \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --tensor-parallel-size $TP_SIZE \
  --pipeline-parallel-size $PP_SIZE \
  --trust-remote-code \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --disable-log-requests \
  --use-padding-aware-scheduling \
  --use-v2-block-manager \
  --distributed_executor_backend ray \
  --gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
  --enable-reasoning \
  --reasoning-parser deepseek_r1
