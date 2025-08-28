#!/bin/bash
#set -x

# machine id, EP, TP, DP Index, DP Host IP
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/dp_d_env.sh

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="xpyd_logs"
mkdir -p "$log_dir"

export MOONCAKE_CONFIG_PATH="$BASH_DIR"/mooncake_$1.json
echo "MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH"

EP_SIZE=$2
echo "EP_SIZE=$EP_SIZE"

TP_SIZE=$3
echo "TP_SIZE=$TP_SIZE"

DP_SIZE=$((EP_SIZE / TP_SIZE))
echo "DP_SIZE=$DP_SIZE"

DP_RANK=$((8 / TP_SIZE))
echo "DP_RANK=$DP_RANK"

DP_INDEX=$4
echo "DP_INDEX=$DP_INDEX"

DP_HOST_IP=$5
echo "DP_HOST_IP=$DP_HOST_IP"

export VLLM_DP_SIZE=$DP_SIZE
export VLLM_DP_MASTER_IP=$DP_HOST_IP
export VLLM_EP_SIZE=$EP_SIZE

if [ "$DP_SIZE" -eq 1 ]; then
  unset VLLM_DP_SIZE
  unset VLLM_DP_MASTER_IP
  unset VLLM_DP_MASTER_PORT
fi

if [ "$INC_FP8" -eq 1 ]; then
  kv_cache_dtype_arg="--kv-cache-dtype fp8_inc"
  echo "<decode>it's inc fp8 kv cache mode"
else
  kv_cache_dtype_arg=""
  echo "<decode>it's bf16 kv cache mode"
fi

export VLLM_TORCH_PROFILER_DIR=./profiles
export VLLM_PROFILER_ENABLED=full
export VLLM_PROFILE_CONFIG_PATH=profile_config.json
export HABANA_PROFILE_WRITE_HLTV=1
export HABANA_PROFILE=profile_api_with_nics


for ((i=0; i<$DP_RANK; i++))
do
  RANK=$((DP_INDEX * DP_RANK + i))
  port=$((8200 + i))

  CMD=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$model_path"
    --port "$port"
    --max-model-len "$model_len"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    -tp "$TP_SIZE"
    --max-num-seqs "$max_num_seqs"
    --trust-remote-code
    --disable-log-requests
    --max-num-batched-tokens "$max_num_batched_tokens"
    --use-padding-aware-scheduling
    --use-v2-block-manager
    --distributed_executor_backend mp
    $kv_cache_dtype_arg
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
  )
  log_file="$log_dir/log_rank${RANK}_${timestamp}.log"

  if [ "$DP_RANK" -ne 1 ]; then
    echo "env VLLM_DP_RANK=$RANK ${CMD[*]}"
    env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${CMD[@]}" 2>&1 | tee "$log_file" &
  else
    echo "${CMD[*]}"
    "${CMD[@]}" &
  fi
done

wait

