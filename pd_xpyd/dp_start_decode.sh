#!/bin/bash
#set -x

# machine id, EP, TP, DP Index, DP Host IP
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/dp_d_env.sh

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
    --enable-reasoning
    --reasoning-parser deepseek_r1
    $kv_cache_dtype_arg
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
  )
  # Only define log_file if XPYD_LOG is set
  if [ -n "$XPYD_LOG" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="$XPYD_LOG/log_rank${RANK}_${timestamp}.log"
  fi

  extra_env=()
#  if [ "$i" -eq 0 ] && [ "$RANK" -eq 0 ]; then
#    extra_env+=(VLLM_PROFILER_ENABLED=true)
#  fi

  # Execute command
  if [ "$DP_RANK" -ne 1 ]; then
    if [ -n "$XPYD_LOG" ]; then
      echo "env VLLM_DP_RANK=$RANK ${CMD[*]} (logging to $log_file)"
      env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${extra_env[@]}" "${CMD[@]}" 2>&1 | tee "$log_file" &
    else
      echo "env VLLM_DP_RANK=$RANK ${CMD[*]} (no logging)"
      env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${extra_env[@]}" "${CMD[@]}" &
    fi
  else
    if [ -n "$XPYD_LOG" ]; then
      echo "${CMD[*]} (logging to $log_file)"
      "${CMD[@]}" 2>&1 | tee "$log_file" &
    else
      echo "${CMD[*]} (no logging)"
      "${CMD[@]}" &
    fi
  fi
done

wait

