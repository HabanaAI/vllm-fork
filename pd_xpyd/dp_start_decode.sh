#!/bin/bash
#set -x

sleep 30

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
  # INC FP8 settings
  if [ "$EP_SIZE" -eq 32 ]; then
    export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp1ep32.json
  else
    export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp1ep16.json
  fi
else
  kv_cache_dtype_arg=""
  echo "<decode>it's bf16 kv cache mode"
fi

# Control whether to apply numactl bindings (1=enable, 0=disable)
NUMACTL_ENABLED=${VLLM_USE_NUMACTL:-1}

# Build CPU/NUMA bindings from hl-smi topology if available
if command -v hl-smi >/dev/null 2>&1; then
  # This sets CPU_BIND_<mod> and MEM_BIND_<mod> shell variables for each module id
  eval "$(hl-smi topo -c -N | awk '
    NR<=2 { next }                                  # skip headers
    $1 ~ /^[0-9]+$/ {
      mod=$1;
      mem=$NF;                                      # last field is NUMA node
      cpu="";
      for (i=2; i<=NF-1; i++) {                    # CPU affinity spans fields 2..NF-1
        part=$i;
        sub(/,$/, "", part);                      # drop trailing comma per field
        if (cpu=="") cpu=part; else cpu=cpu "," part;
      }
      gsub(/ /, "", cpu);                         # remove spaces
      printf "CPU_BIND_%s=%s; MEM_BIND_%s=%s\n", mod, cpu, mod, mem;
    }
  ')"
fi

for ((i=0; i<$DP_RANK; i++))
do
  RANK=$((DP_INDEX * DP_RANK + i))
  port=$((8200 + i))

  # Derive Habana module id for this rank and bind to the corresponding NUMA/CPU
  MOD_ID=$((RANK % 8))
  CPU_BIND_VAR="CPU_BIND_${MOD_ID}"
  MEM_BIND_VAR="MEM_BIND_${MOD_ID}"
  CPU_BIND="${!CPU_BIND_VAR}"
  MEM_BIND="${!MEM_BIND_VAR}"

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
    --preemption-mode swap
    --swap-space "$SWAP_SPACE"
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

  if [ "$NUMACTL_ENABLED" -eq 1 ]; then
    echo "CPU_BIND: $CPU_BIND"
    echo "MEM_BIND: $MEM_BIND"
    echo "HLS_MODULE_ID: $MOD_ID"
    echo "DP_RANK: $RANK"
  fi

  # Execute command
  if [ "$DP_RANK" -ne 1 ]; then
    if [ -n "$XPYD_LOG" ]; then
      if [ "$NUMACTL_ENABLED" -eq 1 ] && [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
        echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]} (logging to $log_file)" 2>&1 | tee -a "$log_file"
        env HLS_MODULE_ID="$MOD_ID" VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" 2>&1 | tee -a "$log_file" &
      else
        echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK ${CMD[*]} (logging to $log_file)" 2>&1 | tee -a "$log_file"
        env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${CMD[@]}" 2>&1 | tee -a "$log_file" &
      fi
    else
      echo "env VLLM_DP_RANK=$RANK ${CMD[*]} (no logging)"
      env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${extra_env[@]}" "${CMD[@]}" &
      if [ "$NUMACTL_ENABLED" -eq 1 ] && [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
        echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]} (no logging)"
        env HLS_MODULE_ID="$MOD_ID" VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" &
      else
        echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK ${CMD[*]} (no logging)"
        VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${CMD[@]}" &
      fi
    fi
  else
    if [ -n "$XPYD_LOG" ]; then
      if [ "$NUMACTL_ENABLED" -eq 1 ] && [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
        echo "numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]}  (logging to $log_file)" 2>&1 | tee -a "$log_file"
        env HLS_MODULE_ID="$MOD_ID" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" 2>&1 | tee -a "$log_file" &
      else
        echo "${CMD[*]} (logging to $log_file)" 2>&1 | tee -a "$log_file"
        "${CMD[@]}" 2>&1 | tee -a "$log_file" &
      fi
    else
      if [ "$NUMACTL_ENABLED" -eq 1 ] && [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
        echo "numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]} (no logging)"
        env HLS_MODULE_ID="$MOD_ID" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" &
      else
        echo "${CMD[*]} (no logging)"
        "${CMD[@]}" &
      fi
    fi
  fi
done

wait
