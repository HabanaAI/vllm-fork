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
export VLLM_PROFILER_ENABLED=true
export VLLM_PROFILE_CONFIG_PATH=profile_config.json
export HABANA_PROFILE_WRITE_HLTV=1
#export HABANA_PROFILE=profile_api_with_nics
export HABANA_PROFILE=profile_api


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
    $kv_cache_dtype_arg
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
  )
  log_file="$log_dir/log_rank${RANK}_${timestamp}.log"
  echo "CPU_BIND: $CPU_BIND"
  echo "MEM_BIND: $MEM_BIND"
  echo "HLS_MODULE_ID: $MOD_ID"
  echo "DP_RANK: $RANK"

  if [ "$DP_RANK" -ne 1 ]; then
    if [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
      echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]}"
      env HLS_MODULE_ID="$MOD_ID" VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" 2>&1 | tee "$log_file" &
    else
      echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK ${CMD[*]}"
      env HLS_MODULE_ID="$MOD_ID" VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${CMD[@]}" 2>&1 | tee "$log_file" &
    fi
  else
    if [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
      echo "numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]}"
      env HLS_MODULE_ID="$MOD_ID" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" &
    else
      echo "${CMD[*]}"
      env HLS_MODULE_ID="$MOD_ID" "${CMD[@]}" &
    fi
  fi
done

wait

