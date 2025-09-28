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

#<<<<<<< HEAD
#=======
export VLLM_TORCH_PROFILER_DIR=./profiles
export VLLM_PROFILER_ENABLED=true
export VLLM_PROFILE_CONFIG_PATH=profile_config.json
export HABANA_PROFILE_WRITE_HLTV=1
export HABANA_PROFILE=profile_api_with_nics
#export HABANA_PROFILE=profile_api


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

# Optional debug of parsed topology
if [ "${VLLM_DEBUG_TOPO:-0}" -eq 1 ]; then
  echo "[DEBUG] hl-smi topo -c -N output:"
  hl-smi topo -c -N | sed 's/^/[DEBUG] /'
  for mid in 0 1 2 3 4 5 6 7; do
    eval "echo [DEBUG] MOD $mid CPU_BIND=\${CPU_BIND_$mid} MEM_BIND=\${MEM_BIND_$mid}"
  done
fi

#>>>>>>> kf-fork/deepseek_r1_ww33_kf
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

  # Optional: split CPU binding into non-overlapping subgroups per module
  # Enable with VLLM_SPLIT_CPU_BIND=1. Assumes CPU_BIND has comma-separated
  # subgroups that can be allocated distinctly to modules on the same NUMA.
  if [ "${VLLM_SPLIT_CPU_BIND:-0}" -eq 1 ] && [ -n "$CPU_BIND" ]; then
    IFS=',' read -r -a __cpu_chunks <<< "$(echo "$CPU_BIND" | tr -d ' ')"
    __num_chunks=${#__cpu_chunks[@]}
    if [ "${VLLM_DEBUG_TOPO:-0}" -eq 1 ]; then
      echo "[DEBUG] MOD ${MOD_ID} original CPU_BIND='$CPU_BIND' chunks(${__num_chunks})='${__cpu_chunks[*]}'"
    fi
    if [ $__num_chunks -lt 1 ]; then
      echo "[ERROR] Cannot split CPU_BIND (empty) for module ${MOD_ID}" >&2
      exit 1
    fi
    # Special case: two large ranges shared among 4 modules on a NUMA. Split into 4 subranges.
    if [ $__num_chunks -eq 2 ]; then
      __r0="${__cpu_chunks[0]}"; __r1="${__cpu_chunks[1]}"
      __a0=${__r0%-*}; __b0=${__r0#*-}
      __a1=${__r1%-*}; __b1=${__r1#*-}
      # ensure integers
      __mid0=$(( (__a0 + __b0) / 2 ))
      __mid1=$(( (__a1 + __b1) / 2 ))
      __sub0="${__a0}-${__mid0}"
      __sub1="$((__mid0+1))-${__b0}"
      __sub2="${__a1}-${__mid1}"
      __sub3="$((__mid1+1))-${__b1}"
      __four_chunks=("$__sub0" "$__sub1" "$__sub2" "$__sub3")
      __idx=$(( MOD_ID % 4 ))
      __sel_chunk="${__four_chunks[$__idx]}"
      if [ "${VLLM_DEBUG_TOPO:-0}" -eq 1 ]; then
        echo "[DEBUG] MOD ${MOD_ID} split 2-ranges into 4: '${__four_chunks[*]}', pick idx ${__idx} -> '$__sel_chunk'"
      fi
      CPU_BIND="$__sel_chunk"
      unset __r0 __r1 __a0 __b0 __a1 __b1 __mid0 __mid1 __sub0 __sub1 __sub2 __sub3 __four_chunks __idx
    else
      __idx=$(( MOD_ID % __num_chunks ))
      if [ $__num_chunks -le $__idx ]; then
        echo "[ERROR] Not enough CPU subgroups in CPU_BIND='$CPU_BIND' for module ${MOD_ID}" >&2
        exit 1
      fi
      __sel_chunk="${__cpu_chunks[$__idx]}"
      if [ -z "$__sel_chunk" ]; then
        echo "[ERROR] Selected CPU subgroup is empty for module ${MOD_ID} from '$CPU_BIND'" >&2
        exit 1
      fi
      if [ "${VLLM_DEBUG_TOPO:-0}" -eq 1 ]; then
        echo "[DEBUG] MOD ${MOD_ID} select chunk index ${__idx} -> '$__sel_chunk'"
      fi
      CPU_BIND="$__sel_chunk"
      unset __idx __sel_chunk
    fi
    unset __cpu_chunks __num_chunks
  fi

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
#<<<<<<< HEAD
#    --enable-reasoning
#    --reasoning-parser deepseek_r1
#    --preemption-mode swap
#    --swap-space "$SWAP_SPACE"
#    $kv_cache_dtype_arg
#    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
#  )
#  # Only define log_file if XPYD_LOG is set
#  if [ -n "$XPYD_LOG" ]; then
#    timestamp=$(date +"%Y%m%d_%H%M%S")
#    log_file="$XPYD_LOG/log_rank${RANK}_${timestamp}.log"
#=======
    $kv_cache_dtype_arg
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
  )
  log_file="$log_dir/log_rank${RANK}_${timestamp}.log"
  if [ "$NUMACTL_ENABLED" -eq 1 ]; then
    echo "CPU_BIND: $CPU_BIND"
    echo "MEM_BIND: $MEM_BIND"
    echo "HLS_MODULE_ID: $MOD_ID"
    echo "DP_RANK: $RANK"
#>>>>>>> kf-fork/deepseek_r1_ww33_kf
  fi

  extra_env=()
#  if [ "$i" -eq 0 ] && [ "$RANK" -eq 0 ]; then
#    extra_env+=(VLLM_PROFILER_ENABLED=true)
#  fi

  # Execute command
  if [ "$DP_RANK" -ne 1 ]; then
#<<<<<<< HEAD
#    if [ -n "$XPYD_LOG" ]; then
#      echo "env VLLM_DP_RANK=$RANK ${CMD[*]} (logging to $log_file)"
#      env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${extra_env[@]}" "${CMD[@]}" 2>&1 | tee "$log_file" &
#    else
#      echo "env VLLM_DP_RANK=$RANK ${CMD[*]} (no logging)"
#      env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${extra_env[@]}" "${CMD[@]}" &
#    fi
#  else
#    if [ -n "$XPYD_LOG" ]; then
#      echo "${CMD[*]} (logging to $log_file)"
#      "${CMD[@]}" 2>&1 | tee "$log_file" &
#    else
#      echo "${CMD[*]} (no logging)"
#=======
    if [ "$NUMACTL_ENABLED" -eq 1 ] && [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
      echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]}"
      env HLS_MODULE_ID="$MOD_ID" VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" 2>&1 | tee "$log_file" &
    else
      echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK ${CMD[*]}"
      env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${CMD[@]}" 2>&1 | tee "$log_file" &
    fi
  else
    if [ "$NUMACTL_ENABLED" -eq 1 ] && [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
      echo "numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]}"
      env HLS_MODULE_ID="$MOD_ID" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" &
    else
      echo "${CMD[*]}"
#>>>>>>> kf-fork/deepseek_r1_ww33_kf
      "${CMD[@]}" &
    fi
  fi
done

wait

