#!/bin/bash
set -euo pipefail
#set -x

# machine id, EP, TP, DP Index, DP Host IP
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/dp_d_env.sh

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="xpyd_logs"
mkdir -p "$log_dir"

export MOONCAKE_CONFIG_PATH="$BASH_DIR"/mooncake_`hostname`.json
DECODE_DP_SIZE=$((DECODE_EP_SIZE / DECODE_TP_SIZE))
DP_RANK=$((CARDS_PER_NODE / DECODE_TP_SIZE))
if [ -z "${1:-}" ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <DP_INDEX (integer)>" >&2
  exit 1
fi
DP_INDEX=$1

echo "============================================================"
echo "                 [ DP Start Decode CONFIG ]                "
echo "============================================================"
echo "🌙 MOONCAKE_CONFIG_PATH : $MOONCAKE_CONFIG_PATH"
echo "🔢 DECODE_EP_SIZE       : $DECODE_EP_SIZE"
echo "🔢 DECODE_TP_SIZE       : $DECODE_TP_SIZE"
echo "🔢 DECODE_DP_SIZE       : $DECODE_DP_SIZE"
echo "💻 CARDS_PER_NODE       : $CARDS_PER_NODE"
echo "🏅 DP_RANK              : $DP_RANK"
echo "🆔 DP_INDEX             : $DP_INDEX"
echo "🌐 DP_MASTER_IP         : $DP_MASTER_IP"
echo "============================================================"


export VLLM_DP_SIZE=$DECODE_DP_SIZE
export VLLM_DP_MASTER_IP=$DP_MASTER_IP
export VLLM_EP_SIZE=$DECODE_EP_SIZE


if [ "$DECODE_DP_SIZE" -eq 1 ]; then
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

#export VLLM_TORCH_PROFILER_DIR=./profiles
#export VLLM_PROFILER_ENABLED=true
#export VLLM_PROFILE_CONFIG_PATH=profile_config.json
#export HABANA_PROFILE_WRITE_HLTV=1
#export HABANA_PROFILE=profile_api_with_nics
#export HABANA_PROFILE=profile_api

# Control whether to apply numactl bindings (1=enable, 0=disable)
NUMACTL_ENABLED=${VLLM_USE_NUMACTL:-1}

# Optional blacklist of CPUs (comma-separated list of cores or ranges).
# Example: export VLLM_CPU_BLACKLIST="0-3,120-123"
export VLLM_CPU_BLACKLIST="110-129,350-369"
CPU_BLACKLIST_RAW=${VLLM_CPU_BLACKLIST:-}

expand_cpu_list() {
  local raw_list=$1
  local part start end
  IFS=',' read -r -a parts <<< "$(echo "$raw_list" | tr -d ' ')"
  for part in "${parts[@]}"; do
    if [[ -z "$part" ]]; then
      continue
    fi
    if [[ $part =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=${BASH_REMATCH[1]}
      end=${BASH_REMATCH[2]}
      for ((cpu=start; cpu<=end; cpu++)); do
        printf '%s\n' "$cpu"
      done
    elif [[ $part =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$part"
    fi
  done
}

apply_cpu_blacklist() {
  local cpu_list=$1
  if [[ -z "$CPU_BLACKLIST_RAW" || -z "$cpu_list" ]]; then
    printf '%s' "$cpu_list"
    return
  fi

  # Build blacklist set
  declare -A blacklist_set=()
  while read -r bl_cpu; do
    [[ -z "$bl_cpu" ]] && continue
    blacklist_set[$bl_cpu]=1
  done < <(expand_cpu_list "$CPU_BLACKLIST_RAW")

  # Filter whitelist
  mapfile -t whitelist < <(expand_cpu_list "$cpu_list")
  if [[ ${#whitelist[@]} -eq 0 ]]; then
    printf ''
    return
  fi

  local filtered=()
  for cpu in "${whitelist[@]}"; do
    if [[ -z ${blacklist_set[$cpu]+_} ]]; then
      filtered+=("$cpu")
    fi
  done

  if [[ ${#filtered[@]} -eq 0 ]]; then
    printf ''
    return
  fi

  local out=""
  local start=${filtered[0]}
  local prev=$start
  local idx cur

  for ((idx=1; idx<${#filtered[@]}; idx++)); do
    cur=${filtered[$idx]}
    if (( cur == prev + 1 )); then
      prev=$cur
      continue
    fi
    if (( start == prev )); then
      out+="$start,"
    else
      out+="$start-$prev,"
    fi
    start=$cur
    prev=$cur
  done

  if (( start == prev )); then
    out+="$start"
  else
    out+="$start-$prev"
  fi

  printf '%s' "$out"
}

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

  if [ -n "$CPU_BIND" ]; then
    CPU_BIND=$(apply_cpu_blacklist "$CPU_BIND")
    if [ -z "$CPU_BIND" ]; then
      echo "[WARN] CPU binding for module ${MOD_ID} became empty after applying blacklist. Disabling numactl for this rank." >&2
      NUMACTL_ENABLED=0
    fi
  fi

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
    -tp "$DECODE_TP_SIZE"
    --max-num-seqs "$max_num_seqs"
    --trust-remote-code
    --disable-log-requests
    --max-num-batched-tokens "$max_num_batched_tokens"
    --use-padding-aware-scheduling
    --use-v2-block-manager
    --distributed_executor_backend mp
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
    $kv_cache_dtype_arg
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
  )
  log_file="$log_dir/log_rank${RANK}_${timestamp}.log"
  if [ "$NUMACTL_ENABLED" -eq 1 ]; then
    echo "CPU_BIND: $CPU_BIND"
    echo "MEM_BIND: $MEM_BIND"
    echo "HLS_MODULE_ID: $MOD_ID"
    echo "DP_RANK: $RANK"
  fi

  extra_env=()

  # Execute command
  if [ "$DP_RANK" -ne 1 ]; then
    if [ "$NUMACTL_ENABLED" -eq 1 ] && [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
      echo "env HLS_MODULE_ID=$MOD_ID VLLM_DP_RANK=$RANK numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]}"
      env HLS_MODULE_ID="$MOD_ID" VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" 2>&1 | tee "$log_file" &
    else
      echo "VLLM_DP_RANK_LOCAL=$i VLLM_DP_RANK=$RANK ${CMD[*]}"
      env VLLM_DP_RANK_LOCAL="$i" VLLM_DP_RANK="$RANK" "${CMD[@]}" 2>&1 | tee "$log_file" &
    fi
  else
    if [ "$NUMACTL_ENABLED" -eq 1 ] && [ -n "$CPU_BIND" ] && [ -n "$MEM_BIND" ]; then
      echo "numactl -C $CPU_BIND -m $MEM_BIND ${CMD[*]}"
      env HLS_MODULE_ID="$MOD_ID" numactl -C "$CPU_BIND" -m "$MEM_BIND" "${CMD[@]}" &
    else
      echo "${CMD[*]}"
      "${CMD[@]}" &
    fi
  fi
done

wait

