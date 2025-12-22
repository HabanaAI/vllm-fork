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
#export VLLM_CPU_BLACKLIST="110-129,350-369"
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
      numa_count[mem]++;                          # count modules per NUMA
    }
    END {
      for (numa in numa_count) {
        printf "MODULES_PER_NUMA_%s=%s; ", numa, numa_count[numa];
      }
    }
  ')"
fi

# Optional debug of parsed topology
if [ "${VLLM_DEBUG_TOPO:-0}" -eq 1 ]; then
  echo "[DEBUG] hl-smi topo -c -N output:"
  hl-smi topo -c -N | sed 's/^/[DEBUG] /'
  # Extract module IDs dynamically from hl-smi output
  MODULE_IDS=$(hl-smi topo -c -N | awk '
    NR<=2 { next }                                  # skip headers
    $1 ~ /^[0-9]+$/ { print $1 }                    # extract module IDs
  ' | sort -n | tr '\n' ' ')
  for mid in $MODULE_IDS; do
    eval "echo [DEBUG] MOD $mid CPU_BIND=\${CPU_BIND_$mid} MEM_BIND=\${MEM_BIND_$mid}"
  done
  # Show modules per NUMA
  for var in $(set | grep '^MODULES_PER_NUMA_' | cut -d= -f1); do
    numa_id=${var#MODULES_PER_NUMA_}
    eval "echo [DEBUG] NUMA $numa_id has \${$var} modules"
  done
fi

# Manual setting for snc-3 on skyriver (6984P-C)
#CPU_BIND_0="0-12,240-252"
#CPU_BIND_1="13-25,253-265"
#CPU_BIND_2="26-38,266-278"
#CPU_BIND_3="40-52,280-292"
#CPU_BIND_4="53-65,293-305"
#CPU_BIND_5="66-78,306-318"
#CPU_BIND_6="80-92,320-332"
#CPU_BIND_7="93-105,333-345"
#CPU_BIND_8="120-132,360-372"
#CPU_BIND_9="133-145,373-385"
#CPU_BIND_10="146-158,386-398"
#CPU_BIND_11="160-172,400-412"
#CPU_BIND_12="173-185,413-425"
#CPU_BIND_13="186-198,426-438"
#CPU_BIND_14="200-212,440-452"
#CPU_BIND_15="213-225,453-465"
#MEM_BIND_0="0"
#MEM_BIND_1="0"
#MEM_BIND_2="0"
#MEM_BIND_3="1"
#MEM_BIND_4="1"
#MEM_BIND_5="1"
#MEM_BIND_6="2"
#MEM_BIND_7="2"
#MEM_BIND_8="3"
#MEM_BIND_9="3"
#MEM_BIND_10="3"
#MEM_BIND_11="4"
#MEM_BIND_12="4"
#MEM_BIND_13="4"
#MEM_BIND_14="5"
#MEM_BIND_15="5"


for ((i=0; i<$DP_RANK; i++))
do
  RANK=$((DP_INDEX * DP_RANK + i))
  port=$((8200 + i))

  # Derive Habana module id for this rank and bind to the corresponding NUMA/CPU
  MOD_ID=$((i % 16))
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
    # Special case: two large ranges shared among modules on a NUMA. Split into subranges.
    if [ $__num_chunks -eq 2 ]; then
      # Get the number of modules per NUMA for this module
      __numa_var="MODULES_PER_NUMA_${MEM_BIND}"
      __modules_per_numa="${!__numa_var}"
      if [ -z "$__modules_per_numa" ]; then
        echo "[WARN] Cannot determine modules per NUMA for module ${MOD_ID} (NUMA ${MEM_BIND}). Using default of 4." >&2
        __modules_per_numa=4
      fi
      __r0="${__cpu_chunks[0]}"; __r1="${__cpu_chunks[1]}"
      __a0=${__r0%-*}; __b0=${__r0#*-}
      __a1=${__r1%-*}; __b1=${__r1#*-}
      # Split into __modules_per_numa subranges, where each subrange combines
      # corresponding parts from both CPU ranges
      __subranges=()
      __range0_size=$(( __b0 - __a0 + 1 ))
      __range1_size=$(( __b1 - __a1 + 1 ))
      __subrange0_size=$(( __range0_size / __modules_per_numa ))
      __subrange1_size=$(( __range1_size / __modules_per_numa ))
      for ((__i=0; __i<__modules_per_numa; __i++)); do
        # Calculate subrange for first CPU range
        __start0=$(( __a0 + __i * __subrange0_size ))
        if [ $__i -eq $((__modules_per_numa - 1)) ]; then
          __end0=$__b0
        else
          __end0=$(( __start0 + __subrange0_size - 1 ))
        fi
        # Calculate subrange for second CPU range
        __start1=$(( __a1 + __i * __subrange1_size ))
        if [ $__i -eq $((__modules_per_numa - 1)) ]; then
          __end1=$__b1
        else
          __end1=$(( __start1 + __subrange1_size - 1 ))
        fi
        # Combine both parts into one subrange
        __subranges+=("${__start0}-${__end0},${__start1}-${__end1}")
      done
      # Select the appropriate subrange for this module
      # Find module index within its NUMA (0-based)
      # Collect all modules on the same NUMA by checking available MEM_BIND variables
      __mods_on_numa=()
      for __check_mod in {0..31}; do
        __check_mem_var="MEM_BIND_${__check_mod}"
        __check_mem="${!__check_mem_var:-}"
        if [ -n "$__check_mem" ] && [ "$__check_mem" = "$MEM_BIND" ]; then
          __mods_on_numa+=($__check_mod)
        fi
      done
      # Find MOD_ID's position in the sorted list
      __mod_idx_in_numa=0
      for __idx_check in "${!__mods_on_numa[@]}"; do
        if [ "${__mods_on_numa[$__idx_check]}" -eq "$MOD_ID" ]; then
          __mod_idx_in_numa=$__idx_check
          break
        fi
      done
      __idx=$(( __mod_idx_in_numa % ${#__subranges[@]} ))
      __sel_chunk="${__subranges[$__idx]}"
      if [ "${VLLM_DEBUG_TOPO:-0}" -eq 1 ]; then
        echo "[DEBUG] MOD ${MOD_ID} (NUMA ${MEM_BIND}, ${__modules_per_numa} modules/NUMA, idx ${__mod_idx_in_numa}) split 2-ranges into ${#__subranges[@]}: '${__subranges[*]}', pick idx ${__idx} -> '$__sel_chunk'"
      fi
      CPU_BIND="$__sel_chunk"
      unset __r0 __r1 __a0 __b0 __a1 __b1 __modules_per_numa __numa_var __subranges __range0_size __range1_size __subrange0_size __subrange1_size __start0 __end0 __start1 __end1 __i __mod_idx_in_numa __idx __sel_chunk __mods_on_numa __check_mod __check_mem_var __check_mem __idx_check
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

