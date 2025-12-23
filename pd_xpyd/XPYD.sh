#!/usr/bin/env bash

set -euo pipefail

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

P_ARGS=
DRY_RUN=0
while getopts ":bn" opt; do
  case "$opt" in
    b)
      P_ARGS="benchmark"
      ;;
    n)
      DRY_RUN=1
      ;;
    *)
      echo "Usage: $0 [-b] [-n]" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND-1))

ENV_FILE=${1:-env_2p4d_sedv+.sh}
source "$BASE_DIR/$ENV_FILE"

# Collect P (prefill) and D (decode) roles from the environment file
declare -a P_KEYS=()
declare -a D_KEYS=()

for role_key in "${!ROLE_HOST[@]}"; do
  if [[ $role_key =~ ^P[0-9]+$ ]]; then
    if [[ -z ${ROLE_IP[$role_key]:-} ]]; then
      echo "Error: ROLE_IP[$role_key] must be set in $ENV_FILE" >&2
      exit 1
    fi
    P_KEYS+=("$role_key")
  elif [[ $role_key =~ ^D[0-9]+$ ]]; then
    if [[ -z ${ROLE_IP[$role_key]:-} ]]; then
      echo "Error: ROLE_IP[$role_key] must be set in $ENV_FILE" >&2
      exit 1
    fi
    D_KEYS+=("$role_key")
  fi
done

if [[ ${#P_KEYS[@]} -eq 0 ]]; then
  echo "Error: No prefill roles (Px) defined in $ENV_FILE" >&2
  exit 1
fi

if [[ ${#D_KEYS[@]} -eq 0 ]]; then
  echo "Error: No decode roles (Dx) defined in $ENV_FILE" >&2
  exit 1
fi

# Sort keys numerically based on their suffix so we launch in index order
IFS=$'\n' P_KEYS=($(printf '%s\n' "${P_KEYS[@]}" | sort -V))
IFS=$'\n' D_KEYS=($(printf '%s\n' "${D_KEYS[@]}" | sort -V))
unset IFS

CARDS_PER_NODE=${USR_CARDS_PER_NODE:-8}
TP_AUTO=${USR_TP_SIZE:-1}

# Multi-instance configuration
P_NUM_INSTANCE=${P_NUM_INSTANCE:-1}
D_NUM_INSTANCE=${D_NUM_INSTANCE:-1}

# Validate instance configuration
if [[ ${#P_KEYS[@]} -lt $P_NUM_INSTANCE ]]; then
  echo "Error: Not enough P nodes (${#P_KEYS[@]}) for $P_NUM_INSTANCE instances" >&2
  exit 1
fi
if [[ ${#D_KEYS[@]} -lt $D_NUM_INSTANCE ]]; then
  echo "Error: Not enough D nodes (${#D_KEYS[@]}) for $D_NUM_INSTANCE instances" >&2
  exit 1
fi

# Calculate nodes per instance
P_NODES_PER_INSTANCE=$(( ${#P_KEYS[@]} / P_NUM_INSTANCE ))
D_NODES_PER_INSTANCE=$(( ${#D_KEYS[@]} / D_NUM_INSTANCE ))

# Validate even distribution
if [[ $(( ${#P_KEYS[@]} % P_NUM_INSTANCE )) -ne 0 ]]; then
  echo "Warning: P nodes (${#P_KEYS[@]}) not evenly divisible by P_NUM_INSTANCE ($P_NUM_INSTANCE)" >&2
fi
if [[ $(( ${#D_KEYS[@]} % D_NUM_INSTANCE )) -ne 0 ]]; then
  echo "Warning: D nodes (${#D_KEYS[@]}) not evenly divisible by D_NUM_INSTANCE ($D_NUM_INSTANCE)" >&2
fi

BENCHMARK_MODE=$P_ARGS

echo "==== Host and IP Configuration ===="
for role_key in "${P_KEYS[@]}"; do
  printf "%-3s: Host=%s, IP=%s\n" "$role_key" "${ROLE_HOST[$role_key]}" "${ROLE_IP[$role_key]}"
done
for role_key in "${D_KEYS[@]}"; do
  printf "%-3s: Host=%s, IP=%s\n" "$role_key" "${ROLE_HOST[$role_key]}" "${ROLE_IP[$role_key]}"
done
echo "CARDS_PER_NODE=${CARDS_PER_NODE}"
echo "TP_AUTO=${TP_AUTO}"
echo "BASE_DIR=${BASE_DIR}"
echo "BENCHMARK_MODE=${BENCHMARK_MODE}"
echo "P_NUM_INSTANCE=${P_NUM_INSTANCE} (${P_NODES_PER_INSTANCE} nodes per instance)"
echo "D_NUM_INSTANCE=${D_NUM_INSTANCE} (${D_NODES_PER_INSTANCE} nodes per instance)"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "DRY-RUN MODE: Commands will not be executed."
fi
echo "==================================="

#export PREFILL_NEED_SCALEOUT=${USR_PREFILL_NEED_SCALEOUT:-0}

#sleep 30

echo "Launching prefill workers"
p_idx=0
for p_instance_idx in $(seq 0 $((P_NUM_INSTANCE - 1))); do
  instance_start=$((p_instance_idx * P_NODES_PER_INSTANCE))
  instance_end=$((instance_start + P_NODES_PER_INSTANCE))
  
  # Get the head node for this instance (first node)
  instance_head_key=${P_KEYS[$instance_start]}
  instance_head_ip=${ROLE_IP[$instance_head_key]}
  instance_head_addr=${instance_head_ip}:${RAY_HEAD_PORT:-6886}
  
  echo "--- Prefill Instance $p_instance_idx (nodes ${instance_start}-$((instance_end - 1))) ---"
  
  for intra_idx in $(seq 0 $((P_NODES_PER_INSTANCE - 1))); do
    global_idx=$((instance_start + intra_idx))
    if [[ $global_idx -ge ${#P_KEYS[@]} ]]; then
      break
    fi
    
    role_key=${P_KEYS[$global_idx]}
    host=${ROLE_HOST[$role_key]}
    ip=${ROLE_IP[$role_key]}
    
    if [[ $intra_idx -eq 0 ]]; then
      role_type="head"
      delay=2
    else
      role_type="node"
      delay=5
    fi
    
    echo "Launching prefill instance $p_instance_idx, node $intra_idx (${role_type}) on ${host} (${ip})"
    prefill_cmd=(
      ssh
      root@"$ip"
      "ROLE=${role_type} P_INSTANCE_IDX=$p_instance_idx P_INTRA_INSTANCE_IDX=$intra_idx BENCHMARK_MODE=$BENCHMARK_MODE ENV_FILE=$ENV_FILE HEAD_ADDR=$instance_head_ip $BASE_DIR/P.sh"
    )
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[DRY-RUN] ${prefill_cmd[*]}"
    else
      "${prefill_cmd[@]}" &
    fi
    sleep "$delay"
  done
done

echo "Launching decode workers"
for d_instance_idx in $(seq 0 $((D_NUM_INSTANCE - 1))); do
  instance_start=$((d_instance_idx * D_NODES_PER_INSTANCE))
  instance_end=$((instance_start + D_NODES_PER_INSTANCE))
  
  # Get the master node for this instance (first node)
  instance_master_key=${D_KEYS[$instance_start]}
  instance_master_ip=${ROLE_IP[$instance_master_key]}
  
  echo "--- Decode Instance $d_instance_idx (nodes ${instance_start}-$((instance_end - 1))) ---"
  
  for intra_idx in $(seq 0 $((D_NODES_PER_INSTANCE - 1))); do
    global_idx=$((instance_start + intra_idx))
    if [[ $global_idx -ge ${#D_KEYS[@]} ]]; then
      break
    fi
    
    role_key=${D_KEYS[$global_idx]}
    host=${ROLE_HOST[$role_key]}
    ip=${ROLE_IP[$role_key]}
    
    echo "Launching decode instance $d_instance_idx, node $intra_idx on ${host} (${ip})"
    decode_cmd=(
      ssh
      root@"$ip"
      "ENV_FILE=$ENV_FILE D_INSTANCE_IDX=$d_instance_idx D_INTRA_INSTANCE_IDX=$intra_idx D_INSTANCE_MASTER_IP=$instance_master_ip $BASE_DIR/D.sh"
    )
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[DRY-RUN] ${decode_cmd[*]}"
    else
      "${decode_cmd[@]}" &
    fi
    sleep 1
  done
done
