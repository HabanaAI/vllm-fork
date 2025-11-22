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

HEAD_ROLE_KEY=${P_KEYS[0]}
HEAD_IP=${ROLE_IP[$HEAD_ROLE_KEY]}
HEAD_ADDR=${HEAD_ADDR:-${HEAD_IP}:6886}
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
echo "PREFILL HEAD_ADDR=${HEAD_ADDR}"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "DRY-RUN MODE: Commands will not be executed."
fi
echo "==================================="

#export PREFILL_NEED_SCALEOUT=${USR_PREFILL_NEED_SCALEOUT:-0}

#sleep 30

echo "Launching prefill workers"
for role_key in "${P_KEYS[@]}"; do
  role_idx=${role_key#P}
  host=${ROLE_HOST[$role_key]}
  ip=${ROLE_IP[$role_key]}
  if [[ $role_key == "$HEAD_ROLE_KEY" ]]; then
    role_type="head"
    delay=2
  else
    role_type="node"
    delay=5
  fi
  echo "Launching prefill ${role_type} on ${host} (${ip})"
  prefill_cmd=(
    ssh
    root@"$ip"
    "cd $BASE_DIR; ROLE=${role_type} ROLE_IDX=$role_idx BENCHMARK_MODE=$BENCHMARK_MODE ENV_FILE=$ENV_FILE HEAD_ADDR=${ROLE_IP[P0]} ./P.sh"
  )
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${prefill_cmd[*]}"
  else
    "${prefill_cmd[@]}" &
  fi
  sleep "$delay"
done

echo "Launching decode workers"
for role_key in "${D_KEYS[@]}"; do
  instance_idx=${role_key#D}
  host=${ROLE_HOST[$role_key]}
  ip=${ROLE_IP[$role_key]}
  echo "Launching decode worker ${role_key} on ${host} (${ip})"
  decode_cmd=(
    ssh
    root@"$ip"
    #"cd $BASE_DIR; DP_MASTER_IP=${ROLE_IP[D0]} DECODE_NEED_SCALEOUT=${USR_DECODE_NEED_SCALEOUT:-1} ./D.sh ${TP_AUTO} $host $instance_idx $ENV_FILE"
    "cd $BASE_DIR; ENV_FILE=$ENV_FILE ./D.sh $instance_idx"

  )
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${decode_cmd[*]}"
  else
    "${decode_cmd[@]}" &
  fi
  sleep 1
done
