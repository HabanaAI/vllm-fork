#!/usr/bin/env bash

#set -euo pipefail

ENV_FILE=${1:-env.sh}
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$BASE_DIR/$ENV_FILE"

nodes=(P0 P1 D0 D1 D2 D3)

for node in "${nodes[@]}"; do
  host_var="ROLE_HOST[$node]"
  ip_var="ROLE_IP[$node]"
  host=${ROLE_HOST[$node]:-}
  ip=${ROLE_IP[$node]:-}
  if [[ -z $ip ]]; then
    echo "Skipping $node; ROLE_IP[$node] not set"
    continue
  fi
  echo "Killing on $node host=${host:-unknown} ip=$ip"
  ssh root@$ip "cd $BASE_DIR; bash ./killall.sh" || echo "Failed to kill on $ip"
done