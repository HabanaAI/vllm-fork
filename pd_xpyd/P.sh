#!/usr/bin/env bash

set -euo pipefail

# Multi-instance support: determine ROLE and ROLE_IDX from instance indices
if [[ -n "${P_INSTANCE_IDX:-}" && -n "${P_INTRA_INSTANCE_IDX:-}" ]]; then
  # New multi-instance mode
  if [[ "${P_INTRA_INSTANCE_IDX}" -eq 0 ]]; then
    ROLE="head"
  else
    ROLE="node"
  fi
  ROLE_IDX=${P_INTRA_INSTANCE_IDX}
  P_INSTANCE_IDX=${P_INSTANCE_IDX}
else
  # Legacy mode: ROLE and ROLE_IDX must be set explicitly
  if [[ -z "$ROLE" || ( "$ROLE" != "head" && "$ROLE" != "node" ) ]]; then
    echo "Error: ROLE must be set to either 'head' or 'node', or P_INSTANCE_IDX and P_INTRA_INSTANCE_IDX must be set" >&2
    exit 1
  fi
  if [[ -z "${ROLE_IDX:-}" ]]; then
    echo "Error: ROLE_IDX must be set, or P_INSTANCE_IDX and P_INTRA_INSTANCE_IDX must be set" >&2
    exit 1
  fi
  P_INSTANCE_IDX=${P_INSTANCE_IDX:-0}
fi

# Check if HEAD_ADDR is set and a valid IP:PORT form
if [[ -z "${HEAD_ADDR:-}" ]]; then
  echo "Error: HEAD_ADDR is not set" >&2
  exit 1
fi
# Basic IP endpoint validation: format IP only (IPv4)
if ! [[ "$HEAD_ADDR" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "Error: HEAD_ADDR ($HEAD_ADDR) is not a valid IPv4 address" >&2
  exit 1
fi

if [[ -z "$ENV_FILE" ]]; then
  echo "Error: ENV_FILE is not set" >&2
  exit 1
fi
if [[ "$ENV_FILE" != /* ]]; then
  ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/$ENV_FILE"
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: ENV_FILE ($ENV_FILE) does not exist" >&2
  exit 1
fi

echo "ENV_FILE=$ENV_FILE"
source "$ENV_FILE"

# This is to avoid open file soft limit
ulimit -S -n 65536

echo "=============================================="
for var in $(compgen -A variable | grep '^USR_'); do
  echo "$var=${!var}"
done
echo "=============================================="

export CARDS_PER_NODE=${USR_CARDS_PER_NODE}
export PLATFORM_TYPE=${USR_PLATFORM_TYPE}
export PREFILL_NEED_SCALEOUT=${USR_PREFILL_NEED_SCALEOUT}
export QUANT_CONFIG_PREFILL=${USR_QUANT_CONFIG_PREFILL}
export PREFILL_TP_SIZE=${USR_PREFILL_TP_SIZE}
export PREFILL_EP_SIZE=${USR_PREFILL_EP_SIZE}
export PREFILL_USE_RAY=${USR_PREFILL_USE_RAY}
# Export instance indices for use in unified_pd_start_prefill.sh
export P_INSTANCE_IDX=${P_INSTANCE_IDX:-0}
export ROLE_IDX=${ROLE_IDX:-0}
export ETCD_META_SERVER=${USR_ETCD_META_SERVER}
export MOONCAKE_SERVER=${USR_MOONCAKE_SERVER}
export MOONCAKE_LOCAL_ADDR_PREFIX=${USR_MOONCAKE_LOCAL_ADDR_PREFIX}

#echo "P Env Vars"
#echo "  CARDS_PER_NODE=$CARDS_PER_NODE"
#echo "  PLATFORM_TYPE=$PLATFORM_TYPE"
#echo "  PREFILL_NEED_SCALEOUT=$PREFILL_NEED_SCALEOUT"

# Stage 1: Local log directory (fast local disk)
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-/workspace/pd_test_log}
mkdir -p "$LOCAL_LOG_DIR"

timestamp=$(TZ="Asia/Shanghai" date +"%Y%m%d_%H%M%S")
# Log file naming: include instance index if multi-instance mode
if [[ -n "${P_INSTANCE_IDX:-}" ]]; then
    log_file="$LOCAL_LOG_DIR/prefill_${PLATFORM_TYPE}_${P_INSTANCE_IDX}_${ROLE_IDX}.log"
    log_file_nfs="${NFS_LOG_DIR:-./pd_test_log}/prefill_${PLATFORM_TYPE}_${P_INSTANCE_IDX}_${ROLE_IDX}.log"
else
    log_file="$LOCAL_LOG_DIR/prefill_${PLATFORM_TYPE}_${ROLE_IDX}.log"
    log_file_nfs="${NFS_LOG_DIR:-./pd_test_log}/prefill_${PLATFORM_TYPE}_${ROLE_IDX}.log"
fi
NFS_LOG_DIR=${NFS_LOG_DIR:-./pd_test_log}
echo "-------------------------------------------------------------------"
echo "Launching prefill $ROLE role"
if [[ -n "${P_INSTANCE_IDX:-}" ]]; then
    echo "Instance: $P_INSTANCE_IDX, Node: $ROLE_IDX"
fi
echo "Stage 1 (local): $log_file"
if [ -n "${NFS_LOG_DIR}" ]; then
    echo "Stage 2 (NFS): $log_file_nfs"
fi
echo "-------------------------------------------------------------------"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
LOG_SYNC="$SCRIPT_DIR/log_sync.py"

# Stage 2: Start log syncer if NFS directory is specified
if [ -n "${NFS_LOG_DIR}" ] && [ -f "$LOG_SYNC" ] && command -v python3 >/dev/null 2>&1; then
    # Start syncer in background (syncs every 2 seconds by default)
    python3 -u "$LOG_SYNC" "$LOCAL_LOG_DIR" "$NFS_LOG_DIR" \
        --sync-interval "${LOG_SYNC_INTERVAL:-1.0}" &
    SYNC_PID=$!
    echo "[P.sh] Log syncer started (PID: $SYNC_PID)"
fi

# Stage 1: Direct redirection to local file (fast, simple)
bash "$SCRIPT_DIR/unified_pd_start_prefill.sh" > "$log_file" 2>&1 &


