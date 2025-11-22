#!/usr/bin/env bash

set -euo pipefail

if [[ -z "$ROLE" || ( "$ROLE" != "head" && "$ROLE" != "node" ) ]]; then
  echo "Error: ROLE must be set to either 'head' or 'node'" >&2
  exit 1
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

#echo "P Env Vars"
#echo "  CARDS_PER_NODE=$CARDS_PER_NODE"
#echo "  PLATFORM_TYPE=$PLATFORM_TYPE"
#echo "  PREFILL_NEED_SCALEOUT=$PREFILL_NEED_SCALEOUT"

# Stage 1: Local log directory (fast local disk)
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-/workspace/pd_test_log}
mkdir -p "$LOCAL_LOG_DIR"

timestamp=$(TZ="Asia/Shanghai" date +"%Y%m%d_%H%M%S")
log_file="$LOCAL_LOG_DIR/prefill_${PLATFORM_TYPE}_${ROLE_IDX}.log"
NFS_LOG_DIR=${NFS_LOG_DIR:-./pd_test_log}
echo "-------------------------------------------------------------------"
echo "Launching prefill $ROLE role"
echo "Stage 1 (local): $log_file"
if [ -n "${NFS_LOG_DIR}" ]; then
    echo "Stage 2 (NFS): ${NFS_LOG_DIR}/prefill_${PLATFORM_TYPE}_${ROLE_IDX}.log"
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


