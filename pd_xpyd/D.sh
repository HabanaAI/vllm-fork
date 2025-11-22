#!/usr/bin/env bash

set -euo pipefail

#if [ $# -lt 3 ]; then
#  echo "Usage: $0 <TP_SIZE> <HOSTNAME> <INSTANCE_IDX> [env_file]" >&2
#  exit 1
#fi

if [ -z "${ENV_FILE:-}" ]; then
  echo "ENV_FILE is not set. Please set ENV_FILE before running this script." >&2
  exit 1
fi
if [[ "$ENV_FILE" != /* ]]; then
  ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/$ENV_FILE"
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "Environment file $ENV_FILE not found" >&2
  exit 1
fi

source "$ENV_FILE"

echo "=============================================="
for var in $(compgen -A variable | grep '^USR_'); do
  echo "$var=${!var}"
done
echo "=============================================="

export DECODE_TP_SIZE=${USR_DECODE_TP_SIZE:-1}
export DECODE_EP_SIZE=${USR_DECODE_EP_SIZE:-16}
#export HOSTNAME=`hostname`
export DECODE_INSTANCE_IDX=$1
export CARDS_PER_NODE=${USR_CARDS_PER_NODE:-8}
export DP_MASTER_IP=${ROLE_IP[D0]}
export PLATFORM_TYPE=${USR_PLATFORM_TYPE:-OAM}
export PREFILL_NEED_SCALEOUT=${USR_PREFILL_NEED_SCALEOUT:-0}
export DECODE_NEED_SCALEOUT=${USR_DECODE_NEED_SCALEOUT:-1}
export QUANT_CONFIG_DECODE=${USR_QUANT_CONFIG_DECODE:-}


#echo "D Env Vars"
#echo "  CARDS_PER_NODE=$CARDS_PER_NODE"
#echo "  DP_MASTER_IP=$DP_MASTER_IP"
#echo "  PLATFORM_TYPE=$PLATFORM_TYPE"
#echo "  PREFILL_NEED_SCALEOUT=$PREFILL_NEED_SCALEOUT"
#echo "  DECODE_NEED_SCALEOUT=$DECODE_NEED_SCALEOUT"

# Stage 1: Local log directory (fast local disk)
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-/workspace/pd_test_log}
mkdir -p "$LOCAL_LOG_DIR"

timestamp=$(TZ="Asia/Shanghai" date +"%Y%m%d_%H%M%S")
log_file="$LOCAL_LOG_DIR/decode_${PLATFORM_TYPE}_${DECODE_INSTANCE_IDX}.log"
NFS_LOG_DIR=${NFS_LOG_DIR:-./pd_test_log}
echo "-------------------------------------------------------------------"
echo "Being brought to background"
echo "Stage 1 (local): $log_file"
if [ -n "${NFS_LOG_DIR}" ]; then
    echo "Stage 2 (NFS): ${NFS_LOG_DIR}/decode_${PLATFORM_TYPE}_${DECODE_INSTANCE_IDX}.log"
fi
echo "-------------------------------------------------------------------"
echo "[D.sh]CARDS_PER_NODE=$CARDS_PER_NODE"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
LOG_SYNC="$SCRIPT_DIR/log_sync.py"

# Stage 2: Start log syncer if NFS directory is specified
if [ -n "${NFS_LOG_DIR}" ] && [ -f "$LOG_SYNC" ] && command -v python3 >/dev/null 2>&1; then
    # Start syncer in background (syncs every 2 seconds by default)
    python3 -u "$LOG_SYNC" "$LOCAL_LOG_DIR" "$NFS_LOG_DIR" \
        --sync-interval "${LOG_SYNC_INTERVAL:-1.0}" &
    SYNC_PID=$!
    echo "[D.sh] Log syncer started (PID: $SYNC_PID)"
fi

# Stage 1: Direct redirection to local file (fast, simple)
bash "$SCRIPT_DIR/unified_pd_start_decode.sh" "$DECODE_INSTANCE_IDX" > "$log_file" 2>&1 &


