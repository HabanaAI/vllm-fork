#!/bin/bash

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_env.sh
# Kill processes matching the executables, but exclude this script and parent shell
# Save current PID to exclude it
SCRIPT_PID=$$
PARENT_PID=$PPID

# Kill mooncake_master processes (excluding this script)
pgrep -f 'mooncake_master' | grep -v "^${SCRIPT_PID}$" | grep -v "^${PARENT_PID}$" | xargs -r kill 2>/dev/null || true

# Kill etcd processes (excluding this script)  
pgrep -f '^[^ ]*/etcd ' | grep -v "^${SCRIPT_PID}$" | grep -v "^${PARENT_PID}$" | xargs -r kill 2>/dev/null || true

sleep 5s

data_dir="default_`hostname`.etcd"
echo data_dir:$data_dir
# Define commands as arrays
ETCD_CMD=(etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379 --data-dir $data_dir)
MOON_CMD=(mooncake_master -max_threads 64 -port 50001 -eviction_high_watermark_ratio 0.8 -eviction_ratio 0.2)
# Check if XPYD_LOG is set
if [ -n "$XPYD_LOG" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    # Run etcd with logging
    ETCD_LOG="$XPYD_LOG/etcd_${timestamp}.log"
    echo "Starting etcd, logging to $ETCD_LOG..."
    "${ETCD_CMD[@]}" > "$ETCD_LOG" 2>&1 &

    # Run mooncake_master with logging
    MOON_LOG="$XPYD_LOG/mooncake_master_${timestamp}.log"
    echo "Starting mooncake_master, logging to $MOON_LOG..."
    "${MOON_CMD[@]}" > "$MOON_LOG" 2>&1 &
else
    # Run without logging
    echo "XPYD_LOG not set, running without logging..."
    "${ETCD_CMD[@]}" > /dev/null 2>&1 &
    "${MOON_CMD[@]}" > /dev/null 2>&1 &
fi


