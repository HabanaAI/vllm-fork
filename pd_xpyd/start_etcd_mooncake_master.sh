#!/bin/bash

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_env.sh

pkill -f mooncake_master
pkill -f etcd
sleep 5s

# Define commands as arrays
ETCD_CMD=(etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379)
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

