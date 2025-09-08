#!/bin/bash
# Parse command line arguments
BENCHMARK_MODE=false
FIRST_TOKEN_FROM_D=false
KILL_MODE=false
RESTART_MODE=false
REPEAT_D_TIMES=127

while getopts "bdkrt:" opt; do
    case $opt in
        b) BENCHMARK_MODE=true ;;
        d) FIRST_TOKEN_FROM_D=true ;;
        k) KILL_MODE=true ;;
        r) RESTART_MODE=true ;;
        t) REPEAT_D_TIMES=$OPTARG ;;
        *) echo "Usage: $0 [-b] [-d] [-k] [-r] [-t repeat_d_times]" >&2
           echo "  -b: Enable benchmark mode" >&2
           echo "  -d: First token from decode node" >&2
           echo "  -k: Kill proxy server processes" >&2
           echo "  -r: Restart proxy server" >&2
           echo "  -t: Set repeat_d_times for benchmark (default: 127)" >&2
           echo "" >&2
           echo "Example: $0 -b -t 50    # Benchmark mode with 50 decode repeats" >&2
           exit 1 ;;
    esac
done
shift $((OPTIND-1))

# Handle kill mode - kill proxy server and exit
if [ "$KILL_MODE" = true ]; then
    echo "-------------------------------------------------------------------"
    echo "Killing proxy server processes..."
    echo "-------------------------------------------------------------------"
    pkill -KILL -f disagg_proxy_demo
    pkill -KILL -f xpyd_start_proxy.sh
    echo "Proxy server processes killed."
    exit 0
fi

# Handle restart mode - kill and restart proxy server
if [ "$RESTART_MODE" = true ]; then
    echo "-------------------------------------------------------------------"
    echo "Restarting proxy server - killing existing processes..."
    echo "-------------------------------------------------------------------"
    pkill -KILL -f disagg_proxy_demo
    pkill -KILL -f xpyd_start_proxy.sh
    echo "Existing proxy server processes killed."
fi

# Build the command arguments
CMD_ARGS="1 2 1"

# Set first token source argument (arg 4)
if [ "$FIRST_TOKEN_FROM_D" = true ]; then
    CMD_ARGS="$CMD_ARGS false"
else
    CMD_ARGS="$CMD_ARGS true"
fi

# Set repeat_d_times argument (arg 5)
CMD_ARGS="$CMD_ARGS $REPEAT_D_TIMES"

# Add benchmark argument if benchmark mode is enabled (arg 6)
if [ "$BENCHMARK_MODE" = true ]; then
    CMD_ARGS="$CMD_ARGS benchmark"
fi

mkdir -p pd_test_log

log_file="pd_test_log/proxy.log"
echo "-------------------------------------------------------------------"
echo "Being brought to background"
echo "Log will be redirect to $log_file"
echo "Benchmark mode: $BENCHMARK_MODE"
echo "First token from d: $FIRST_TOKEN_FROM_D"
echo "Repeat D times: $REPEAT_D_TIMES"
echo "Command: bash xpyd_start_proxy.sh $CMD_ARGS"
echo "..."
echo "-------------------------------------------------------------------"
bash xpyd_start_proxy.sh $CMD_ARGS >> $log_file 2>&1 &
