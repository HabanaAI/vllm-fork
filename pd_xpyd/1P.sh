mkdir -p pd_test_log

log_file="pd_test_log/prefill.log"
echo "-------------------------------------------------------------------"
echo "Being brought to background"
echo "Log will be redirect to $log_file"
echo "..."
echo "-------------------------------------------------------------------"

# Simple log rotation if log exceeds 10MB; keep current log path the same
MAX_SIZE=$((50*1024*1024))
if [ -f "$log_file" ]; then
    cur_size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
    if [ "$cur_size" -gt "$MAX_SIZE" ]; then
        ts=$(TZ="Asia/Shanghai" date +"%Y%m%d_%H%M%S")
        mv "$log_file" "${log_file}.${ts}"
        : > "$log_file"
    fi
fi

# Accept hostname and benchmark mode as optional arguments
# Usage: ./1P.sh [HOSTNAME] [BENCHMARK_MODE]
HOSTNAME=${1:-G3D-sys03}
BENCHMARK_MODE=${2:-""}

if [ -n "$BENCHMARK_MODE" ]; then
    echo "Benchmark mode argument detected: $BENCHMARK_MODE"
    echo "Using hostname: $HOSTNAME"
    bash ./1p_start_prefill.sh "$HOSTNAME" benchmark >> $log_file 2>&1 &
else
    echo "No benchmark mode argument, running in normal mode"
    echo "Using hostname: $HOSTNAME"
    bash ./1p_start_prefill.sh "$HOSTNAME" >> $log_file 2>&1 &
fi
