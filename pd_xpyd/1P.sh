mkdir -p pd_test_log

log_file="pd_test_log/prefill.log"
echo "-------------------------------------------------------------------"
echo "Being brought to background"
echo "Log will be redirect to $log_file"
echo "..."
echo "-------------------------------------------------------------------"

# Accept benchmark mode as an optional argument (default: not benchmark)
BENCHMARK_MODE=${1:-""}

if [ -n "$BENCHMARK_MODE" ]; then
    echo "Benchmark mode argument detected: $BENCHMARK_MODE"
    bash ./1p_start_prefill.sh G3D-sys03 benchmark >> $log_file 2>&1 &
else
    echo "No benchmark mode argument, running in normal mode"
    bash ./1p_start_prefill.sh G3D-sys03 >> $log_file 2>&1 &
fi
