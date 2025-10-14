#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Combined Benchmark Script: QPS + Max Concurrency Testing
# =============================================================================

# Configuration
HOST="172.26.47.98"
PORT="32628"
MODEL="Meta-Llama-3.1-8B-Instruct"
TOKENIZER="/model-cache/Meta-Llama-3.1-8B-Instruct"
INPUT_LEN=8192
OUTPUT_LEN=256

# Test configurations
QPS_VALUES=(0.1 0.25 0.5 1 2 3 4 5)
QPS_NUM_PROMPTS=(32 64 128 256 256 256 256 256)
CONCURRENCY_VALUES=(1 2 4 8 16 32 64 128)
CONCURRENCY_NUM_PROMPTS=256

# Log directories
QPS_LOG_DIR="bmlogs_qps_$(date +%Y%m%d)"
CONCURRENCY_LOG_DIR="bmlogs_concurrency_$(date +%Y%m%d)"
RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# Helper Functions
# =============================================================================

create_directories() {
    mkdir -p "$QPS_LOG_DIR" "$CONCURRENCY_LOG_DIR" "$RESULTS_DIR"
    echo "Created log directories:"
    echo "  QPS tests: $QPS_LOG_DIR"
    echo "  Concurrency tests: $CONCURRENCY_LOG_DIR"
    echo "  Results summary: $RESULTS_DIR"
    echo
}

log_test_start() {
    local test_type="$1"
    local params="$2"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] Starting $test_type test: $params"
}

run_benchmark() {
    local logfile="$1"
    local test_params="$2"
    shift 2
    
    echo "$test_params" | tee "$logfile"
    
    vllm bench serve \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --seed "$(date +%s)" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --burstiness 100 \
        --backend openai \
        --endpoint /v1/completions \
        --ignore-eos \
        "$@" \
        2>&1 | tee -a "$logfile"
}

# =============================================================================
# QPS Testing
# =============================================================================

run_qps_tests() {
    echo "=========================================="
    echo "Starting QPS Testing"
    echo "=========================================="
    
    # Sanity check
    if [ "${#QPS_VALUES[@]}" -ne "${#QPS_NUM_PROMPTS[@]}" ]; then
        echo "❌ QPS_VALUES and QPS_NUM_PROMPTS arrays must have the same length"
        exit 1
    fi
    
    for i in "${!QPS_VALUES[@]}"; do
        local qps=${QPS_VALUES[$i]}
        local num_prompts=${QPS_NUM_PROMPTS[$i]}
        local timestamp=$(date +"%Y%m%d_%H%M%S")
        local qps_clean=${qps//./p}  # Replace . with p for filename
        local logfile="${QPS_LOG_DIR}/qps_${qps_clean}_prompts_${num_prompts}_${timestamp}.log"
        
        local test_params="[$(date +"%Y-%m-%d %H:%M:%S")] QPS Test - input=${INPUT_LEN}, output=${OUTPUT_LEN}, qps=${qps}, num_prompts=${num_prompts}"
        
        log_test_start "QPS" "qps=${qps}, prompts=${num_prompts}"
        
        run_benchmark "$logfile" "$test_params" \
            --num-prompts "$num_prompts" \
            --request-rate "$qps"
        
        echo "✅ QPS test completed: qps=${qps}, saved to ${logfile}"
        echo
    done
}

# =============================================================================
# Max Concurrency Testing
# =============================================================================

run_concurrency_tests() {
    echo "=========================================="
    echo "Starting Max Concurrency Testing"
    echo "=========================================="
    
    for mc in "${CONCURRENCY_VALUES[@]}"; do
        local timestamp=$(date +"%Y%m%d_%H%M%S")
        local logfile="${CONCURRENCY_LOG_DIR}/concurrency_${mc}_prompts_${CONCURRENCY_NUM_PROMPTS}_${timestamp}.log"
        
        local test_params="[$(date +"%Y-%m-%d %H:%M:%S")] Concurrency Test - input=${INPUT_LEN}, output=${OUTPUT_LEN}, max_concurrency=${mc}, request_rate=inf, num_prompts=${CONCURRENCY_NUM_PROMPTS}"
        
        log_test_start "Concurrency" "max_concurrency=${mc}, prompts=${CONCURRENCY_NUM_PROMPTS}"
        
        run_benchmark "$logfile" "$test_params" \
            --num-prompts "$CONCURRENCY_NUM_PROMPTS" \
            --request-rate "inf" \
            --max-concurrency "$mc"
        
        echo "✅ Concurrency test completed: max_concurrency=${mc}, saved to ${logfile}"
        echo
    done
}

# =============================================================================
# Results Summary
# =============================================================================

generate_summary() {
    local summary_file="${RESULTS_DIR}/benchmark_summary.txt"
    
    cat > "$summary_file" << EOF
Benchmark Test Summary
Generated: $(date +"%Y-%m-%d %H:%M:%S")

Configuration:
  Host: $HOST:$PORT
  Model: $MODEL
  Input Length: $INPUT_LEN
  Output Length: $OUTPUT_LEN

QPS Tests:
  Log Directory: $QPS_LOG_DIR
  QPS Values: ${QPS_VALUES[*]}
  Num Prompts: ${QPS_NUM_PROMPTS[*]}

Concurrency Tests:
  Log Directory: $CONCURRENCY_LOG_DIR
  Concurrency Values: ${CONCURRENCY_VALUES[*]}
  Num Prompts: $CONCURRENCY_NUM_PROMPTS

Log Files:
EOF

    echo "QPS Test Logs:" >> "$summary_file"
    find "$QPS_LOG_DIR" -name "*.log" -printf "  %f\n" | sort >> "$summary_file"
    
    echo >> "$summary_file"
    echo "Concurrency Test Logs:" >> "$summary_file"
    find "$CONCURRENCY_LOG_DIR" -name "*.log" -printf "  %f\n" | sort >> "$summary_file"
    
    echo "📋 Summary saved to: $summary_file"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo "🚀 Starting Combined Benchmark Tests"
    echo "Configuration: ${HOST}:${PORT}, Input: ${INPUT_LEN}, Output: ${OUTPUT_LEN}"
    echo
    
    create_directories
    
    # Run tests
    run_qps_tests
    run_concurrency_tests
    
    # Generate summary
    generate_summary
    
    echo "=========================================="
    echo "🎉 All benchmark tests completed!"
    echo "=========================================="
    echo "QPS logs: $QPS_LOG_DIR"
    echo "Concurrency logs: $CONCURRENCY_LOG_DIR"
    echo "Summary: $RESULTS_DIR"
    echo
    echo "Quick verification commands:"
    echo "  ls -la $QPS_LOG_DIR/"
    echo "  ls -la $CONCURRENCY_LOG_DIR/"
    echo "  cat $RESULTS_DIR/benchmark_summary.txt"
}

# Handle script arguments
case "${1:-all}" in
    "qps")
        create_directories
        run_qps_tests
        ;;
    "concurrency")
        create_directories
        run_concurrency_tests
        ;;
    "all"|"")
        main
        ;;
    *)
        echo "Usage: $0 [qps|concurrency|all]"
        echo "  qps         - Run only QPS tests"
        echo "  concurrency - Run only concurrency tests"
        echo "  all         - Run both test suites (default)"
        exit 1
        ;;
esac
