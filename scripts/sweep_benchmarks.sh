#!/bin/bash
set -x

# Usage: sweep_benchmarks.sh [NUM_NODES] [EVAL_TASKS]
NUM_NODES=${1:-1}
EVAL_TASKS=${2:-}
echo "Starting benchmark sweeper with NUM_NODES=${NUM_NODES}"

python3 -m pip install datasets

# Multi-node pre-check
if [ "$NUM_NODES" -gt 1 ]; then
  echo "[Warning] Multi-node mode: Ensure ray cluster is started via run_cluster.sh."

  # Verify Ray HPUs
  TOTAL_HPU=$((8 * NUM_NODES))
  if ray status | grep -q "0.0/${TOTAL_HPU}.0 HPU"; then
    echo "Ray cluster ready with ${TOTAL_HPU} HPUs."
  else
    echo "Ray cluster not ready; expected ${TOTAL_HPU} HPUs. Exiting."
    exit 1
  fi
fi

# Prepare logging
BASE_LOG_DIR=$(pwd)/logs/$(date +"%Y%m%d")/${NUM_NODES}-node
mkdir -p "$BASE_LOG_DIR"
SUMMARY_LOG=${BASE_LOG_DIR}/summary.log

# Build header
HEADER='nodes,pp_size,tp_size,comm_backend,kv_dtype,partition,max_model_len,input_tokens,output_tokens,num_prompts,max_concurrency,mean_ttft,mean_tpot,total_throughput,output_throughput'
if [[ -n "$EVAL_TASKS" && "$EVAL_TASKS" != "none" ]]; then
  IFS=',' read -ra TASKS <<< "$EVAL_TASKS"
  for idx in "${!TASKS[@]}"; do
    HEADER+",eval_task_$((idx+1))"
  done
fi

echo "$HEADER" | tee -a "$SUMMARY_LOG"

# Default client/server parameters
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8688}
MODEL_PATH=${MODEL_PATH:-/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2}

# Unified config list: max_len,input,output,num_prompts,conc,pp,tp,backend,partition
KV=auto

config_list=(
  "8192,2048,2048,72,24,2,4,gloo,${KV},[32,29]"
  "8192,2048,2048,96,32,2,4,gloo,${KV},[32,29]"
  "8192,2048,2048,120,40,2,4,gloo,${KV},[32,29]"
  "8192,2048,2048,72,24,2,4,gloo,fp8_inc,[32,29]"
  "8192,2048,2048,72,24,2,4,hccl,${KV},[32,29]"
  "16384,8192,8192,24,8,2,4,gloo,${KV},[32,29]"
  "16384,8192,8192,32,12,2,4,gloo,${KV},[32,29]"
  "16384,8192,8192,48,16,2,4,gloo,${KV},[32,29]"
)

for cfg in "${config_list[@]}"; do
  IFS=',' read -r MAX_MODEL_LEN INPUT_TOKENS OUTPUT_TOKENS NUM_PROMPTS MAX_CONCURRENCY PP_SIZE TP_SIZE COMM_BACKEND KV_CACHE_DTYPE PARTITION <<< "$cfg"
  PARTITION=$(echo $PARTITION | tr -d '[]')
  LOG_PREFIX=${MAX_MODEL_LEN}_in${INPUT_TOKENS}_out${OUTPUT_TOKENS}_conc${MAX_CONCURRENCY}

  CONFIG_LOG_DIR=${BASE_LOG_DIR}
  CONFIG_LOG_DIR=${CONFIG_LOG_DIR}/kv_${KV_CACHE_DTYPE}
  mkdir -p "$CONFIG_LOG_DIR"
  CONFIG_LOG_DIR=${CONFIG_LOG_DIR}/tp${TP_SIZE}_pp${PP_SIZE}
  mkdir -p "$CONFIG_LOG_DIR"
  CONFIG_LOG_DIR=${CONFIG_LOG_DIR}/pp_comm_${COMM_BACKEND}
  mkdir -p "$CONFIG_LOG_DIR"

  # Kill any existing OpenAI server processes
  ps -ef | grep openai | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

  # Launch server
  if [ "$MAX_CONCURRENCY" = "1" ]; then
    PER_PP_CONCURRENCY=1
  else
    PER_PP_CONCURRENCY=$((MAX_CONCURRENCY / PP_SIZE))
  fi
  bash -x benchmark_server_param.sh ${NUM_NODES} ${MAX_MODEL_LEN} ${PER_PP_CONCURRENCY} ${TP_SIZE} ${PP_SIZE} ${COMM_BACKEND} "${PARTITION}" ${KV_CACHE_DTYPE} ${HOST} ${PORT} ${MODEL_PATH}\
    > ${CONFIG_LOG_DIR}/${LOG_PREFIX}_server.log 2>&1 &
  SERVER_PID=$!

  # Wait for server startup
  connected=0
  timeout=900
  interval=5
  start_time=$(date +%s)
  while :; do
    if grep -q "Application startup complete" ${CONFIG_LOG_DIR}/${LOG_PREFIX}_server.log; then
      connected=1; break
    fi
    if grep -q "Fatal Python error" ${CONFIG_LOG_DIR}/${LOG_PREFIX}_server.log; then
      echo "[Error] Server startup failed with fatal error."; break
    fi
    if (( $(date +%s) - start_time >= timeout )); then
      echo "[Timeout] Server did not start within ${timeout}s."; break
    fi
    sleep $interval
  done

  if [ "$connected" -eq 1 ]; then
    # Run client benchmark
    source benchmark_client_param.sh
    # manual warmup run
    test_benchmark_client_serving ${INPUT_TOKENS} ${OUTPUT_TOKENS} ${MAX_CONCURRENCY} ${NUM_PROMPTS} 0.8 ${HOST} ${PORT} ${MODEL_PATH} ${CONFIG_LOG_DIR} \
      | tee -a ${CONFIG_LOG_DIR}/${LOG_PREFIX}_benchmark.log
    # recorded run
    #curl -X POST http://${HOST}:${PORT}/start_profile
    test_benchmark_client_serving ${INPUT_TOKENS} ${OUTPUT_TOKENS} ${MAX_CONCURRENCY} ${NUM_PROMPTS} 0.8 ${HOST} ${PORT} ${MODEL_PATH} ${CONFIG_LOG_DIR} \
      | tee -a ${CONFIG_LOG_DIR}/${LOG_PREFIX}_benchmark.log
    #curl -X POST http://${HOST}:${PORT}/stop_profile

    # Collect metrics
    mean_ttft=$(grep 'Mean TTFT (ms):' ${CONFIG_LOG_DIR}/${LOG_PREFIX}_benchmark.log | tail -1 | awk '{print $NF}')
    mean_tpot=$(grep 'Mean TPOT (ms):' ${CONFIG_LOG_DIR}/${LOG_PREFIX}_benchmark.log | tail -1 | awk '{print $NF}')
    total_throughput=$(grep 'Total Token throughput (tok/s):' ${CONFIG_LOG_DIR}/${LOG_PREFIX}_benchmark.log | tail -1 | awk '{print $NF}')
    output_throughput=$(grep 'Output token throughput (tok/s):' ${CONFIG_LOG_DIR}/${LOG_PREFIX}_benchmark.log | tail -1 | awk '{print $NF}')

    # Build summary line
    row="${NUM_NODES},${PP_SIZE},${TP_SIZE},${COMM_BACKEND},${KV_CACHE_DTYPE},\"${PARTITION}\",${MAX_MODEL_LEN},${INPUT_TOKENS},${OUTPUT_TOKENS},${NUM_PROMPTS},${MAX_CONCURRENCY},${mean_ttft},${mean_tpot},${total_throughput},${output_throughput}"

    # Append eval metrics if requested
    if [[ -n "$EVAL_TASKS" && "$EVAL_TASKS" != "none" ]]; then
      IFS=',' read -ra TASKS <<< "$EVAL_TASKS"
      for task in "${TASKS[@]}"; do
        test_benchmark_client_accuracy ${MAX_CONCURRENCY} ${HOST} ${PORT} ${MODEL_PATH} "$EVAL_TASKS" ${CONFIG_LOG_DIR} \
        | tee -a ${CONFIG_LOG_DIR}/${LOG_PREFIX}_accuracy.log
        # Extract metric from last eval JSON
        f=$(ls ${CONFIG_LOG_DIR}/lm_eval_output_${task%%:*}_*.json | tail -1)
        metric="N/A"
        if [ -f "$f" ]; then
          metric=$(jq '.results["'"${task%%:*}"'"].metrics | to_entries[0].value' "$f" 2>/dev/null || echo "N/A")
        fi
        row+=",${metric}"
      done
    fi

    echo "$row" | tee -a "$SUMMARY_LOG"
  fi

  # Teardown server
  kill -9 $SERVER_PID 2>/dev/null
  sleep 20

done