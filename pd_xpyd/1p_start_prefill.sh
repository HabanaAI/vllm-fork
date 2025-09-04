#!/bin/bash

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

BENCHMARK_MODE=0

if [ "$3" == "benchmark" ]; then
    BENCHMARK_MODE=1
    sed -i 's/export VLLM_USE_ASYNC_TRANSFER_IN_PD=.*/export VLLM_USE_ASYNC_TRANSFER_IN_PD=0/' $BASH_DIR/pd_env.sh
    echo " Benchmark mode enabled"
else
    sed -i 's/export VLLM_USE_ASYNC_TRANSFER_IN_PD=.*/export VLLM_USE_ASYNC_TRANSFER_IN_PD=1/' $BASH_DIR/pd_env.sh
    echo " Normal mode enabled"
fi

if [ "$2" == "master" ]; then
    if [ "$BENCHMARK_MODE" == "1" ]; then
        source "$BASH_DIR"/start_etcd_mooncake_master.sh benchmark
        echo "source "$BASH_DIR"/start_etcd_mooncake_master.sh benchmark"
    else
        source "$BASH_DIR"/start_etcd_mooncake_master.sh
        echo "source "$BASH_DIR"/start_etcd_mooncake_master.sh"
    fi
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MOONCAKE_CONFIG_PATH="$BASH_DIR"/mooncake_${1:-g10}.json

echo "Using Mooncake config: $MOONCAKE_CONFIG_PATH"

source "$BASH_DIR"/dp_p_env.sh

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="xpyd_logs"
mkdir -p "$log_dir"
log_file="$log_dir/prefill_${timestamp}.log"

if [ "$INC_FP8" -eq 1 ]; then
  kv_cache_dtype_arg="--kv-cache-dtype fp8_inc"
  echo "<prefill>it's inc fp8 kv cache mode"
else
  kv_cache_dtype_arg=""
  echo "<prefill>it's bf16 kv cache mode"
fi

python3 -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --port 8100 \
  --max-model-len "$model_len" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  -tp 8 \
  --max-num-seqs "$max_num_seqs" \
  --trust-remote-code \
  --disable-async-output-proc \
  --disable-log-requests \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --use-padding-aware-scheduling \
  --use-v2-block-manager \
  --distributed_executor_backend mp \
  $kv_cache_dtype_arg \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}' 2>&1 | tee "$log_file"
