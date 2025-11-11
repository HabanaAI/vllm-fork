#!/bin/bash
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

# for backward compatible. following nodes are started as mooncake master node
if [ "$2" == "master" ] || [ -z "$1" ] || [ "$1" == "g10" ] || [ "$1" == "pcie4" ]; then
    source "$BASH_DIR"/start_etcd_mooncake_master.sh
    echo "source "$BASH_DIR"/start_etcd_mooncake_master.sh"
fi

sleep 20

export MOONCAKE_CONFIG_PATH="$BASH_DIR"/mooncake_${1:-g10}.json

echo "Using Mooncake config: $MOONCAKE_CONFIG_PATH"

source "$BASH_DIR"/dp_p_env.sh


if [ "$INC_FP8" -eq 1 ]; then
  kv_cache_dtype_arg="--kv-cache-dtype fp8_inc"
  echo "<prefill>it's inc fp8 kv cache mode"
else
  kv_cache_dtype_arg=""
  echo "<prefill>it's bf16 kv cache mode"
fi

# Define the Python command as an array
CMD=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$model_path"
    --port 8100
    --max-model-len "$model_len"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    -tp 8
    --max-num-seqs "$max_num_seqs"
    --trust-remote-code
    --disable-async-output-proc
    --disable-log-requests
    --max-num-batched-tokens "$max_num_batched_tokens"
    --use-padding-aware-scheduling
    --use-v2-block-manager
    --distributed_executor_backend mp
    $kv_cache_dtype_arg
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
)

# Check if XPYD_LOG is set
if [ -n "$XPYD_LOG" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="$XPYD_LOG/Prefill_${timestamp}.log"
    echo "Logging to $log_file..."

    # Execute command and log stdout+stderr using tee
    "${CMD[@]}" 2>&1 | tee "$log_file"
else
    echo "XPYD_LOG not set, running without logging..."
    # Execute command without logging
    "${CMD[@]}"
fi
