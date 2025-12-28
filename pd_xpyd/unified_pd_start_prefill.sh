#set -x
set -euo pipefail

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")


if [ "${PREFILL_NEED_SCALEOUT:-0}" == "1" ]; then
    export HCCL_OVER_OFI=1
    export HCCL_GAUDI_DIRECT=1
    export HCCL_SOCKET_IFNAME=enp24s0f0np0
    export LD_LIBRARY_PATH=/opt/libfabric/lib:${LD_LIBRARY_PATH:-}
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/lib

if [ "$ROLE" = "head" ]; then
    if [ "$BENCHMARK_MODE" = "benchmark" ]; then
        sed -i 's/export VLLM_USE_ASYNC_TRANSFER_IN_PD=.*/export VLLM_USE_ASYNC_TRANSFER_IN_PD=0/' "$BASH_DIR/pd_env.sh"
        echo " Benchmark mode enabled"
    else
        sed -i 's/export VLLM_USE_ASYNC_TRANSFER_IN_PD=.*/export VLLM_USE_ASYNC_TRANSFER_IN_PD=1/' "$BASH_DIR/pd_env.sh"
        echo " Normal mode enabled"
    fi
    
    #export VLLM_TORCH_PROFILER_DIR=./profiles
    #export VLLM_PROFILER_ENABLED=full
    #export VLLM_PROFILE_CONFIG_PATH=profile_config.json
    #export HABANA_PROFILE_WRITE_HLTV=1
    #export HABANA_PROFILE=profile_api_with_nics
    
    source "$BASH_DIR/start_etcd_mooncake_master.sh" ${BENCHMARK_MODE:+benchmark}
    
fi

export MOONCAKE_CONFIG_PATH="$BASH_DIR/mooncake_`hostname`.json"
echo "MOONCAKE_CONFIG_PATH:"$MOONCAKE_CONFIG_PATH

# Function to get local IP address filtered by MOONCAKE_LOCAL_ADDR_PREFIX
get_local_ip() {
    local addr_prefix="${MOONCAKE_LOCAL_ADDR_PREFIX:-}"
    local local_ip=""
    if [ -z "$addr_prefix" ]; then
        echo "Error: MOONCAKE_LOCAL_ADDR_PREFIX is not set" >&2
        return 1
    fi
    # Get all IPs from hostname -I and filter by prefix, print first match
    local_ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 "^$addr_prefix")
    if [ -z "$local_ip" ]; then
        # If no match, fallback to empty string
        local_ip=""
    fi
    echo "$local_ip"
}

# Get local IP and server addresses
LOCAL_HOSTNAME_IP=$(get_local_ip)
ETCD_META_SERVER_VALUE=${ETCD_META_SERVER}
MOONCAKE_SERVER_VALUE=${MOONCAKE_SERVER}

# Ensure metadata_server has etcd:// prefix if not present
if [[ ! "$ETCD_META_SERVER_VALUE" =~ ^etcd:// ]]; then
    ETCD_META_SERVER_VALUE="etcd://${ETCD_META_SERVER_VALUE}"
fi

# Determine device_name array size (default to 16, or use CARDS_PER_NODE if available)
DEVICE_COUNT=${CARDS_PER_NODE}

# Load hostname to device mapping from external file
hostname=$(hostname)
HOST_DEVICE_MAP_FILE="${HOST_DEVICE_MAP_FILE:-$BASH_DIR/host_cx7_map.sh}"

if [ ! -f "$HOST_DEVICE_MAP_FILE" ]; then
    echo "ERROR: Host device map file not found: $HOST_DEVICE_MAP_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$HOST_DEVICE_MAP_FILE"

# Look up device array for current hostname
if [[ -z "${HOST_DEVICE_MAP[$hostname]:-}" ]]; then
    echo "ERROR: Unknown hostname $hostname (not found in $HOST_DEVICE_MAP_FILE)" >&2
    exit 1
fi

# Convert space-separated string to array
read -r -a DEVICE_NAME_ARRAY <<< "${HOST_DEVICE_MAP[$hostname]}"

# Check that array size matches DEVICE_COUNT
if [ "${#DEVICE_NAME_ARRAY[@]}" -ne "$DEVICE_COUNT" ]; then
    echo "Error: DEVICE_NAME_ARRAY size (${#DEVICE_NAME_ARRAY_ARR[@]}) does not match DEVICE_COUNT ($DEVICE_COUNT)" >&2
    exit 1
fi

# Join the array into a JSON array with double quotes
DEVICE_NAME_ARRAY_JSON=$(printf '"%s",' "${DEVICE_NAME_ARRAY[@]}")
DEVICE_NAME_ARRAY_JSON="[${DEVICE_NAME_ARRAY_JSON%,}]"

# Create/overwrite mooncake JSON file
cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
    "local_hostname": "$LOCAL_HOSTNAME_IP",
    "metadata_server": "$ETCD_META_SERVER_VALUE",
    "protocol": "rdma",
    "device_name": $DEVICE_NAME_ARRAY_JSON,
    "master_server_address": "$MOONCAKE_SERVER_VALUE"
}
EOF

echo "Generated mooncake config at $MOONCAKE_CONFIG_PATH:"
cat "$MOONCAKE_CONFIG_PATH"


export PREFILL_NUM_CARDS=${PREFILL_NUM_CARDS:-8}
source "$BASH_DIR/dp_p_env.sh"
echo "PREFILL_USE_RAY:"$PREFILL_USE_RAY
distributed_executor_backend="mp"
if [[ $PREFILL_USE_RAY -eq 1 ]]; then
    distributed_executor_backend="ray"
    ray stop --force || true
    sleep 3s
fi

if [ "$ROLE" = "head" ]; then
    if [[ $PREFILL_USE_RAY -eq 1 ]]; then
        ray start --head --port=${RAY_HEAD_PORT:-6886} --disable-usage-stats
    fi

    #while true; do
    #    read -p "Continue? (y): " answer
    #    [[ "$answer" =~ ^[yY]$ ]] && break
    #done

    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_dir="xpyd_logs"
    mkdir -p "$log_dir"
    # Include instance index in log file name if available
    if [[ -n "${P_INSTANCE_IDX:-}" ]]; then
        log_file="$log_dir/prefill_${P_INSTANCE_IDX}_${ROLE_IDX}_$(hostname)_${timestamp}.log"
    else
        log_file="$log_dir/prefill_2p_$(hostname)_${timestamp}.log"
    fi

    if [ "${INC_FP8:-0}" -eq 1 ]; then
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
      -tp $PREFILL_TP_SIZE \
      --max-num-seqs "$max_num_seqs" \
      --trust-remote-code \
      --disable-async-output-proc \
      $kv_cache_dtype_arg \
      --disable-log-requests \
      --max-num-batched-tokens "$max_num_batched_tokens" \
      --use-padding-aware-scheduling \
      --use-v2-block-manager \
      --distributed_executor_backend $distributed_executor_backend \
      --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}' \
      2>&1 | tee "$log_file"
else
    # just join head node via ray
    echo "Joining head node via ray: $HEAD_ADDR:${RAY_HEAD_PORT:-6886}"
    ray start --address="$HEAD_ADDR":${RAY_HEAD_PORT:-6886}
fi
