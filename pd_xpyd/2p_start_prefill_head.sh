#set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")


export HCCL_OVER_OFI=1
export HCCL_GAUDI_DIRECT=1
export HCCL_SOCKET_IFNAME=enp24s0f0np0
export LD_LIBRARY_PATH=/opt/libfabric/lib:$LD_LIBRARY_PATH

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=10.112.242.154,localhost,127.0.0.1
export no_proxy=10.112.242.154,localhost,127.0.0.1

BENCHMARK_MODE=0

export VLLM_TORCH_PROFILER_DIR=./profiles
export VLLM_PROFILER_ENABLED=full
export VLLM_PROFILE_CONFIG_PATH=profile_config.json
export HABANA_PROFILE_WRITE_HLTV=1
export HABANA_PROFILE=profile_api_with_nics

#if [ -z "$1" ] || [ "$1" == "g10" ]; then
#    source "$BASH_DIR"/start_etcd_mooncake_master.sh
#    echo "source "$BASH_DIR"/start_etcd_mooncake_master.sh"
#fi

if [ "$2" == "benchmark" ]; then
    BENCHMARK_MODE=1
    sed -i 's/export VLLM_USE_ASYNC_TRANSFER_IN_PD=.*/export VLLM_USE_ASYNC_TRANSFER_IN_PD=0/' $BASH_DIR/pd_env.sh
    echo " Benchmark mode enabled"
else
    sed -i 's/export VLLM_USE_ASYNC_TRANSFER_IN_PD=.*/export VLLM_USE_ASYNC_TRANSFER_IN_PD=1/' $BASH_DIR/pd_env.sh
    echo " Normal mode enabled"
fi

#if [ -z "$1" ] || [ "$1" == "G3D-sys01" ] || [ "$1" == "pcie4" ]; then
if [ True ]; then
    if [ "$BENCHMARK_MODE" == "1" ]; then
	source "$BASH_DIR"/start_etcd_mooncake_master2.sh benchmark
 	echo "source "$BASH_DIR"/start_etcd_mooncake_master2.sh benchmark"
    else
	source "$BASH_DIR"/start_etcd_mooncake_master2.sh
	echo "source "$BASH_DIR"/start_etcd_mooncake_master2.sh"
    fi
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MOONCAKE_CONFIG_PATH="$BASH_DIR"/mooncake_${1:-g10}.json

source "$BASH_DIR"/dp_2p_env.sh

ray start --head --port=6886


while true; do
    read -p "Continue? (y): " answer
    if [[ "$answer" == [yY] ]]; then
        echo "Proceeding..."
        break
    else
        echo "Invalid input. Please enter y or n."
    fi
done

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="xpyd_logs"
mkdir -p "$log_dir"
log_file="$log_dir/prefill_2p_`hostname`_${timestamp}.log"

if [ "$INC_FP8" -eq 1 ]; then
  kv_cache_dtype_arg="--kv-cache-dtype fp8_inc"
  echo "<prefill>it's inc fp8 kv cache mode"
else
  kv_cache_dtype_arg=""
  echo "<prefill>it's bf16 kv cache mode"
fi
#unset VLLM_SKIP_WARMUP
#export PT_HPU_RECIPE_CACHE_CONFIG=./_prefill_cache,false,16384

python3 -m vllm.entrypoints.openai.api_server --model $model_path --port 8100 --max-model-len $model_len --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION -tp 16  --max-num-seqs $max_num_seqs --trust-remote-code --disable-async-output-proc $kv_cache_dtype_arg --disable-log-requests --max-num-batched-tokens $max_num_batched_tokens --use-padding-aware-scheduling --use-v2-block-manager --distributed_executor_backend ray --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}' 2>&1 | tee "$log_file"
