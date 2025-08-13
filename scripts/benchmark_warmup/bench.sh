# usage: 
#  fast_warm_up: bash bench.sh  --fast
#  fast_warm_up: bash bench.sh  --no-fast

if [ "$1" == "--fast" ]; then
    enable_fast=true
    echo "*************enable_fast_warm_up*************"
elif [ "$1" == "--no-fast" ]; then
    enable_fast=false
    echo "*************disable_fast_warm_up*************"
else
    echo "Usage: $0 [--fast|--no-fast]"
    exit 1
fi

model_path="/mnt/ctrl/disk3/HF_models/llama-3-8b"
input_len=1024
output_len=512
batch_size=64
num_prompts=10
num_models=4

if [ "$enable_fast" = true ]; then
    cache_path="/mnt/disk2/mingzhil/current_cache"
    log_file="result_fast_warm_up.log"
    export VLLM_FAST_WARMUP=True
else
    cache_path="/mnt/disk2/mingzhil/current_cache_2"
    log_file="result_disable_fast_warm_up.log"
    unset VLLM_FAST_WARMUP
fi

# clean up cache
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "cache_path: $cache_path"
rm -rf "${cache_path}/*"
echo "after rm:"
ls "${cache_path}"
export PT_HPU_RECIPE_CACHE_CONFIG="${cache_path},false,8192"

# run
bash benchmark_throughput.sh \
    -w "$model_path" \
    -i "$input_len" \
    -o "$output_len" \
    -b "$batch_size" \
    -p "$num_prompts" \
    -n "$num_models" |& tee "$log_file"