#! /bin/bash
set -ex

#model_path=/root/.cache/huggingface/Mixtral-8x7B-v0.1
#model_path=/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2
model_path=/root/.cache/huggingface/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2
#model_path=/root/.cache/huggingface/DeepSeek-R1-G2
#model_path=/root/.cache/huggingface/Llama-3.1-70B
cache_path=$model_path/.hpu_cache

# PP execution fails is these are not set. This need to be resolved.
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0

export VLLM_DEVICE_PROFILER_ENABLED=false
export VLLM_DEVICE_PROFILER_WARMUP_STEPS=15
export VLLM_DEVICE_PROFILER_STEPS=3
export VLLM_DEVICE_PROFILER_REPEAT=1
# export HABANA_PROFILE='profile_api_with_nics'

# Comment this out if running with PP=1
export VLLM_PP_LAYER_PARTITION="32,29"
# This may or may not be needed.
#export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=0

# -r: Input size range
# -d: KVCache data type
# -i: Input Size
# -o: Output Size
# -t: Max Num Batched Tokens
# -l: Max Model Len
# -b: Batch Size, Max Concurrency = Batch Size * PP Size
# -p: Num Samples
# -n: TP/EP Size
# -g: PP size

bash scripts/benchmark_throughput.sh \
    -w $model_path \
    -s \
    -f \
    -r 0.8 \
    -i 1000 \
    -o 1000 \
    -t 4096 \
    -l 4096 \
    -b 32 \
    -p 64 \
    -n 4 \
    -g 2 \
    -c $cache_path