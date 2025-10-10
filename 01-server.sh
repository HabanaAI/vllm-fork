# export HABANA_LOGS=./.habana_logs
# export LOG_LEVEL_ALL=2
export VLLM_SKIP_WARMUP=False
# hl-prof-config --use-template profile_api --hw-trace off
# export HABANA_PROFILE=1
# export VLLM_PROFILER_ENABLED=full
# export VLLM_TORCH_PROFILER_DIR=./profiler-dir-ovis-precompute-warmup
export HOST=localhost
export PORT=8088
export MODEL="ibm-granite/granite-4.0-tiny-preview"
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_DECODE_BS_BUCKET_MIN=4
export VLLM_DECODE_BS_BUCKET_STEP=4
export VLLM_DECODE_BS_BUCKET_MAX=52 ## used to be 52
export VLLM_PROMPT_SEQ_BUCKET_MIN=4096
export VLLM_PROMPT_SEQ_BUCKET_STEP=256
export VLLM_PROMPT_SEQ_BUCKET_MAX=4352 ## used to be 3328
export VLLM_DECODE_BLOCK_BUCKET_MIN=1024
export VLLM_DECODE_BLOCK_BUCKET_STEP=512
export VLLM_DECODE_BLOCK_BUCKET_MAX=2048

export VLLM_MULTIMODAL_BUCKETS="12544"

export VLLM_DELAYED_SAMPLING=true
export TRANSFORMERS_VERBOSITY=info
#export VLLM_FP32_SOFTMAX=true
export VLLM_EXPONENTIAL_BUCKETING=false
export VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False
export PT_HPU_LAZY_MODE=1

## newly added
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true

# Ovis - need to run with this off for vision part
# export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=False

export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_GRAPH_PROMPT_RATIO=0.1

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 8000 \
    --gpu-memory-util 0.99 \
    --max-num-prefill-seqs 1  \
    --disable-log-requests \
    --disable-log-stats 2>&1 | tee granite-server.txt