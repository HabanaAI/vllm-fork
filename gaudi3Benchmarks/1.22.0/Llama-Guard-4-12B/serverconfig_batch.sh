# Llama-Guard-4-12B for batch mode size 8, gaurd model ensure the latency to be range
#
VLLM_FUSED_BLOCK_SOFTMAX="true" \
VLLM_DELAYED_SAMPLING="false" \
PT_HPU_LAZY_MODE="1" \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_ALLOW_LONG_MAX_MODEL_LEN="1" \
VLLM_EXPONENTIAL_BUCKETING="true" \
PT_HPU_WEIGHT_SHARING=0 \
vllm serve  meta-llama/Llama-Guard-4-12B \
    --port 8080 \
    --tensor-parallel-size 1 \
    --max-num-seqs 8 \
    --disable-log-requests \
    --dtype bfloat16 \
    --num_scheduler_steps 1 \
    --max-model-len 131072 \
    --use-padding-aware-scheduling \
    --block-size 128 \
    --gpu_memory_utilization 0.9 \
    --trust_remote_code 2>&1 | tee Llama-Guard-4-12B_serverlog.txt
