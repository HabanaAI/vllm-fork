#Llama-3.3-70B-Instruct for interactive with batch size 8 with SLA TTFT <2s 

RUNTIME_SCALE_PATCHING=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_HPU_USE_DELAYED_SAMPLING=false \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_LAZY_MODE=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
DEFAULT_INCLUDE_STOP_SEQS=false \
VLLM_FUSED_BLOCK_SOFTMAX=true \
VLLM_DISABLE_MARK_SCALES_AS_CONST=true \
VLLM_HANDLE_TOPK_DUPLICATES=true \
QUANT_CONFIG="/root/llama-3.3-70b-instruct/maxabs_quant_g3.json" \
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --quantization inc \
    --kv-cache-dtype fp8_inc \
    --weights-load-device cpu \
    --tensor_parallel_size 2 \
    --disable-log-requests \
    --use-v2-block-manager \
    --use-padding-aware-scheduling \
    --dtype bfloat16 \
    --block-size 256 \
    --max-num-prefill-seqs 1 \
    --max-num-seqs 8 \
    --gpu_memory_utilization 0.8 \
    --max-model-len 131072 \
    --port 8080
