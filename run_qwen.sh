VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_STEP=1 \
VLLM_PROMPT_BS_BUCKET_MAX=1 \
VLLM_DECODE_BS_BUCKET_MIN=1 \
VLLM_DECODE_BS_BUCKET_STEP=1 \
VLLM_DECODE_BS_BUCKET_MAX=1 \
VLLM_PROMPT_SEQ_BUCKET_MIN=256 \
VLLM_PROMPT_SEQ_BUCKET_STEP=256 \
VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
VLLM_DECODE_BLOCK_BUCKET_MIN=256 \
VLLM_DECODE_BLOCK_BUCKET_STEP=256 \
VLLM_DECODE_BLOCK_BUCKET_MAX=1024 \
RUNTIME_SCALE_PATCHING=1 \
PT_HPU_LAZY_MODE=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_EXPONENTIAL_BUCKETING=false \
QUANT_CONFIG=/software/data/vllm-benchmarks/inc/qwen3-30b-a3b/maxabs_quant_g3.json \
python -m vllm.entrypoints.openai.api_server --port 8080 \
    --model Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 1 \
    --max-num-seqs 1 \
    --dtype bfloat16 \
    --block-size 128 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --quantization inc \
    --kv-cache-dtype fp8_inc \
    --weights-load-device cpu
 
