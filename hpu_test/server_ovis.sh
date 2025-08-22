export VLLM_SKIP_WARMUP=True 
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=8
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=4
export VLLM_DECODE_BS_BUCKET_MAX=64 ## used to be 52
export VLLM_PROMPT_SEQ_BUCKET_MIN=3072
export VLLM_PROMPT_SEQ_BUCKET_STEP=256
export VLLM_PROMPT_SEQ_BUCKET_MAX=3840 ## used to be 3328 
export VLLM_DECODE_BLOCK_BUCKET_MIN=1024
export VLLM_DECODE_BLOCK_BUCKET_STEP=512
export VLLM_DECODE_BLOCK_BUCKET_MAX=2048

export VLLM_MULTIMODAL_BUCKETS="6912"

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
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=False

python3 -m vllm.entrypoints.openai.api_server --model "AIDC-AI/Ovis2.5-9B" --task generate --trust-remote-code --host 0.0.0.0 --port 10900 --tensor-parallel-size 1 \
    --mm-processor-kwargs '{"min_pixels": 401408, "max_pixels": 1605632}' --max-parallel-loading-workers 16 --max-model-len 8000 \
    --gpu-memory-util 0.99 \
    --max-num-prefill-seqs 1
