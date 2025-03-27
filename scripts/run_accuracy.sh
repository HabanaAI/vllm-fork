# !/bin/bash

MODEL=/data/models/DeepSeek-R1-static/
TP_SIZE=8
LIMIT=64
FEWSHOT=5
BATCH_SIZE=64

VLLM_DELAYED_SAMPLING=true \
USE_FP8_MATMUL=true \
VLLM_SKIP_WARMUP=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_MOE_N_SLICE=1 \
VLLM_EP_SIZE=8 \
VLLM_MLA_DISABLE_REQUANTIZATION=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
lm_eval --model vllm \
  --model_args "pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend=mp,trust_remote_code=true,max_model_len=16384,dtype=bfloat16,gpu_memory_utilization=0.9,kv_cache_dtype=fp8_inc" \
  --tasks gsm8k --num_fewshot "$FEWSHOT" --limit "$LIMIT" \
  --batch_size ${BATCH_SIZE} \
  --log_samples --output_path lm_eval_tp${TP_SIZE}_l${LIMIT}_f${FEWSHOT}_b${BATCH_SIZE}.json \