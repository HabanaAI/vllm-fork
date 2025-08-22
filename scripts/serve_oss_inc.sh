model_path=/mnt/disk5/lmsys/gpt-oss-20b-bf16
model_path=/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-20b-bf16

export QUANT_CONFIG=./scripts/quant_configs/inc_unit_scale.json
tp_size=1
VLLM_SKIP_WARMUP=true \
VLLM_ENABLE_FUSED_MOE_WITH_BIAS=1 \
VLLM_PROMPT_USE_FUSEDSDPA=0 \
    PT_HPU_LAZY_MODE=1 \
        vllm serve $model_path \
        --tensor-parallel-size $tp_size \
        --dtype bfloat16 \
        --max-model-len  2048 \
        --disable-log-requests \
        --max_num_seqs 128 \
        --quantization inc \
        --kv_cache_dtype fp8_inc