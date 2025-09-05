model_path=/mnt/disk5/lmsys/gpt-oss-20b-bf16
model_path=/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-20b-bf16
# export QUANT_CONFIG=./quant_configs/inc_quant.json
export QUANT_CONFIG=./quant_configs/inc_unit_scale.json
# export QUANT_CONFIG=./quant_configs/inc_quant.json

export INC_PT_ONLY=1

tp_size=2
VLLM_BUILD=1.23.0.248 \
VLLM_SKIP_WARMUP=true \
VLLM_DISABLE_MARK_SCALES_AS_CONST=1 \
VLLM_ENABLE_FUSED_MOE_WITH_BIAS=1 \
VLLM_PROMPT_USE_FUSEDSDPA=0 \
    PT_HPU_LAZY_MODE=1 \
        vllm serve $model_path \
        --tensor-parallel-size $tp_size \
        --dtype bfloat16 \
        --port 8688 \
        --max-model-len  8192 \
        --disable-log-requests \
        --max_num_seqs 128 \
        --quantization inc \
        --kv_cache_dtype fp8_inc 
        # --enable-expert-parallel 