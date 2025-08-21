model_path=/mnt/disk5/lmsys/gpt-oss-20b-bf16

VLLM_SKIP_WARMUP=true \
VLLM_PROMPT_USE_FUSEDSDPA=0 \
    PT_HPU_LAZY_MODE=1 \
        vllm serve $model_path \
        --tensor-parallel-size 8 \
        --dtype bfloat16 \
        --max-model-len  16384 \
        --disable-log-requests \
        --max_num_seqs 128