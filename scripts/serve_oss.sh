model_path=/mnt/disk5/lmsys/gpt-oss-20b-bf16
model_path=/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-20b-bf16

timestamp=$(date +%Y%m%d-%H%M%S)
log_file=server.$timestamp.log
tp_size=1
VLLM_ENABLE_FUSED_MOE_WITH_BIAS=1 \
VLLM_SKIP_WARMUP=true \
VLLM_PROMPT_USE_FUSEDSDPA=0 \
    PT_HPU_LAZY_MODE=1 \
        vllm serve $model_path \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len  2048 \
        --disable-log-requests \
        --max_num_seqs 128 2>&1 | tee $log_file