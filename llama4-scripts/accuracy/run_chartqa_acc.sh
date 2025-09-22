# model_path="/mnt/weka/llm/Llama-4-Scout-17B-16E-Instruct/"
# output_dir="Llama-4-Scout-17B-16E-Instruct-chartqa-pro-acc"
model_path="/mnt/weka/llm/Llama-4-Maverick-17B-128E-Instruct/"
output_dir="Llama-4-Maverick-17B-128E-Instruct-chartqa-acc"
mkdir -p ${output_dir}
PT_HPU_LAZY_MODE=1 \
VLLM_SKIP_WARMUP=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
lm_eval --model vllm-vlm \
  --model_args "pretrained=${model_path},enforce_eager=True,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=64,gpu_memory_utilization=0.9,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=False,enable_expert_parallel=True" \
  --tasks chartqa --num_fewshot 0 --fewshot_as_multiturn --apply_chat_template --batch_size 'auto' --log_samples --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log.txt