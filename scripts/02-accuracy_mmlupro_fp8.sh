# !/bin/bash

if [ $# -gt 0 ] && [ "$1" == "--model_path" ]; then
    model_path=$2
else
    model_path="/mnt/weka/llm/qwen3_pre_release/Qwen3-32B-250426/"
fi

if [ $# -eq 4 ] && [ "$3" == "--tp_size" ]; then
    tp_size=$4
else
    tp_size=1
fi

model_name=$(basename ${model_path})
output_dir="${model_name}-tp${tp_size}-mmlu-pro-acc"
#limit=None
if [ ${model_name} == "Qwen3-30B-A3B-250425" ]; then
    quant_file_path="inc_quant_g3_30B_A3B.json"
elif [ ${model_name} == "Qwen3-32B-250426" ]; then
    quant_file_path="inc_quant_g3_32B.json"
elif [ ${model_name} == "Qwen3-235B-A22B-250426" ]; then
    quant_file_path="inc_quant_g3_235B_A22B.json"
else
    echo "Unknown model name: ${model_name}"
    exit 1
fi

echo "Eval model ${model_name} with config ${quant_file_path}"

mkdir -p ${output_dir}

QUANT_CONFIG=${quant_file_path} \
PT_HPU_LAZY_MODE=1 \
VLLM_SKIP_WARMUP=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
lm_eval --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,quantization=inc" \
  --tasks mmlu_pro --apply_chat_template --batch_size 128 --log_samples --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log.txt