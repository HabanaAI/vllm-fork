# !/bin/bash

if [ $# -gt 0 ] && [ "$1" == "--model_path" ]; then
    model_path=$2
else
    model_path="/mnt/weka/llm/qwen3_pre_release/Qwen3-32B-250426/"
fi

if [ $# -gt 4 ] && [ "$3" == "--tp_size" ]; then
    tp_size=$4
else
    tp_size=1
fi

if [ $# -eq 6 ] && [ "$5" == "--ep_size" ]; then
    ep_size=$6
else
    ep_size=1
fi

echo ${ep_size}
if [ "${ep_size}" -gt 1 ]; then
    enable_expert_parallel="True"
else
    enable_expert_parallel="False"
fi


export VLLM_LOGGING_LEVEL=DEBUG

model_name=$(basename ${model_path})
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="${model_name}-tp${tp_size}-gsm8k-acc-fp8-${timestamp}"
#limit=None

echo "Eval model ${model_name} with config ${QUANT_CONFIG}"

mkdir -p ${output_dir}


VLLM_MOE_N_SLICE=1 \
VLLM_EP_SIZE=${ep_size} \
PT_HPU_LAZY_MODE=1 \
VLLM_SKIP_WARMUP=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
lm_eval --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,quantization=inc,enable_expert_parallel=${enable_expert_parallel}" \
  --tasks gsm8k --batch_size 128 --log_samples --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log.txt
