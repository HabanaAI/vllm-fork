#!/bin/bash

script_dir="$(dirname "$(realpath "$0")")"
vllm_dir="$(dirname "$script_dir")"

tp_parrallel=8
in_len=1024
out_len=1024
multi_step=1
total_len=$((in_len + out_len))
# if total_len is not multiple of 128, round up to the next multiple of 128
if [ $((total_len % 128)) -ne 0 ]; then
    echo 'round up for 128'
    total_len=$(((total_len / 128 +  1) * 128 ))
fi
ep_size=16
moe_n_slice=1
gpu_utils=0.95
bs=512
num_prompts=1536
request_rate=inf
# block_size=256 #not valid  api_server.py: error: argument --block-size: invalid choice: 256 (choose from 1, 8, 16, 32, 64, 128)
block_size=128
block_step=128

log_name="online-gaudi3-${gpu_utils}util-TPparallel${tp_parrallel}-EP${ep_size}-loop${moe_n_slice}moegroups-multistep${multi_step}_nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}_mdllen${total_len}"

prompt_bs_max=8
# Increase prompt max seq len bucket just in case there are more tokens than provided due to tokenizer
possible_extra_tokens=256
prompt_seq_min=$((in_len - possible_extra_tokens))
if [ $prompt_seq_min -lt 128 ]; then
    prompt_seq_min=128
fi
prompt_seq_max=$((in_len + possible_extra_tokens))
total_len_aligned=$(((total_len + block_size - 1) / block_size * block_size + possible_extra_tokens))
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len * bs / block_size - block_step))
if [ $VLLM_DECODE_BLOCK_BUCKET_MIN -lt 128 ]; then
    VLLM_DECODE_BLOCK_BUCKET_MIN=128
fi
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len_aligned * bs / block_size + block_step))

model="/mnt/weka/llm/pixtral/Pixtral-Large-Instruct-2411"
tokenizer_mode="mistral"
model_name="Pixtral-Large-Instruct-2411"

########################################################## Concurrency 64 Sonnet #################################################################
max_concurrency_client=${bs}
echo "Startig benchmark..."
pushd "${vllm_dir}"
start_time=$(date +%s)
python3 ./benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --endpoint /chat/completions \
     --base-url http://127.0.0.1:18080/v1 \
    --model ${model} \
    --served_model_name "pixtrallarge" \
    --tokenizer-mode ${tokenizer_mode} \
    --request-rate ${request_rate} \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
    --num-prompts ${num_prompts} \
    --port 18080 \
    --dataset-path lmms-lab/LLaVA-OneVision-Data \
    --dataset-name hf \
    --hf-subset "chart2text(cauldron)" \
    --hf-split train \
    --num_prompts=10 \
    --max-concurrency ${max_concurrency_client} \
    --save-result 2>&1 | tee "${logs_dir}/g3-${model_name}-in${in_len}-out${out_len}-req${request_rate}-num_prompts${num_prompts}-concurrency${max_concurrency_client}.log"
end_time=$(date +%s)
popd
echo "Time elapsed: $((end_time - start_time))s"
sleep 10
