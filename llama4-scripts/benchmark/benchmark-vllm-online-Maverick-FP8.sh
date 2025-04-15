#!/bin/bash
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
bs=64
num_prompts=640
request_rate=inf
block_size=256
block_step=256

log_name="[inc-staticquant-scalar-fp8matmul-split]online-gaudi3-${gpu_utils}util-TPparallel${tp_parrallel}-EP${ep_size}-loop${moe_n_slice}moegroups-multistep${multi_step}_nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}_mdllen${total_len}"

prompt_bs_max=8
# Increase prompt max seq len bucket just in case there are more tokens than provided due to tokenizer
prompt_seq_max=$((in_len + $block_size))
total_len_aligned=$(((total_len + block_size - 1) / block_size * block_size))
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len * bs / block_size))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len_aligned * bs / block_size + block_step))

model="/mnt/weka/llm/Llama-4-Maverick-17B-128E-Instruct/"
tokenizer="/mnt/weka/llm/Llama-4-Maverick-17B-128E-Instruct/"
model_name="Maverick"

mkdir -p benchmark_logs
# QUANT_CONFIG="scripts/inc_quant_with_fp8kv_config.json" \
# VLLM_REQUANT_FP8_INC=1 \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# VLLM_USE_FP8_MATMUL=true \
# VLLM_MOE_N_SLICE=${moe_n_slice} \
# VLLM_MLA_DISABLE_REQUANTIZATION=1 \
QUANT_CONFIG="../quant.json" \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=$prompt_bs_max \
VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len} \
VLLM_PROMPT_SEQ_BUCKET_MAX=${prompt_seq_max} \
VLLM_DECODE_BS_BUCKET_MIN=${bs} \
VLLM_DECODE_BS_BUCKET_MAX=${bs} \
VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
VLLM_DECODE_BLOCK_BUCKET_STEP=$block_step \
VLLM_DELAYED_SAMPLING=true \
HABANA_VISIBLE_DEVICES="ALL" \
VLLM_EP_SIZE=${ep_size} \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
python3 -m vllm.entrypoints.openai.api_server \
    --port 18080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --num_scheduler_steps ${multi_step} \
    --max-model-len 9216 \
    --max-num-batched-tokens 9216 \
    --use-padding-aware-scheduling \
    --block-size ${block_size} \
    --distributed_executor_backend ray \
    --gpu_memory_utilization ${gpu_utils} \
    --enable-expert-parallel \
    --quantization="inc" \
    --override-generation-config='{"attn_temperature_tuning": true}' \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
pid=$(($!-1))
# QUANT_CONFIG="../inc_unit_scale_quant.json" \
#     --kv_cache_dtype "fp8_inc" \
    # --quantization="inc" \
#VLLM_PROMPT_BS_BUCKET_MIN=1 \
#VLLM_PROMPT_BS_BUCKET_MAX=8 \
#VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
#VLLM_PROMPT_SEQ_BUCKET_MAX=1152 \
#VLLM_DECODE_BS_BUCKET_MIN=64 \
#VLLM_DECODE_BS_BUCKET_MAX=${bs} \
#VLLM_DECODE_BLOCK_BUCKET_MAX=1280 \
#VLLM_DECODE_BLOCK_BUCKET_STEP=256 \
#VLLM_DELAYED_SAMPLING=true \
#HABANA_VISIBLE_DEVICES="ALL" \
#VLLM_EP_SIZE=${ep_size} \
#PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
#PT_HPU_WEIGHT_SHARING=0 \
#python3 -m vllm.entrypoints.openai.api_server \
#    --port 18080 \
#    --model ${model} \
#    --tensor-parallel-size ${tp_parrallel} \
#    --max-num-seqs ${bs} \
#    --seed 2024  \
#    --disable-log-requests \
#    --dtype bfloat16 \
#    --use-v2-block-manager \
#    --num_scheduler_steps ${multi_step} \
#    --max-model-len 9216 \
#    --max-num-batched-tokens 9216 \ # Server see this token then only inference for prefill
#    --distributed_executor_backend ray \
#    --gpu_memory_utilization ${gpu_utils} \
#    --override-generation-config='{"attn_temperature_tuning": true}' \
#    --enable-expert-parallel \
#    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
#pid=$(($!-1))

    # --kv_cache_dtype "fp8_inc" \

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
sleep 10s
echo ${pid}

# ########################################################## Concurrency 64 Random #################################################################
# max_concurrency_client=64
# in_len=1024
# out_len=1024
# start_time=$(date +%s)
# echo "Start to benchmark"
# python3 ../../benchmarks/benchmark_serving.py \
#     --backend vllm \
#     --model ${model} \
#     --tokenizer ${tokenizer} \
#     --request-rate ${request_rate} \
#     --percentile-metrics ttft,tpot,itl,e2el \
#     --dataset-name "random" \
#     --random-input-len ${in_len} \
#     --random-output-len ${out_len} \
#     --random-range-ratio 1.0 \
#     --ignore-eos \
#     --num-prompts ${num_prompts} \
#     --port 18080 \
#     --max-concurrency ${max_concurrency_client} \
#     --save-result 2>&1 | tee benchmark_logs/g3-${model_name}-in${in_len}-out${out_len}-req${request_rate}-num_prompts${num_prompts}-concurrency${max_concurrency_client}.log
# end_time=$(date +%s)
# echo "Time elapsed: $((end_time - start_time))s"
# sleep 10
########################################################## Concurrency 64 Sonnet #################################################################
max_concurrency_client=64
in_len=1024
out_len=1024
start_time=$(date +%s)
echo "Start to benchmark"
python3 ../../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model ${model} \
    --tokenizer ${tokenizer} \
    --dataset-name sonnet \
    --dataset-path ../../benchmarks/sonnet.txt \
    --request-rate ${request_rate} \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
    --num-prompts ${num_prompts} \
    --port 18080 \
    --sonnet-input-len ${in_len} \
    --sonnet-output-len ${out_len} \
    --sonnet-prefix-len 100 \
    --max-concurrency ${max_concurrency_client} \
    --save-result 2>&1 | tee benchmark_logs/g3-${model_name}-in${in_len}-out${out_len}-req${request_rate}-num_prompts${num_prompts}-concurrency${max_concurrency_client}.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"
sleep 10

########################################################## Concurrency 1024 #################################################################
# max_concurrency_client=1024
# in_len=1024
# out_len=1024
# start_time=$(date +%s)
# echo "Start to benchmark"
# python3 ../../benchmarks/benchmark_serving.py \
#     --backend vllm \
#     --model ${model} \
#     --tokenizer ${tokenizer} \
#     --dataset-name sonnet \
#     --dataset-path ../../benchmarks/sonnet.txt \
#     --request-rate ${request_rate} \
#     --percentile-metrics ttft,tpot,itl,e2el \
#     --ignore-eos \
#     --num-prompts ${num_prompts} \
#     --port 18080 \
#     --sonnet-input-len ${in_len} \
#     --sonnet-output-len ${out_len} \
#     --sonnet-prefix-len 100 \
#     --max-concurrency ${max_concurrency_client} \
#     --save-result 2>&1 | tee benchmark_logs/g3-${model_name}-in${in_len}-out${out_len}-req${request_rate}-num_prompts${num_prompts}-concurrency${max_concurrency_client}.log
# end_time=$(date +%s)
# echo "Time elapsed: $((end_time - start_time))s"
# sleep 10
kill ${pid}
