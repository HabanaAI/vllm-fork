#!/bin/bash
tp_parrallel=8
in_len=10240
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
bs=128
num_prompts=128
request_rate=inf

log_name="[inc-staticquant-scalar-fp8matmul-split]online-gaudi3-${gpu_utils}util-TPparallel${tp_parrallel}-EP${ep_size}-loop${moe_n_slice}moegroups-multistep${multi_step}_nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}_mdllen${total_len}"

in_len_aligned=$(((in_len + 127) / 128 * 128))
echo "====================================================== in_len_aligned = ${in_len_aligned}"
total_len_aligned=$(((total_len + 127) / 128 * 128))
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len_aligned * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MIN=$(((VLLM_DECODE_BLOCK_BUCKET_MIN + 127) / 128 * 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len_aligned * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$(((VLLM_DECODE_BLOCK_BUCKET_MAX + 127) / 128 * 128))

# model="/data/models/DeepSeek-R1-static/"
# tokenizer="/data/models/DeepSeek-R1-static/"
model="/root/data/Llama-4-Scout-17B-16E-Instruct/"
tokenizer="/root/data/Llama-4-Scout-17B-16E-Instruct/"
# tokenizer="/data/models/DeepSeek-R1/"
model_name="Scout"

mkdir -p benchmark_logs
# QUANT_CONFIG="scripts/inc_quant_with_fp8kv_config.json" \
# VLLM_REQUANT_FP8_INC=1 \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# VLLM_USE_FP8_MATMUL=true \
# VLLM_MOE_N_SLICE=${moe_n_slice} \
# VLLM_MLA_DISABLE_REQUANTIZATION=1 \
ENABLE_CONSOLE=true LOG_LEVEL_PT_FALLBACK=1 \
PT_HPU_LAZY_MODE=1 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=8 \
VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len_aligned} \
VLLM_PROMPT_SEQ_BUCKET_MAX=${in_len_aligned} \
VLLM_DECODE_BS_BUCKET_MIN=${bs} \
VLLM_DECODE_BS_BUCKET_MAX=${bs} \
VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
VLLM_DELAYED_SAMPLING=true \
HABANA_VISIBLE_DEVICES="ALL" \
VLLM_EP_SIZE=${ep_size} \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
python -m vllm.entrypoints.openai.api_server \
    --port 18080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --num_scheduler_steps ${multi_step} \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --distributed_executor_backend ray \
    --gpu_memory_utilization ${gpu_utils} \
    --enforce-eager \
    --override-generation-config='{"attn_temperature_tuning": true}' \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
pid=$(($!-1))
