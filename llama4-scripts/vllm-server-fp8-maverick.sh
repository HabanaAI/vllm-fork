#!/bin/bash
tp_parrallel=8
in_len=122880
out_len=8192
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
bs=32
block_size=256
block_step=256
 
log_name="[inc-staticquant-scalar-fp8matmul-split]online-gaudi3-${gpu_utils}util-TPparallel${tp_parrallel}-EP${ep_size}-loop${moe_n_slice}moegroups-multistep${multi_step}_nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}_mdllen${total_len}"
 
prompt_bs_max=1
# Increase prompt max seq len bucket just in case there are more tokens than provided due to tokenizer
prompt_seq_max=$((in_len + $block_size))
total_len_aligned=$(((total_len + block_size - 1) / block_size * block_size))
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len * bs / block_size))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len_aligned * bs / block_size + block_step))
 
model="/mnt/weka/llm/Llama-4-Maverick-17B-128E-Instruct"
tokenizer="/mnt/weka/llm/Llama-4-Maverick-17B-128E-Instruct"
model_name="Maverick"
 
mkdir -p benchmark_logs
#export VLLM_SKIP_WARMUP=True

RECIPE_FLAG="visionbf16_kvfp8_splitqkv" # Currently best functional with perf

if [[ "$RECIPE_FLAG" == "meta" ]]; then
  echo "INFO: Flag is 'meta'. Using meta recipe without FP8 KV cache. Meta's recipe (Vision and text is working but no fp8 kv cache)"
  export QUANT_CONFIG="./vllm-hpu-extension-quant/llama-4-maverick-17b-128e-instruct/maxabs_quant_g3_meta_recipe.json"
  KV_CACHE_ARGS=""
elif [[ "$RECIPE_FLAG" == "high_perf" ]]; then
  echo "INFO: Flag is 'high_perf'. Using default recipe with FP8 KV cache. Current high performance code"
  export QUANT_CONFIG="./vllm-hpu-extension-quant/llama-4-maverick-17b-128e-instruct/maxabs_quant_g3.json"
  KV_CACHE_ARGS="--kv-cache-dtype fp8_inc"
elif [[ "$RECIPE_FLAG" == "visionbf16_kvfp8" ]]; then
  echo "INFO: Flag is 'visionbf16_kvfp8'. Using default recipe with FP8 KV cache with vision BF16."
  export QUANT_CONFIG="./vllm-hpu-extension-quant/llama-4-maverick-17b-128e-instruct/maxabs_quant_g3_vision_enabled_kvfp8_recipe.json"
  KV_CACHE_ARGS="--kv-cache-dtype fp8_inc"
elif [[ "$RECIPE_FLAG" == "visionbf16_kvfp8_splitqkv" ]]; then
  echo "INFO: Flag is 'visionbf16_kvfp8_splitqkv'. Using default recipe with FP8 KV cache with vision BF16 with splitqkv."
  export QUANT_CONFIG="./vllm-hpu-extension-quant-splitqkv/llama-4-maverick-17b-128e-instruct/maxabs_quant_g3.json"
  KV_CACHE_ARGS="--kv-cache-dtype fp8_inc --split_qkv"
fi

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
PT_HPU_LAZY_MODE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_EXPONENTIAL_BUCKETING=false \
vllm serve ${model} \
    --port 18080 \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --num_scheduler_steps ${multi_step} \
    --max-model-len ${total_len} \
    --max-num-batched-tokens 131072 \
    --use-padding-aware-scheduling \
    --block-size ${block_size} \
    --weights-load-device cpu \
    --use-v2-block-manager \
    ${KV_CACHE_ARGS} \
    --gpu_memory_utilization ${gpu_utils} \
    --quantization inc \
    --enable-expert-parallel \
    --override-generation-config='{"attn_temperature_tuning": true}' \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &

pid=$(($!-1))
echo "vLLM server started with PID: $pid"

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
echo ${pid}
