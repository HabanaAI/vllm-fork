#!/bin/bash
tp_parrallel=8
gpu_utils=0.8
bs=64

log_name=$1
model_path=$2

model=$model_path

echo ========= Run accuracy on ${model} ==============
mkdir -p benchmark_logs
#ENABLE_CONSOLE=true LOG_LEVEL_PT_FALLBACK=1 \
PT_HPU_LAZY_MODE=1 \
VLLM_SKIP_WARMUP=true \
HABANA_VISIBLE_DEVICES="ALL" \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
python -m vllm.entrypoints.openai.api_server \
    --port 18080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --max-model-len 16384 \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --distributed_executor_backend mp \
    --gpu_memory_utilization ${gpu_utils} \
    --enforce-eager \
    --enable-expert-parallel \
    --override-generation-config='{"attn_temperature_tuning": true}' \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
pid=$(($!-1))
echo $pid > accuracy_server.pid
