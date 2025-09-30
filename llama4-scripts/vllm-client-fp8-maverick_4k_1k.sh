#!/bin/bash
model="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
tokenizer="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
model_name="Maverick"
 
mkdir -p benchmark_logs

########################################################## Concurrency 16 Sonnet #################################################################
max_concurrency_client=32
num_prompts=32
request_rate=inf
in_len=4096
out_len=1024
start_time=$(date +%s)
echo "Start to benchmark"
python3 ../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model ${model} \
    --tokenizer ${tokenizer} \
    --dataset-name random \
    --request-rate ${request_rate} \
    --percentile-metrics ttft,tpot,itl,e2el \
	--metric-percentiles 50,90,99 \
    --ignore-eos \
    --num-prompts ${num_prompts} \
    --port 18080 \
    --random-input-len ${in_len} \
    --random-output-len ${out_len} \
    --max-concurrency ${max_concurrency_client} \
    --save-result 2>&1 | tee benchmark_logs/g3-${model_name}-in${in_len}-out${out_len}-req${request_rate}-num_prompts${num_prompts}-concurrency${max_concurrency_client}.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"
sleep 10