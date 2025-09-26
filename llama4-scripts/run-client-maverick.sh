#!/bin/bash
max_concurrency_client=16
in_len=25
out_len=10
model="/workdir/hf_models/meta-llama_Llama-4-Maverick-17B-128E-Instruct/"
tokenizer="/workdir/hf_models/meta-llama_Llama-4-Maverick-17B-128E-Instruct/"
num_prompts=1
model_name="Maverick"
request_rate=inf
start_time=$(date +%s)
echo "Start to benchmark"
export no_proxy=127.0.0.1,localhost

python3 ../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model ${model} \
    --tokenizer ${tokenizer} \
    --dataset-name sonnet \
    --dataset-path ../benchmarks/sonnet.txt \
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
curl -X POST http://localhost:18080/start_profile
python3 ../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model ${model} \
    --tokenizer ${tokenizer} \
    --dataset-name sonnet \
    --dataset-path ../benchmarks/sonnet.txt \
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
curl -X POST http://localhost:18080/stop_profile
