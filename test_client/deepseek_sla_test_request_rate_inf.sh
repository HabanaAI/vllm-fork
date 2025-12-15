#!/bin/bash

#model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2-static/
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
ip_addr=127.0.0.1
port=8868

export PT_HPU_LAZY_MODE=1

export no_proxy=127.0.0.1

mkdir -p benchmark_log

test_benchmark_serving() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$(( local_max_concurrency * 16 ))
    config=$4
    local_len_ratio=1.0
    start=$(date +%s)
    request_rate=inf

    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency, prompt_num: $local_num_prompts"
    log_name=benchmark_serving_DeepSeek-R1_cardnumber_${config}_datatype_bfloat16_sonnet_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_${request_rate}_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)

    python3 ../benchmarks/benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port --dataset-name sonnet --dataset-path ../benchmarks/sonnet.txt --sonnet-input-len $local_input --sonnet-output-len $local_output --sonnet-prefix-len 100 --max_concurrency $local_max_concurrency --num-prompts $local_num_prompts --request-rate $request_rate --seed 0 --ignore_eos --burstiness 1000 --save-result --result-filename ${log_name}.json |& tee ${log_name}.log > /dev/null

    end=$(date +%s)
    output_throughput=$(grep "Output token throughput (tok/s):" ${log_name}.log | awk -F ':' '{print $2}' | xargs)
    mean_tpot=$(grep "Mean TPOT (ms):" ${log_name}.log | awk -F ":" '{print $2}' | xargs)
    echo "Fixed-length dataset, input len: $local_input, output len: $local_output, output throughput (tok/s): $output_throughput, mean TPOT (ms): $mean_tpot, time taken: $(( end - start )) seconds"

    mv benchmark_serving_DeepSeek* benchmark_log
}

config="${1:-1p2d}"

echo "Runing under config: $config"

for bs in 1 2 4 8 16 32 64 96 128 192 208 224 256 384 512 672 800 1024 1280
do
  test_benchmark_serving 256 256 $bs $config
  test_benchmark_serving 256 1024 $bs $config
  test_benchmark_serving 1024 256 $bs $config
  test_benchmark_serving 512 512 $bs $config
  test_benchmark_serving 1024 1024 $bs $config
  test_benchmark_serving 2048 1024 $bs $config
  test_benchmark_serving 3584 1536 $bs $config
  test_benchmark_serving 8192 1024 $bs $config
done 
