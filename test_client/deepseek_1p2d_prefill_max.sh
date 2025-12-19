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
    log_name=benchmark_serving_DeepSeek-R1_cardnumber_${config}_datatype_fp8_sonnet_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_${request_rate}_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)

    python3 ../benchmarks/benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port --dataset-name sonnet --dataset-path ../benchmarks/sonnet.txt --sonnet-input-len $local_input --sonnet-output-len $local_output --sonnet-prefix-len 100 --max_concurrency $local_max_concurrency --num-prompts $local_num_prompts --request-rate $request_rate --seed 0 --ignore_eos --burstiness 1000 --save-result --result-filename ${log_name}.json |& tee ${log_name}.log > /dev/null

    end=$(date +%s)
    output_throughput=$(grep "Output token throughput (tok/s):" ${log_name}.log | awk -F ':' '{print $2}' | xargs)
    mean_tpot=$(grep "Mean TPOT (ms):" ${log_name}.log | awk -F ":" '{print $2}' | xargs)
    echo "Fixed-length dataset, input len: $local_input, output len: $local_output, output throughput (tok/s): $output_throughput, mean TPOT (ms): $mean_tpot, time taken: $(( end - start )) seconds"

    mv benchmark_serving_DeepSeek* benchmark_log
}

config="${1:-1p2d}"

echo "Runing under config: $config"

#for bs in 1 2 4 6 8 10 12 14 16 
for bs in 1 8
do
  test_benchmark_serving 2048 4 $bs $config
  test_benchmark_serving 3584 4 $bs $config
  test_benchmark_serving 8192 4 $bs $config
  test_benchmark_serving 16384 4 $bs $config
  test_benchmark_serving 24576 4 $bs $config
  test_benchmark_serving 32768 4 $bs $config
  test_benchmark_serving 40960 4 $bs $config
  test_benchmark_serving 49152 4 $bs $config
  test_benchmark_serving 57344 4 $bs $config
  test_benchmark_serving 65536 4 $bs $config
  test_benchmark_serving 73728 4 $bs $config
  test_benchmark_serving 81920 4 $bs $config
  test_benchmark_serving 130000 1 $bs $config
  test_benchmark_serving 134000 1 $bs $config
  test_benchmark_serving 140000 1 $bs $config
done 
