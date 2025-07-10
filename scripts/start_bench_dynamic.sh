#!/bin/bash

export no_proxy=localhost,127.0.0.1
test_model_path="/mnt/weka/data/pytorch/Qwen/Qwen3-30B-A3B"
test_model_path="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/"
# model_path="/mnt/weka/data/pytorch/Qwen/Qwen3-30B-A3B"
model_path="/mnt/weka/data/pytorch/Qwen/Qwen3-30B-A3B"
model_path="/mnt/weka/data/pytorch/DeepSeek-R1/"

# check existence of model path
if [ ! -d "$test_model_path" ]; then
    echo "Model path $test_model_path does not exist."
    test_model_path="/mnt/weka/data/pytorch/DeepSeek-R1/"
    if [ ! -d "$test_model_path" ]; then
        echo "Model path $test_model_path does not exist. Exiting."
        exit 1
    else
        echo "Using model path $test_model_path"
    fi
fi

echo "Using model path: $test_model_path"


echo "Running benchmark with dynamic quantization..."
bash bench_dynamic_quant.sh --model_path $test_model_path  --tp_size 8

echo "Running benchmark with dynamic quantization and INC fork..."
bash bench_dynamic_quant.sh --model_path $test_model_path --tp_size 8 --use_inc 1 --quant_config inc_dynamic_quant_config.json
sleep 10



echo "Running benchmark with dynamic quantization and INC fork..."
bash bench_dynamic_quant.sh --model_path $test_model_path --tp_size 8 --use_inc 1 --quant_config inc_dynamic_quant_only_moe.json
sleep 10


echo "Running benchmark with dynamic quantization and INC fork..."
bash bench_dynamic_quant.sh --model_path $test_model_path --tp_size 8 --use_inc 1 --quant_config inc_dynamic_quant_only_linears.json
sleep 10
