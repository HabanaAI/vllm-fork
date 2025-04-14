#!/bin/bash

save_dir="Llama-4-Scout-mmlupro_acc_output"
model="/mnt/weka/llm/Llama-4-Scout-17B-16E-Instruct/"
selected_subjects="all"
gpu_util=0.8

cd MMLU-Pro/
PT_HPU_LAZY_MODE=1 \
VLLM_SKIP_WARMUP=true \
HABANA_VISIBLE_DEVICES="ALL" \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
python evaluate_from_local.py \
    --selected_subjects $selected_subjects \
    --save_dir $save_dir \
    --model $model \
    --global_record_file ${save_dir}/eval_record_collection.csv \
    --gpu_util $gpu_util

cd ..