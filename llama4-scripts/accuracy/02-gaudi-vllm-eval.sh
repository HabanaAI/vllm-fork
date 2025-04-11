#!/bin/bash
model_path=$1
python3 -m eval.run eval_vllm \
        --model_name ${model_path} \
        --url http://localhost:18080 \
        --output_dir ~/tmp \
        --eval_name "chartqa"