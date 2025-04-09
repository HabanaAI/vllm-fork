#!/bin/bash

python3 -m eval.run eval_vllm \
        --model_name "/root/data/Llama-4-Scout-17B-16E-Instruct/" \
        --url http://localhost:18080 \
        --output_dir ~/tmp \
        --eval_name "chartqa"