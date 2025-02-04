
#!/bin/bash

 VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 python /app/vllm-fork/benchmarks/benchmark_throughput.py --model /mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/ --device hpu --seed 2024 --backend vllm --dataset ../ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2 --dtype bfloat16

