#!/bin/bash

#model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2-static/
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
ip_addr=127.0.0.1
port=8868

export PT_HPU_LAZY_MODE=1

export no_proxy=127.0.0.1

for i in {1..50}
do
  for bs in {64,256,320,384,448,464,480,492,512,640,672,640,512,492,480,464,448,384,320,256,64}
  do
    echo "Starting iteration $i under bs $bs at $(date)"

    bs_repeat=$((bs - 1))
    echo $bs_repeat

    sed -i "s/\(repeat_d_times\) [0-9]*/\1 $bs_repeat/" /workspace/vllm-fork/pd_xpyd/xpyd_start_proxy.sh

    grep "repeat_d_times" /workspace/vllm-fork/pd_xpyd/xpyd_start_proxy.sh

    cd /workspace/vllm-fork/
    source pd_xpyd/xpyd_start_proxy.sh 1 2 1 benchmark &
    #source pd_xpyd/xpyd_start_proxy.sh 2 2 1 &
    PROXY_PID=$!
    sleep 5

    cd /workspace/vllm-fork/test_client
    python3 ../benchmarks/benchmark_serving.py --backend vllm --model $model_path --dataset-name sonnet --request-rate inf --host $ip_addr --port $port --sonnet-input-len 2048 --sonnet-output-len 1024 --sonnet-prefix-len 100 --trust-remote-code --max-concurrency 512 --num-prompts 1 --ignore-eos --burstiness 1000 --dataset-path ../benchmarks/sonnet.txt --save-result 2>&1 | tee -a benchmark_2k_1k_bs_${bs}_run_$i.log

    #./deepseek_sla_test_2p2d.sh
    sleep 10

    echo "Stopping processes..."

    #pkill -f "disagg_proxy_advanced.py"
    pkill -f "disagg_proxy_benchmark.py"

    #kill -9 $(ps -o pgid= $PROXY_PID | grep -o '[0-9]*') 2>/dev/null || true

    sleep 2
  done
done

