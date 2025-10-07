#!/bin/bash

python3 benchmarks/ben2.py \
    --backend openai-chat \
    --model "ibm-granite/granite-3.3-8b-instruct" \
    --dataset-name tool_calling \
    --num-prompts 128 \
    --max-concurrency 8 \
    --request-rate inf \
    --tool-calling-input-tokens 4096 \
    --tool-calling-output-tokens 1024 \
    --endpoint "/v1/chat/completions" --metric-percentiles 90 --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos --port 8080