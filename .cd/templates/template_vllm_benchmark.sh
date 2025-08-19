#!/bin/bash

#@VARS

# Wait for vLLM server to be ready
until curl -s http://localhost:8000${ENDPOINT} > /dev/null; do
    echo "Waiting for vLLM server to be ready..."
    sleep 15
done
echo "vLLM server is ready. Starting benchmark..."

SONNET_ARGS=""
if [[ "$DATASET_NAME" == "sonnet" ]]; then
    SONNET_ARGS="--sonnet-prefix-len $PREFIX_LEN --sonnet-input-len $INPUT_TOK --sonnet-output-len $OUTPUT_TOK"
fi

HF_ARGS=""
if [[ "$DATASET_NAME" == "hf" ]]; then
    HF_ARGS="--hf-split train"
fi

CMD="python3 /workspace/vllm/benchmarks/benchmark_serving.py"
CMD+=" --model $MODEL"
CMD+=" --base-url http://localhost:8000"
[[ -n "$TOKENIZER" ]] && CMD+=" --tokenizer $TOKENIZER"
[[ -n "$ENDPOINT" ]] && CMD+=" --endpoint $ENDPOINT"
[[ -n "$BACKEND" ]] && CMD+=" --backend $BACKEND"
[[ -n "$DATASET_NAME" ]] && CMD+=" --dataset-name $DATASET_NAME"
[[ -n "$DATASET" ]] && CMD+=" --dataset-path $DATASET"
[[ -n "$SONNET_ARGS" ]] && CMD+=" $SONNET_ARGS"
[[ -n "$HF_ARGS" ]] && CMD+=" $HF_ARGS"
[[ -n "$NUM_PROMPTS" ]] && CMD+=" --num-prompts $NUM_PROMPTS"
[[ -n "$CONCURRENT_REQ" ]] && CMD+=" --max-concurrency $CONCURRENT_REQ"
[[ -n "$REQUEST_RATE" ]] && CMD+=" --request-rate $REQUEST_RATE"
CMD+=" --metric-percentiles 90"
CMD+=" --ignore-eos"
CMD+=" --trust-remote-code"

echo "Running command: $CMD"

eval "$CMD" 2>&1 | tee -a logs/perftest_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.log