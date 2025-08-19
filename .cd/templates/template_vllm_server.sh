#!/bin/bash

#@VARS

## Start server
CMD="vllm serve $MODEL"

[[ -n "$BLOCK_SIZE" ]] && CMD+=" --block-size $BLOCK_SIZE"
[[ -n "$DTYPE" ]] && CMD+=" --dtype $DTYPE"
[[ -n "$TENSOR_PARALLEL_SIZE" ]] && CMD+=" --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
[[ -n "$HF_HOME" ]] && CMD+=" --download_dir $HF_HOME"
[[ -n "$MAX_MODEL_LEN" ]] && CMD+=" --max-model-len $MAX_MODEL_LEN"
[[ -n "$GPU_MEM_UTILIZATION" ]] && CMD+=" --gpu-memory-utilization $GPU_MEM_UTILIZATION"
[[ -n "$MAX_NUM_SEQS" ]] && CMD+=" --max-num-seqs $MAX_NUM_SEQS"
[[ -n "$MAX_NUM_PREFILL_SEQS" ]] && CMD+=" --max-num-prefill-seqs $MAX_NUM_PREFILL_SEQS"
CMD+=" --use-padding-aware-scheduling"
CMD+=" --num-scheduler-steps 1"
CMD+=" --disable-log-requests"

echo "Running command: $CMD"

eval "$CMD" 2>&1 | tee -a logs/vllm_server.log
