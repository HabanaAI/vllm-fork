pkill -9 python

export PT_HPU_LAZY_MODE=1 
export GRAPH_VISUALIZATION=1 
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DISABLE_MARK_SCALES_AS_CONST=1


#############################
# Qwen
#############################
# FIXME: (Yi) Enable the static MoE path 
export VLLM_DYNAMIC_MOE_MIN_TOKENS=0



#!/bin/bash

set -e

MODE=$1  # First argument
MODEL=$2 # Second argument (model name)
TOKENIZER=$MODEL
tp_size=1
ep_size=1

while [ $# -gt 0 ]; do
  if [ "$1" == "--ep_size" ]; then
    shift
    ep_size=$1
  elif [ "$1" == "--tp_size" ]; then
    shift
    tp_size=$1
  elif [ "$1" == "--tokenizer" ]; then
    shift
    TOKENIZER=$1
  else
    shift
  fi
done


if [ -z "$MODE" ] || [ -z "$MODEL" ] || [ -z "$TOKENIZER" ]; then
  echo "Usage: $0 {bf16|calib|quant|eval} <model> <tokenizer>"
  exit 1
fi

COMMON_ARGS="--model $MODEL --tokenizer $TOKENIZER --osl 32 --max_model_len 2048 --max_num_seqs 1 --tp_size ${tp_size} --ep_size ${ep_size}"

if [ "$MODE" == "bf16" ]; then
  python run_example_tp_qwen.py $COMMON_ARGS

elif [ "$MODE" == "calib" ]; then
  echo "Calibrate model ${MODEL} with config ${QUANT_CONFIG}"
  python run_example_tp_qwen.py $COMMON_ARGS --inc --dataset pile --nprompts 512

else
  echo "Unknown mode: $MODE"
  echo "Valid modes are: bf16, calib"
  exit 1
fi

