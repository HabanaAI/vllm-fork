pkill -9 python

export PT_HPU_LAZY_MODE=1 
export GRAPH_VISUALIZATION=1 
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DISABLE_MARK_SCALES_AS_CONST=1

export OFFICIAL_MODEL="/mnt/disk5/Qwen3-30B-A3B-250425"

#############################
# Qwen
#############################
# FIXME: (Yi) Enable the static MoE path 
export VLLM_DYNAMIC_MOE_MIN_TOKENS=0



#!/bin/bash

set -e

MODE=$1  # First argument
MODEL=$2 # Second argument (model name)
TOKENIZER=$3 # Third argument (tokenizer name)

if [ -z "$TOKENIZER" ]; then
  TOKENIZER=$MODEL
fi

if [ -z "$MODE" ] || [ -z "$MODEL" ] || [ -z "$TOKENIZER" ]; then
  echo "Usage: $0 {bf16|calib|quant|eval} <model> <tokenizer>"
  exit 1
fi

COMMON_ARGS="--model $MODEL --tokenizer $TOKENIZER --osl 32 --max_model_len 2048 --max_num_seqs 1 --tp_size 1 --ep_size 1"

model_name=$(basename ${MODEL})
if [ ${model_name} == "Qwen3-30B-A3B-250425" ]; then
    quant_file_path="inc_meaure_g3_30B_A3B.json"
elif [ ${model_name} == "Qwen3-32B-250426" ]; then
    quant_file_path="inc_meaure_g3_32B.json"
else
    echo "Unknown model name: ${model_name}"
    exit 1
fi


if [ "$MODE" == "bf16" ]; then
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS

elif [ "$MODE" == "calib" ]; then
  QUANT_CONFIG=${quant_file_path} \
  python ./scripts/run_example_tp_qwen.py $COMMON_ARGS --inc --dataset pile --nprompts 512

else
  echo "Unknown mode: $MODE"
  echo "Valid modes are: bf16, calib"
  exit 1
fi

