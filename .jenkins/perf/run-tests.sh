#!/bin/bash

usage() {
    echo``
    echo "Runs perf tests using vllm and compares to "
    echo "precomputed baseline"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -b    - baseline throughput [total tokens/s]"
    echo "  -i    - input length"
    echo "  -m    - path to the model"
    echo "  -o    - output length"
    echo "  -p    - number of prompts"
    echo "  -s    - max number of sequences"
    echo "  -t    - tensor parallel size"
    echo
}

while getopts "b:i:m:o:p:s:t:" OPT; do
  case ${OPT} in
    b )
        baseline="$OPTARG"
        ;;
    i )
        input_len="$OPTARG"
        ;;
    m )
        model="$OPTARG"
        ;;
    o )
        output_len="$OPTARG"
        ;;
    p )
        num_prompts="$OPTARG"
        ;;
    s )
        max_num_seqs="$OPTARG"
        ;;
    t )
        tp="$OPTARG"
        ;;
    \? )
        usage
        exit 1
        ;;
  esac
done

export VLLM_DECODE_BS_BUCKET_MIN=$max_num_seqs
export VLLM_DECODE_BLOCK_BUCKET_MIN=$(($input_len/128*$max_num_seqs))
export VLLM_DECODE_BLOCK_BUCKET_MAX=$((($input_len+$output_len)/128*$max_num_seqs+2*128))
export VLLM_PROMPT_SEQ_BUCKET_MIN=$input_len
export VLLM_PROMPT_SEQ_BUCKET_MAX=$input_len

OUTPUT=$(python3 ../../benchmarks/benchmark_throughput.py \
--device=hpu \
--seed=2024 \
--backend=vllm \
--num-prompts="$num_prompts" \
--dtype=bfloat16 \
--max-model-len=16384 \
--max-num-batched-tokens=16384 \
--max-num-seqs="$max_num_seqs" \
--model="$model" \
--use-padding-aware-scheduling \
--input-len $input_len \
--output-len $output_len \
--num-scheduler-steps=32 \
-tp $tp | tail -n 1)
echo $OUTPUT

THROUGHPUT=$(echo $OUTPUT | grep -oP '\d+.\d+.(?= total tokens)')
if (( $(echo "$THROUGHPUT > $baseline" | bc -l) )); then
    echo "=== PASSED MODEL: ${model} ==="
    exit 0
else
    echo "=== FAILED MODEL: ${model} ==="
    exit 1
fi