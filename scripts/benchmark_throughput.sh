#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

Help() {
    # Display Help
    echo "Benchmark vLLM throughput for a huggingface model on Gaudi."
    echo
    echo "Syntax: bash benchmark_throughput.sh <-w> [-t:m:d:i:o:r:j:t:l:b:c:sfza] [-h]"
    echo "options:"
    echo "-w  Weights of the model, str, could be model id in huggingface or local path."
    echo "    DO NOT change the model name as some of the parameters depend on it."
    echo "-t  Number of HPUs to use, [1-8], default=1. Used to set TP and EP size."
    echo "-m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None"
    echo "    Used to select HPUs and to set NUMA accordingly. It's recommended to set"
    echo "    for cases with 4 or less HPUs."
    echo "-d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'"
    echo "-i  Input length, int, default=1024"
    echo "-p  Max number of prefill sequences, int, default=${PREFERED_BATCHED_TOKENS}/input_min"
    echo "-o  Output length, int, default=512"
    echo "-b  max_num_seqs for vllm, int, default=128"
    echo "    Used to control the max batch size for decoding phase."
    echo "    It is recommended to set this value according to the 'Maximum concurrency'"
    echo "    reported by a test run."
    echo "-r  random-range-ratio for benchmark_throughput.py, float, default=0.0"
    echo "    The result range is [length * (1 - range_ratio), length * (1 + range_ratio)]."
    echo "-j  Json path of the ShareGPT dataset, str, default=None"
    echo "    set -j <sharegpt json path> will override -i, -o and -r"
    echo "-n  Number of prompts, int, default=max_num_seqs*4"
    echo "-g  max-seq-len-to-capture for vLLM, int, default=${PREFERED_SEQ_LEN_TO_CAPTURE}"
    echo "    Used to control the maximum batched tokens to be captured in HPUgraph."
    echo "    Reduce this value could decrease memory usage, but not smaller than 2048."
    echo "-u  gpu-memory-utilization, float, default=0.9"
    echo "    Used to control the GPU memory utilization. Reduce this value if OOM occurs."
    echo "-e  Extra vLLM server parameters, str, default=None"
    echo "    Extra parameters that will pass to the vLLM engine."
    echo "-l  Use linear bucketing or not, bool, default=false"
    echo "    The exponential bucketing is used by default to reduce number of buckets."
    echo "    Turn on to switch to linear bucketing that introduce less padding, and more"
    echo "    buckets and thus longer warmup time."
    echo "-c  Cache HPU recipe to the specified path, str, default=None"
    echo "    The recipe cache could be reused to reduce the warmup time."
    echo "-s  Skip warmup or not, bool, default=false"
    echo "    Skip warmup to reduce startup time. Used in debug/dev environment only."
    echo "    DO NOT use in production environment."
    echo "-f  Enable high-level profiler or not, bool, default=false"
    echo "-h  Help info"
    echo
}

# Get the options
while getopts hw:t:m:d:i:p:o:b:r:j:n:g:u:e:lc:sf flag; do
    case $flag in
    h) # display Help
        Help
        exit
        ;;
    w) # get model path
        weights_path=$OPTARG ;;
    t) # get number of HPUs
        num_hpu=$OPTARG ;;
    m) # get module ids to use
        module_ids=$OPTARG ;;
    d) # get data type
        dtype=$OPTARG ;;
    i) # input range
        input_len=$OPTARG ;;
    p) # max number of prefill sequences
        max_num_prefill_seqs=$OPTARG ;;
    o) # output range
        output_len=$OPTARG ;;
    b) # batch size
        max_num_seqs=$OPTARG ;;
    r) # random-range-ratio
        range_ratio=$OPTARG ;;
    j) # json path
        json_path=$OPTARG ;;
    n) # number of prompts
        num_prompts=$OPTARG ;;
    g) # max-seq-len-to-capture
        max_seq_len_to_capture=$OPTARG ;;
    u) # gpu-memory-utilization
        gpu_memory_utilization=$OPTARG ;;
    e) # extra vLLM server parameters
        IFS=" " read -r -a extra_params <<< "$OPTARG" ;;
    l) # use linear bucketing
        use_linear_bucketing=true ;;
    c) # use_recipe_cache
        cache_path=$OPTARG ;;
    s) # skip_warmup
        skip_warmup=true ;;
    f) # enable profiling
        profile=true ;;
    \?) # Invalid option
        echo "Error: Invalid option"
        Help
        exit
        ;;
    esac
done

if [ -z "$weights_path" ]; then
    echo "[ERROR]: No model specified. Usage:"
    Help
    exit
fi

num_hpu=${num_hpu:-"1"}
module_ids=${module_ids:-"None"}
dtype=${dtype:-"bfloat16"}
input_len=${input_len:-"1024"}
max_num_prefill_seqs=${max_num_prefill_seqs:-""}
output_len=${output_len:-"512"}
max_num_seqs=${max_num_seqs:-${PREFERED_NUM_SEQS}}
range_ratio=${range_ratio:-"0.0"}
json_path=${json_path:-""}
num_prompts=${num_prompts:-$(($max_num_seqs * 4))}
max_seq_len_to_capture=${max_seq_len_to_capture:-$PREFERED_SEQ_LEN_TO_CAPTURE}
gpu_memory_utilization=${gpu_memory_utilization:-"0.9"}
extra_params=(${extra_params[@]:-})
use_linear_bucketing=${use_linear_bucketing:-"false"}
cache_path=${cache_path:-""}
skip_warmup=${skip_warmup:-"false"}
profile=${profile:-"false"}

model_name=$(basename "$weights_path")
model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')

if [ "$json_path" != "" ]; then
    input_min=4
    input_max=1024
    output_min=4
    output_max=2048
    IO_FLAGS=(--dataset-name "sharegpt" --dataset-path "$json_path")
    echo "Benchmarking throughput for ${model_name} from ${weights_path} using ${num_prompts} random prompts from ${json_path} with max_num_batched_tokens=${max_num_batched_tokens}, max_model_len=${max_model_len} using ${num_hpu} HPUs with module_ids=${module_ids}"
    case_name="benchmark_throughput_${model_name}_${dtype}_${device}_sharegpt_bs${max_num_seqs}_tp${num_hpu}_$(date +%F-%H-%M-%S)"
elif [ "$range_ratio" == "0.0" ]; then
    input_min=$input_len
    input_max=$input_len
    output_min=$output_len
    output_max=$output_len
    disable_zero_padding=true
    IO_FLAGS=(--input-len "$input_len" --output-len "$output_len")
    echo "Benchmarking throughput for ${model_name} from ${weights_path} using ${num_prompts} fixed-length prompts with input_len=${input_len}, output_len=${output_len}, max_num_seqs=${max_num_seqs}, max_num_batched_tokens=${max_num_batched_tokens}, max_model_len=${max_model_len} using ${num_hpu} HPUs with module_ids=${module_ids}"
    case_name="benchmark_throughput_${model_name}_${dtype}_${device}_in${input_len}_out${output_len}_bs${max_num_seqs}_tp${num_hpu}_$(date +%F-%H-%M-%S)"
else
    input_min=$(bc <<< "($input_len * ( 1 - $range_ratio) + 0.5) / 1")
    input_max=$(bc <<< "($input_len * ( 1 + $range_ratio) + 0.5) / 1")
    output_min=$(bc <<< "($output_len * ( 1 - $range_ratio) + 0.5) / 1")
    output_max=$(bc <<< "($output_len * ( 1 + $range_ratio) + 0.5) / 1")
    IO_FLAGS=(--dataset-name "random" --input-len "$input_len" --output-len "$output_len" --random-range-ratio "$range_ratio")
    echo "Benchmarking throughput for ${model_name} from ${weights_path} using ${num_prompts} random-length prompts with input_range=[${input_min}, ${input_max}], output_range=[${output_min}, ${output_max}], max_num_seqs=${max_num_seqs}, max_num_batched_tokens=${max_num_batched_tokens}, max_model_len=${max_model_len} using ${num_hpu} HPUs with module_ids=${module_ids}"
    case_name="benchmark_throughput_${model_name}_${dtype}_${device}_in${input_min}-${input_max}_out${output_min}-${output_max}_bs${max_num_seqs}_tp${num_hpu}_$(date +%F-%H-%M-%S)"
fi
log_file="${case_name}.log"

set_config

echo "Changed environment variables:" |& tee "${log_file}"
echo -e "${changed_env}\n" |& tee -a "${log_file}"

command_string=$(echo ${NUMA_CTL} \
python3 "$BASH_DIR/../benchmarks/benchmark_throughput.py" \
    --backend vllm \
    --block-size "${BLOCK_SIZE}" \
    --model "${weights_path}" \
    --dtype "${DATA_TYPE}" \
    "${IO_FLAGS[@]}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-seq-len-to-capture "${max_seq_len_to_capture}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --max-model-len "${max_model_len}" \
    --tensor-parallel-size "${num_hpu}" \
    --trust-remote-code \
    --seed 2025 \
    --num-prompts "${num_prompts}" \
    --output-json "${case_name}"_result.json \
    --distributed_executor_backend "${dist_backend}" \
    "${extra_params[@]}")

echo "Benchmark throughput for ${model_name} on Gaudi ${device} with command:" |& tee -a "${log_file}"
echo "${command_string}" |& tee -a "${log_file}"
echo "The log will be saved to ${case_name}.log"

eval "${command_string}" |& tee -a "${case_name}".log 2>&1
