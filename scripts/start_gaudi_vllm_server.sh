#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

Help() {
    # Display Help
    echo "Start a vLLM server for a huggingface model on Gaudi."
    echo
    echo "Usage: bash start_gaudi_vllm_server.sh <-w> [-t:m:a:d:i:p:o:b:g:u:e:lc:sf] [-h]"
    echo "Options:"
    echo "-w  Weights of the model, str, could be model id in huggingface or local path."
    echo "    DO NOT change the model name as some of the parameters depend on it."
    echo "-t  tensor-parallel-size for vLLM, int, default=1."
    echo "    Also used to set EP size if it's enable by --enable-expert-parallel"
    echo "-m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None"
    echo "    Used to select HPUs and to set NUMA accordingly. It's recommended to set"
    echo "    for cases with 4 or less HPUs."
    echo "-a  API server URL, str, 'IP:PORT', default=127.0.0.1:30001"
    echo "-d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'"
    echo "-i  Input range, str, format='input_min,input_max', default='4,1024'"
    echo "    Make sure the range cover all the possible lengths from the benchmark/client."
    echo "-p  Max number of prefill sequences, int, default=${PREFERED_BATCHED_TOKENS}/input_min"
    echo "    Used to control the max batch size for prefill to optimize the TTFT."
    echo "-o  Output range, str, format='output_min,output_max', default='4,2048'"
    echo "    Make sure the range cover all the possible lengths from the benchmark/client."
    echo "-b  max-num-seqs for vLLM, int, default=128"
    echo "    Used to control the max batch size for decoding phase."
    echo "    It is recommended to set this value according to the 'Maximum concurrency'"
    echo "    reported by a test run."
    echo "-g  max-seq-len-to-capture for vLLM, int, default=${PREFERED_SEQ_LEN_TO_CAPTURE}"
    echo "    Used to control the maximum batched tokens to be captured in HPUgraph."
    echo "    Reduce this value could decrease memory usage, but not smaller than 2048."
    echo "-u  gpu-memory-utilization, float, default=0.9"
    echo "    Used to control the GPU memory utilization. Reduce this value if OOM occurs."
    echo "-e  Extra vLLM server parameters, str, default=None"
    echo "    Extra parameters that will pass to the vLLM server."
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

weights_path=""
num_hpu=1
module_ids=None
host=127.0.0.1
port=30001
dtype=bfloat16
input_min=4
input_max=1024
max_num_prefill_seqs=""
output_min=4
output_max=2048
max_num_seqs=$PREFERED_NUM_SEQS
max_seq_len_to_capture=$PREFERED_SEQ_LEN_TO_CAPTURE
gpu_memory_utilization=0.9
extra_params=()
cache_path=""
skip_warmup=false
profile=false

# Get the options
while getopts hw:t:m:a:d:i:p:o:b:e:g:u:lc:sf flag; do
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
    a) # get the URL of the server
        host=${OPTARG%%:*}
        port=${OPTARG##*:}
        ;;
    d) # get data type
        dtype=$OPTARG ;;
    i) # input range
        input_min=${OPTARG%%,*}
        input_max=${OPTARG##*,}
        ;;
    p) # max number of prefill sequences
        max_num_prefill_seqs=$OPTARG ;;
    o) # output range
        output_min=${OPTARG%%,*}
        output_max=${OPTARG##*,}
        ;;
    b) # batch size
        max_num_seqs=$OPTARG ;;
    e) # extra vLLM server parameters
        IFS=" " read -r -a extra_params <<< "$OPTARG" ;;
    g) # max-seq-len-to-capture
        max_seq_len_to_capture=$OPTARG ;;
    u) # gpu-memory-utilization
        gpu_memory_utilization=$OPTARG ;;
    l) # use linear bucketing
        use_linear_bucketing=true ;;
    c) # use_recipe_cache
        cache_path=$OPTARG ;;
    s) # skip_warmup
        skip_warmup=true ;;
    f) # enable high-level profiler
        profile=true ;;
    \?) # Invalid option
        echo "Error: Invalid option"
        Help
        exit
        ;;
    esac
done

if [ "$weights_path" == "" ]; then
    echo "[ERROR]: No model specified. Usage:"
    Help
    exit
fi

model_name=$(basename "$weights_path")
model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')

max_num_batched_tokens=$(( $input_max + $output_max ))
if [ "$max_num_batched_tokens" -lt $PREFERED_BATCHED_TOKENS ]; then
    max_num_batched_tokens=$PREFERED_BATCHED_TOKENS
fi
# Ceiling max_num_batched_tokens to a multiple of BLOCK_SIZE
max_num_batched_tokens=$( ceil $max_num_batched_tokens $BLOCK_SIZE )

if [ "$max_num_prefill_seqs" == "" ]; then
    # Ceiling input_min to a multiple of BLOCK_SIZE
    input_min_ceil=$( ceil $input_min $BLOCK_SIZE )
    max_num_prefill_seqs=$(( $max_num_batched_tokens / input_min_ceil ))
    max_num_prefill_seqs=$( max $max_num_prefill_seqs 1 )
fi

max_model_len=$(( $input_max + $output_max ))
# Ceiling max_model_len to a multiple of BLOCK_SIZE
max_model_len=$( ceil $max_model_len $BLOCK_SIZE )

if [ "$input_min" == "$input_max" ]; then
    disable_zero_padding=true
fi


echo "Starting vllm server for ${model_name} from ${weights_path} with:"
echo "    device: ${num_hpu} HPUs with module_ids=${module_ids}"
echo "    URL: ${host}:${port}"
echo "    input_range: [${input_min}, ${input_max}]"
echo "    output_range: [${output_min}, ${output_max}]"
echo "    max_num_seqs: ${max_num_seqs}"

case_name=serve_${model_name}_${dtype}_${device}_in${input_min}-${input_max}_out${output_min}-${output_max}_bs${max_num_seqs}_tp${num_hpu}_$(date +%F-%H-%M-%S)

set_config

command_string=$(echo ${NUMA_CTL} \
python3 -m vllm.entrypoints.openai.api_server \
    --device hpu \
    --block-size "${BLOCK_SIZE}" \
    --host "${host}" --port "${port}" \
    --model "${weights_path}" \
    --dtype "${DATA_TYPE}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-seq-len-to-capture "${max_seq_len_to_capture}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --max-model-len "${max_model_len}" \
    --tensor-parallel-size "${num_hpu}" \
    --trust-remote-code \
    --seed 2025 \
    --distributed_executor_backend "${dist_backend}" \
    "${extra_params[@]}" \
    |& tee "${case_name}".log 2>&1)

echo "vllm server started with command:"
echo "${command_string}"

eval "${command_string}"
