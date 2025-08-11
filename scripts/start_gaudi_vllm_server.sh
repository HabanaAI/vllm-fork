#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

Help() {
    # Display Help
    echo "Start vllm server for a huggingface model on Gaudi."
    echo
    echo "Syntax: bash start_gaudi_vllm_server.sh <-w> [-n:m:u:p:d:i:o:t:l:b:e:c:sfza] [-h]"
    echo "options:"
    echo "w  Weights of the model, could be model id in huggingface or local path"
    echo "n  Number of HPU to use, [1-8], default=1"
    echo "m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None"
    echo "u  URL of the server, str, default=127.0.0.1"
    echo "p  Port number for the server, int, default=30001"
    echo "d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'"
    echo "i  Input range, str, format='input_min,input_max', default='4,1024'"
    echo "o  Output range, str, format='output_min,output_max', default='4,2048'"
    echo "t  max_num_batched_tokens for vllm, int, default=8192"
    echo "l  max_model_len for vllm, int, default=4096"
    echo "b  max_num_seqs for vllm, int, default=128"
    echo "e  number of scheduler steps, int, default=1"
    echo "c  Cache HPU recipe to the specified path, str, default=None"
    echo "s  Skip warmup or not, bool, default=false"
    echo "f  Enable profiling or not, bool, default=false"
    echo "z  Disable zero-padding, bool, default=false"
    echo "a  Disable FusedFSDPA, bool, default=false"
    echo "h  Help info"
    echo
}

model_path=""
num_hpu=1
module_ids=None
host=127.0.0.1
port=30001
dtype=bfloat16
input_range=(4 1024)
output_range=(4 2048)
max_num_batched_tokens=8192
max_model_len=4096
max_num_seqs=128
cache_path=""
skip_warmup=false
profile=false
disable_zero_padding=false
disable_fsdpa=false

block_size=128
scheduler_steps=1

# Get the options
while getopts hw:n:m:u:p:d:i:o:t:l:b:e:c:sfza flag; do
    case $flag in
    h) # display Help
        Help
        exit
        ;;
    w) # get model path
        model_path=$OPTARG ;;
    n) # get number of HPUs
        num_hpu=$OPTARG ;;
    m) # get module ids to use
        module_ids=$OPTARG ;;
    u) # get the URL of the server
        host=$OPTARG ;;
    p) # get the port of the server
        port=$OPTARG ;;
    d) # get data type
        dtype=$OPTARG ;;
    i) # input range
        IFS="," read -r -a input_range <<< "$OPTARG" ;;
    o) # output range
        IFS="," read -r -a output_range <<< "$OPTARG" ;;
    t) # max-num-batched-tokens
        max_num_batched_tokens=$OPTARG ;;
    l) # max-model-len
        max_model_len=$OPTARG ;;
    b) # batch size
        max_num_seqs=$OPTARG ;;
    e) # number of scheduler steps
        scheduler_steps=$OPTARG ;;
    c) # use_recipe_cache
        cache_path=$OPTARG ;;
    s) # skip_warmup
        skip_warmup=true ;;
    f) # enable profiling
        profile=true ;;
    z) # disable zero-padding
        disable_zero_padding=true ;;
    a) # disable FusedSDPA
        disable_fsdpa=true ;;
    \?) # Invalid option
        echo "Error: Invalid option"
        Help
        exit
        ;;
    esac
done

if [ "$model_path" = "" ]; then
    echo "[ERROR]: No model specified. Usage:"
    Help
    exit
fi

model_name=$(basename "$model_path")
model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')

input_min=${input_range[0]}
input_max=${input_range[1]}
output_min=${output_range[0]}
output_max=${output_range[1]}

if [ "$input_min" == "$input_max" ]; then
    disable_zero_padding=true
fi


echo "Starting vllm server for ${model_name} from ${model_path} with input_range=[${input_min}, ${input_max}], output_range=[${output_min}, ${output_max}], max_num_seqs=${max_num_seqs}, max_num_batched_tokens=${max_num_batched_tokens}, max_model_len=${max_model_len} using ${num_hpu} HPUs with module_ids=${module_ids}"

case_name=serve_${model_name}_${dtype}_${device}_in${input_min}-${input_max}_out${output_min}-${output_max}_bs${max_num_seqs}_tp${num_hpu}_steps${scheduler_steps}_$(date +%F-%H-%M-%S)

set_config

${NUMA_CTL} \
python3 -m vllm.entrypoints.openai.api_server \
    --host "${host}" --port "${port}" \
    --device hpu \
    --dtype "${dtype}" \
    "${QUANT_ARGS[@]}" \
    --model "${model_path}" \
    --trust-remote-code \
    --tensor-parallel-size "${num_hpu}" \
    ${ENABLE_EXPERT_PARALLEL} \
    --seed 0 \
    --block-size "${block_size}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-model-len "${max_model_len}" \
    --max-seq-len-to-capture "${VLLM_MAX_SEQ_LEN_TO_CAPTURE}" \
    --disable-log-requests \
    --use-v2-block-manager \
    --use-padding-aware-scheduling \
    --num-scheduler-steps "${scheduler_steps}" \
    --distributed_executor_backend "${dist_backend}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    |& tee "${case_name}".log 2>&1
