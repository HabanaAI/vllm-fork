#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

Help() {
    # Display Help
    echo "Start a vLLM server for a huggingface model on Gaudi."
    echo
    echo "Usage: bash start_gaudi_vllm_server.sh <-w> [-t:r:m:a:d:q:x:p:n:b:g:k:u:e:l:c:sf] [-h]"
    echo "Options:"
    echo "-w  Weights of the model, str, could be model id in huggingface or local path."
    echo "    DO NOT change the model name as some of the parameters depend on it."
    echo "-t  tensor-parallel-size for vLLM, int, default=1."
    echo "    Also used to set EP size if it's enable by --enable-expert-parallel"
    echo "-r  pipeline-parallel-size for vLLM, int, default=1."
    echo "-m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None"
    echo "    Used to select HPUs and to set NUMA accordingly. It's recommended to set"
    echo "    for cases with 4 or less HPUs."
    echo "-a  API server URL, str, 'IP:PORT', default=127.0.0.1:30001"
    echo "-d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'"
    echo "    Set to 'fp8' if -q or environment variable 'QUANT_CONFIG' is provided."
    echo "-q  Path to the quantization config file, str, default=None"
    echo "    default=$BASH_DIR/quantization/<model_name_lower>/maxabs_quant_g2.json for -d 'fp8'"
    echo "    The environment variable 'QUANT_CONFIG' will override this option."
    echo "-x  max-model-len for vllm, int, default=16384"
    echo "-p  Max number of the prefill sequences, int, default=${PREFERED_PREFILL_BS}"
    echo "    Used to control the max batch size for prefill to balance the TTFT and throughput."
    echo "    The default value of 1 is used to optimize the TTFT."
    echo "    Set to '' to optimize the throughput for short prompts."
    echo "-n  Max number of the mixed sequences, int, default=${PREFERED_PREFILL_BS}"
    echo "    Used to control the max batch size for prefill+decode to balance the TTFT and throughput."
    echo "    The default value of 1 is used to optimize the TTFT."
    echo "    This must be between 'max_num_prefill_seqs' and 'max_num_seqs'."
    echo "-b  max-num-seqs for vLLM, int, default=${PREFERED_DECODING_BS}"
    echo "    Used to control the max batch size for decoding phase."
    echo "    It is recommended to set this value according to the 'Maximum concurrency'"
    echo "    reported by a test run."
    echo "-g  max-seq-len-to-capture for vLLM, int, default=${PREFERED_SEQ_LEN_TO_CAPTURE}"
    echo "    Used to control the maximum batched tokens to be captured in HPUgraph."
    echo "    Reduce this value could decrease memory usage, but not smaller than 2048."
    echo "-k  enable-chunked prefill and cap max-num-batched-tokens"
    echo "    to the specified value, int, default=disabled"
    echo "-u  gpu-memory-utilization, float, default=0.9"
    echo "    Used to control the GPU memory utilization. Reduce this value if OOM occurs."
    echo "-e  Extra vLLM server parameters, str, default=None"
    echo "    Extra parameters that will pass to the vLLM server."
    echo "-l  Limit of the padding ratio, float, [0.0, 0.5], default=0.25"
    echo "    The padding strategy ensures that padding_size/bucket_size <= limit."
    echo "    Smaller limit values result in more aggressive padding and more buckets."
    echo "    Set to 0.5 is equivalent to use the exponential bucketing."
    echo "    Set to 0.0 is equivalent to use the linear bucketing without padding limits."
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
while getopts hw:t:r:m:a:d:q:x:p:n:b:g:k:u:e:l:c:sf flag; do
    case $flag in
    h) # display Help
        Help
        exit
        ;;
    w) # get model path
        weights_path=$OPTARG ;;
    t) # get number of TP HPUs
        num_tp_hpu=$OPTARG ;;
    r) # get number of PP HPUs
        num_pp_hpu=$OPTARG ;;
    m) # get module ids to use
        module_ids=$OPTARG ;;
    a) # get the URL of the server
        host=${OPTARG%%:*}
        port=${OPTARG##*:}
        ;;
    d) # get data type
        dtype=$OPTARG ;;
    q) # get quantization config
        quant_config=$OPTARG ;;
    x) # max model length
        max_model_len=$OPTARG ;;
    p) # max number of prefill sequences
        max_num_prefill_seqs=$OPTARG ;;
    n) # max number of mixed sequences
        max_num_mixed_seqs=$OPTARG ;;
    b) # batch size
        max_num_seqs=$OPTARG ;;
    g) # max-seq-len-to-capture
        max_seq_len_to_capture=$OPTARG ;;
    k) # max-num-batched-tokens for chunked prefill
        chunk_size=$OPTARG ;;
    u) # gpu-memory-utilization
        gpu_memory_utilization=$OPTARG ;;
    e) # extra vLLM server parameters
        IFS=" " read -r -a extra_params <<< "$OPTARG" ;;
    l) # limit of the padding ratio
        max_padding_ratio=$OPTARG
        # make sure max_padding_ratio is a float and in [0.0, 0.5]
        if ! [[ "$max_padding_ratio" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "[ERROR]: max_padding_ratio should be a float."
            exit 1
        fi
        if (( $(echo "$max_padding_ratio < 0.0" | bc -l) )) || \
            (( $(echo "$max_padding_ratio > 0.5" | bc -l) )); then
            echo "[ERROR]: max_padding_ratio should be in [0.0, 0.5]."
            exit 1
        fi
        ;;
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

if [ -z "$weights_path" ]; then
    echo "[ERROR]: No model specified. Usage:"
    Help
    exit
fi

num_tp_hpu=${num_tp_hpu:-"1"}
num_pp_hpu=${num_pp_hpu:-"1"}
num_hpu=$(( num_tp_hpu * num_pp_hpu ))
module_ids=${module_ids:-"None"}
host=${host:-"127.0.0.1"}
port=${port:-"30001"}
dtype=${dtype:-"bfloat16"}
quant_config=${quant_config:-""}
max_model_len=${max_model_len:-"16384"}
max_num_prefill_seqs=${max_num_prefill_seqs-${PREFERED_PREFILL_BS}}
max_num_mixed_seqs=${max_num_mixed_seqs-${PREFERED_PREFILL_BS}}
max_num_seqs=${max_num_seqs:-$PREFERED_DECODING_BS}
max_seq_len_to_capture=${max_seq_len_to_capture:-$PREFERED_SEQ_LEN_TO_CAPTURE}
chunk_size=${chunk_size:-""}
gpu_memory_utilization=${gpu_memory_utilization:-"0.9"}
extra_params=(${extra_params[@]:-})
max_padding_ratio=${max_padding_ratio:-"0.25"}
cache_path=${cache_path:-""}
skip_warmup=${skip_warmup:-"false"}
profile=${profile:-"false"}

# check if the port is occupied. If occupied, notify user and exit
if command -v ss >/dev/null 2>&1; then
    CHECK=$(ss -tlnp | grep ":$port ")
    if [ ! -z "$CHECK" ]; then
        echo "[ERROR]: The port $port is occupied. Please specify another port with '-a <IP:PORT>'."
        exit 1
    fi
fi

model_name=$(basename "$weights_path")
model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')

# if quant_config or QUANT_CONFIG is provided, set dtype to 'fp8'
if [ "$quant_config" != "" ] || [ "$QUANT_CONFIG" != "" ]; then
    dtype="fp8"
fi

echo "Starting vllm server for ${model_name} from ${weights_path} with:"
echo "    device: ${num_hpu} HPUs with module_ids=${module_ids}"
echo "    TP=${num_tp_hpu} PP=${num_pp_hpu}"
echo "    URL: ${host}:${port}"
echo "    max_num_seqs: ${max_num_seqs}"
echo "    max_model_len: ${max_model_len}"
if [[ "$chunk_size" != "" ]]; then
    echo "    chunked prefill enabled with max-num-batched-tokens: ${chunk_size}"
    # Append enable-chunked-prefill flag to extra_params without disrupting existing values
    extra_params+=("--enable-chunked-prefill")
fi

case_name=serve_${model_name}_${dtype}_${DEVICE_NAME}_len${max_model_len}_bs${max_num_seqs}_ps${max_num_prefill_seqs}
if [[ "$chunk_size" != "" ]]; then
    case_name=${case_name}_ms${max_num_mixed_seqs}_chunk${chunk_size}
fi
case_name=${case_name}_tp${num_tp_hpu}_pp${num_pp_hpu}_$(date +%F-%H-%M-%S)
log_file="${case_name}.log"

set_config

echo "Changed environment variables:" |& tee "${log_file}"
echo -e "${changed_env}\n" |& tee -a "${log_file}"

command_string=$(echo ${NUMA_CTL_CMD} \
python3 -m vllm.entrypoints.openai.api_server \
    --block-size "${BLOCK_SIZE}" \
    --host "${host}" --port "${port}" \
    --model "${weights_path}" \
    --dtype "${DATA_TYPE}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-prefill-seqs "${max_num_prefill_seqs}" \
    --max-num-mixed-seqs "${max_num_mixed_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-seq-len-to-capture "${max_seq_len_to_capture}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --max-model-len "${max_model_len}" \
    --tensor-parallel-size "${num_tp_hpu}" \
    --pipeline-parallel-size "${num_pp_hpu}" \
    --trust-remote-code \
    --seed 2025 \
    --distributed_executor_backend "${dist_backend}" \
    "${extra_params[@]}")

echo "Start a vLLM server for ${model_name} on Gaudi $DEVICE_NAME with command:" |& tee -a "${log_file}"
echo -e "${command_string}\n" |& tee -a "${log_file}"
echo "The log will be saved to ${case_name}.log"

eval "${command_string}" |& tee -a "${case_name}".log 2>&1
