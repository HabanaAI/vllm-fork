usage() {
    echo "Usage: $0"
    echo "Options:"
    echo " --model                          Model path lub model stub"
    echo " --skip_warmup                    Skip VLLM warmups"
    echo " --gpu                            Run for nvidia GPU"
    echo " --hpu                            Run for Gaudi HPU"
    echo " --dataset                        comma separate img-txt datasets, i.e. --dataset lmarena-ai/vision-arena-bench-v0.1,LIME-DATA/infovqa,sonnet"
    echo " --save_generated-text            save the generated text by vllm"
    echo " --tp                             tensor-parallel size"
}

# Cleanup function
cleanup() {
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
}

# a function that waits on vLLM server to start
wait_for_server() {
    echo "INFO: Waiting for the sever"
    local port=$1
    timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

check_device() {
    device=$1
    if [[ $device == "gpu" ]]; then
        nvidia-smi >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "failed to executed nvidia-smi on this machine. exiting"
            exit
        fi
    elif [[ $device == "hpu" ]]; then
        hl-smi >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "failed to executed hl-smi on this machine. exiting"
            exit
        fi
    else
        echo "unknown device ${device}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --model | -m)
        model=$2
        shift 2
        ;;
    --gpu)
        GPU=true
        shift 1
        ;;
    --hpu)
        HPU=true
        shift 1
        ;;
    --skip_warmup)
        SkipWarmup="on"
        shift 1
        ;;
    --skip_server)
        SkipServer=true
        shift 1
        ;;
    --gpu_mem_usage | -gmu)
        GMU=$2
        shift 2
        ;;
    --dataset | -ds)
        DataSet=$2
        shift 2
        ;;
    --max_model_len | -mml)
        MML=$2
        shift 2
        ;;
    --tensor_parallel_size | -tp)
        TP=$2
        shift 2
        ;;
    --num-prompts | -np)
        NumPrompt=$2
        shift 2
        ;;
    --use_vllm_v1)
        VLLMV1="On"
        shift 1
        ;;
    --save_generated-text | -sgt)
        SaveGenratedText=true
        shift 1
        ;;
    --fp8)
        FP8="On"
        shift 1
        ;;
    --iter)
        ITER=$2
        shift 2
        ;;
    --help)
        usage
        ;;
    esac
done

# Setting the defaults
model=${model:-"Qwen/Qwen2.5-VL-3B-Instruct"}
GPU=${GPU:-false}
HPU=${HPU:-false}
GMU=${GMU:-"0.9"}
MML=${MML:-"32768"}
NumPrompt=${NumPrompt:-"100"}
TP=${TP:-"1"}
ITER=${ITER:-"1"}

# DataSet=${DataSet:-"lmarena-ai/vision-arena-bench-v0.1,LIME-DATA/infovqa,echo840/OCRBench"}
IFS=',' read -r -a DSArray <<<"$DataSet"
VLLM_DIR=$(realpath .)
OUTPUT_DIR=$VLLM_DIR/online.$(hostname).$(date -u +%Y%m%d%H%M)

mkdir -p ${OUTPUT_DIR}

if $HPU && $GPU; then
    echo "ERR: please specify --gpu or --hpu (not both)..exiting"
    exit 1
elif ! $HPU && ! $GPU; then
    echo "ERR: please specifiy at least one --gpu or --hpu. exiting"
    exit 1
fi

if $GPU; then
    uv pip install datasets pandas -q
    check_device "gpu"
fi

if $HPU; then
    pip install datasets pandas -q
    check_device "hpu"
fi

if [[ -n "$VLLMV1" ]]; then
    echo "INFO: vllm V1 is enabled"
    export VLLM_USE_V1=1
else
    echo "INFO: vllm V1 is disabled"
    export VLLM_USE_V1=0
fi

if $HPU; then
    DriverVer=$(hl-smi -Q driver_version -f csv | tail -n 1 | cut -d"-" -f1)
    if (($(echo "$(echo $DriverVer | cut -d"." -f1-2 | bc) >= 1.21" | bc -l))); then
        echo "INFO: DriverVer $DriverVer appears on/ahead  1.21. Setting lazy mode via PT_HPU_LAZY_MODE=1."
        export PT_HPU_LAZY_MODE=1
    fi
fi

if [[ -n "$SkipWarmup" ]]; then
    export VLLM_SKIP_WARMUP=true
else
    export VLLM_SKIP_WARMUP=false
    export VLLM_PROMPT_SEQ_BUCKET_MIN=256
    export VLLM_PROMPT_SEQ_BUCKET_MAX=2048
    export VLLM_PROMPT_BS_BUCKET_MAX=1
    export VLLM_DECODE_BS_BUCKET_MIN=32
    export VLLM_DECODE_BS_BUCKET_STEP=32
    export VLLM_DECODE_BS_BUCKET_MAX=32
    export VLLM_DECODE_BLOCK_BUCKET_MAX=2048
    export VLLM_DECODE_BLOCK_BUCKET_STEP=2048
fi

if [[ "$model" == *"Qwen2"* ]]; then
    #export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false; unset VLLM_QWEN_SPLIT_GRAPHS
    export VLLM_QWEN_SPLIT_GRAPHS=true
    unset PT_HPUGRAPH_DISABLE_TENSOR_CACHE
fi

set_intel_proxy() {
    echo "-------------------------------------"
    echo "Warnning: Set Intel Proxies (if needed) before proceeding"
    echo "-------------------------------------"
    sleep 5
}

set_intel_proxy

BenchSaveTextArg=" "
if [[ -n "$SaveGenratedText" ]]; then
    BenchSaveTextArg=" --save-result --result-dir ${OUTPUT_DIR}"
fi

ServerFP8Args=" "
if [[ -n "$FP8" ]]; then
    if [[ -z "${QUANT_CONFIG}" ]]; then
        echo "ERRO: QUANT_CONFIG env var is not set. please set the proper quant config env"
        echo "INFO: check under ./calibration/ folder for more info"
        exit 1
    fi
    if [[ -z $(pip list | grep optimum-habana) ]]; then
        echo "INFO: running in FP8 mode requires OH installation and quantizaiton files...installing dependecies now"
        git clone https://github.com/huggingface/optimum-habana.git /root/optimum-habana
        cd /root/optimum-habana
        git checkout v1.17.0
        pip install . -q
        # need to reinstall the transformers for vllm
        cd $VLLM_DIR/
        pip install git+https://github.com/malkomes/transformers.git@ac372cd18f836c41f57cdce46094db00019d4280
    fi
    ServerFP8Args=" --quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu"
fi

if [[ ! -n "$SkipServer" ]]; then
    echo "INFO: Lunching the server"
    logfile=${OUTPUT_DIR}/server.log
    cmd="python -m vllm.entrypoints.openai.api_server --port 8080 --model $model --tensor-parallel-size $TP --max-num-seqs 128 --dtype bfloat16 --gpu-memory-util $GMU --max-num-batched-tokens 32768 --max-model-len $MML --block-size 128 ${ServerFP8Args} &"
    echo $cmd && echo $cmd >>$logfile
    eval $cmd 2>&1 | tee -a $logfile &
else
    echo "WARN: Server lunch was skipped!"
fi

wait_for_server 8080

# for GPU we don't want to use the vllm-fork code path
if $GPU; then
    cd /tmp
    ln -sf $VLLM_DIR/benchmarks/ .
fi

cd benchmarks/

if [ ${#DSArray[@]} -gt 0 ]; then
    for item in "${DSArray[@]}"; do
        echo "INFO: Running for dataset: $item"
        # test for the img-and-text
        if [[ $item == "echo840/OCRBench" ]]; then
            Split=test
            Type=random
        elif [[ $item == "lmarena-ai/vision-arena-bench-v0.1" ]]; then
            Split=train
            Type=hf
        elif [[ $item == "LIME-DATA/infovqa" ]]; then
            Split=train
            Type=random
        elif [[ $item == "sonnet" ]]; then
            echo "INFO: $item dataset is text-only"
        else
            echo "ERR: unknown dataset: $item"
            continue
        fi

        if [[ $item == "echo840/OCRBench" ]] || [[ $item == "lmarena-ai/vision-arena-bench-v0.1" ]] || [[ $item == "LIME-DATA/infovqa" ]]; then
            cmd="python benchmark_serving.py --backend openai-chat --model $model --trust-remote-code --port 8080 --endpoint /v1/chat/completions \
            --dataset-path $item --dataset-name $Type --hf-split $Split --num-prompts $NumPrompt --request-rate inf --seed 0 --ignore_eos ${BenchSaveTextArg}"
        elif [[ $item == "sonnet" ]]; then
            cmd="python benchmark_serving.py --backend openai-chat --model $model --trust-remote-code --port 8080 --endpoint /v1/chat/completions \
            --dataset-name $item --dataset-path ./sonnet.txt  --sonnet-input-len 2048 --sonnet-output-len 128 --sonnet-prefix-len 100 \
            --num-prompts $NumPrompt --request-rate inf --seed 0 --ignore_eos ${BenchSaveTextArg}"
        fi

        # running over the specified iterations
        for j in $(seq $ITER); do
            logfile=${OUTPUT_DIR}/$(echo $item | cut -d"/" -f2).iter$j.log
            echo $cmd && echo $cmd >>$logfile
            eval $cmd 2>&1 | tee -a $logfile
        done
    done
else
    echo "WARN: argument dataset is empty! no dataset test is conducted"
fi

## collect a summary of results
PerfArray=("Request throughput (req/s)" "Output token throughput (tok/s)" "Total Token throughput (tok/s)")
temp_file=$(mktemp)
for item in "${PerfArray[@]}"; do grep "$item" ${OUTPUT_DIR}/*log >>$temp_file; done
mv $temp_file ${OUTPUT_DIR}/summary.log

# cleanup after done
cleanup
