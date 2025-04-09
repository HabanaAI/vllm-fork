usage() {
    echo "Usage: $0"
    echo "Options:"
    echo " --model                          Model path lub model stub"
    echo " --skip_warmup                    Skip VLLM warmups"
    echo " --gpu                            Run for nvidia GPU"
    echo " --hpu                            Run for Gaudi HPU"
    echo " --dataset                        comma separate img-txt datasets, i.e. --dataset lmarena-ai/vision-arena-bench-v0.1,LIME-DATA/infovqa,"
    echo " --run_textonly                   Run the sonet text-only case as well"
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
    --num-prompts | -np)
        NumPrompt=$2
        shift 2
        ;;
    --run_sonet)
        RunSonet=true
        shift 1
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

# DataSet=${DataSet:-"lmarena-ai/vision-arena-bench-v0.1,LIME-DATA/infovqa,echo840/OCRBench"}
IFS=',' read -r -a DSArray <<<"$DataSet"
VLLM_DIR=$(realpath .)

if $HPU && $GPU; then
    echo "ERR: please specify --gpu or --hpu (not both)..exiting"
    exit 1
elif ! $HPU && ! $GPU; then
    echo "ERR: please specifiy at least one --gpu or --hpu. exiting"
    exit 1
fi

if $GPU; then
    # using vllm v0 till further notice
    export VLLM_USE_V1=0
    uv pip install datasets pandas -q
fi

if $HPU; then
    pip install datasets pandas -q
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

export HTTPS_PROXY=http://proxy-dmz.intel.com:912
export HTTP_PROXY=http://proxy-dmz.intel.com:912
export no_proxy=0.0.0.0,localhost,intel.com,.intel.com,10.0.0.0/8,192.168.0.0/16

if [[ ! -n "$SkipServer" ]]; then
    echo "INFO: Lunching the server"
    logfile=online.$(hostname).$(date -u +%Y%m%d%H%M).server.log
    cmd="python -m vllm.entrypoints.openai.api_server --port 8080 --model $model --tensor-parallel-size 1 --max-num-seqs 128 --dtype bfloat16 --gpu-memory-util $GMU --max-num-batched-tokens 32768 --max-model-len $MML --block-size 128 &"
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
        else
            echo "ERR: unknown dataset: $item"
            continue
        fi

        cmd="python benchmark_serving.py --backend openai-chat --model $model --trust-remote-code --port 8080 --endpoint /v1/chat/completions \
        --dataset-path $item --dataset-name $Type --hf-split $Split --num-prompts $NumPrompt --request-rate inf --seed 0 --ignore_eos"
        logfile=online.$(hostname).$(date -u +%Y%m%d%H%M).$(echo $item | cut -d"/" -f2).log
        echo $cmd && echo $cmd >>$logfile
        eval $cmd 2>&1 | tee -a $logfile
    done
else
    echo "WARN: argument dataset is empty! no img-text dataset test is conducted"
fi

if [[ -n "$RunSonet" ]]; then
    # test for the sonet
    cmd="python benchmark_serving.py --backend openai-chat --model $model --trust-remote-code --port 8080 --endpoint /v1/chat/completions --dataset-name sonnet --dataset-path ./sonnet.txt \
        --num-prompts 1000 --port 8080 --sonnet-input-len 2048 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NumPrompt --request-rate inf --seed 0 --ignore_eos"
    logfile=online.$(hostname).$(date -u +%Y%m%d%H%M).sonet.log
    echo $cmd && echo $cmd >>$logfile
    eval $cmd 2>&1 | tee -a $logfile
fi

# cleanup after done
cleanup
