set -ex

usage() {
    echo "Usage: $0"
    echo "Options:"
    echo "  --model|-m               Model path lub model stub"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --model | -m)
        model=$2
        shift 2
        ;;
    --skip_warmup)
        fp8="On"
        shift 1
        ;;
    --download_image)
        Donwload="On"
        shift 1
        ;;
    --install_vllm)
        InstallVLLM="On"
        shift 1
        ;;
    --help)
        usage
        ;;
    esac
done

if [[ -n $HELP ]]; then
    usage
fi

if [[ -n $Donwload ]]; then
    python3 -c "import requests; from PIL import Image; url = 'https://huggingface.co/alkzar90/ppaine-landscape/resolve/main/assets/snowscat-H3oXiq7_bII-unsplash.jpg'; filename = url.split('/')[-1]; r = requests.get(url, allow_redirects=True); open(filename, 
'wb').write(r.content); image = Image.open(f'./{filename}'); image = image.resize((1200, 600)); image.save(f'/tmp/{filename}')"
fi

if [[ -n $InstallVLLM ]]; then
    if [[ "$model" == *"Qwen2"* ]]; then
        git clone https://github.com/HabanaAI/vllm-fork.git -b sarkar/qwen2
    elif [[ "$model" == *"Llama-3.2"* ]]; then
        git clone https://github.com/HabanaAI/vllm-fork.git -b v1.20.0
    fi
    cd vllm-fork
    pip install -r requirements-hpu.txt -q
    python setup.py develop
    cd ..
fi

if [[ -n "$fp8" ]]; then
    export VLLM_SKIP_WARMUP=true
else
    export VLLM_PROMPT_SEQ_BUCKET_MIN=1024
    export VLLM_PROMPT_SEQ_BUCKET_MAX=1024
    export VLLM_PROMPT_BS_BUCKET_MAX=4
    export VLLM_DECODE_BS_BUCKET_MIN=64
    export VLLM_DECODE_BS_BUCKET_STEP=64
    export VLLM_DECODE_BS_BUCKET_MAX=64
    export VLLM_DECODE_BLOCK_BUCKET_MAX=2048
    export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
fi

#export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
#export EXPERIMENTAL_WEIGHT_SHARING=0

if [[ "$model" == *"Qwen2"* ]]; then
    export WORKAROUND=1
fi
python offline_inferece.py -m $model
