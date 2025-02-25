set -ex

usage() {
    echo "Usage: $0"
    echo "Options:"
    echo "  --model|-m                    Model path lub model stub"
    echo "  --image_type|-i               Type model: snowscat synthetic"
    echo "  --iter                        number of iterations(Default:1)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --model | -m)
        model=$2
        shift 2
        ;;
    --skip_warmup)
        SKIPWARMUP="On"
        shift 1
        ;;
    --install_vllm)
        InstallVLLM="On"
        shift 1
        ;;
    --multiple_prompts)
        MultiPrompt="On"
        shift 1
        ;;
    --image_type | -i)
        ImageType=$2
        shift 2
        ;;
    --iter)
        iter=$2
        shift 2
        ;;
    --help)
        usage
        ;;
    esac
done

#Set Default values
iter=${iter:-1}
ImageType=${ImageType:-"snowscat"}

if [[ -n $HELP ]]; then
    usage
fi

# VLLM Variables
export VLLM_GRAPH_RESERVED_MEM=0.05 # default 0.1
export VLLM_GRAPH_PROMPT_RATIO=0.3 # 0.3 default
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=256
export VLLM_PROMPT_SEQ_BUCKET_MAX=1024 # suggested max-model-len
export VLLM_DECODE_BLOCK_BUCKET_STEP=32
export VLLM_DECODE_BLOCK_BUCKET_MAX=128
export VLLM_DECODE_BS_BUCKET_MIN=64
export VLLM_DECODE_BS_BUCKET_STEP=64
export VLLM_DECODE_BS_BUCKET_MAX=128


export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=true
export PT_HPU_METRICS_GC_DETAILS=1
export VLLM_HPU_LOG_STEP_CPU_FALLBACKS=1

if [[ $ImageType == "snowscat" ]] && [[ ! $(md5sum /tmp/snowscat-H3oXiq7_bII-unsplash.jpg | cut -d" " -f1) == "3ad5657658d1a7684e35541778d30204" ]]; then
    python3 -c "import requests; from PIL import Image; url = 'https://huggingface.co/alkzar90/ppaine-landscape/resolve/main/assets/snowscat-H3oXiq7_bII-unsplash.jpg'; filename = url.split('/')[-1]; r = requests.get(url, allow_redirects=True); open(filename, 
'wb').write(r.content); image = Image.open(f'./{filename}'); image = image.resize((1200, 600)); image.save(f'/tmp/{filename}')"
fi

if [[ -n $InstallVLLM ]]; then
    git clone https://github.com/HabanaAI/vllm-fork.git -b sarkar/qwen2
    cd vllm-fork
    pip install -r requirements-hpu.txt -q
    python setup.py develop
    if [[ "$model" == *"Qwen2"* ]]; then
        pip install git+https://github.com/huggingface/transformers.git@6b550462139655d488d4c663086a63e98713c6b9
    fi
    cd ..
fi

if [[ -n "$SKIPWARMUP" ]]; then
    export VLLM_SKIP_WARMUP=true
else
    export VLLM_SKIP_WARMUP=false
fi

EXTRAARGS=" "
if [[ -n "$MultiPrompt" ]]; then
    EXTRAARGS=" --multiple_prompts"
fi

if [[ "$model" == *"Qwen2"* ]]; then
    export WORKAROUND=1
fi

ARGS="-m $model -i $ImageType --iter $iter $EXTRAARGS"
python offline_inferece.py $ARGS
