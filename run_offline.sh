# set -ex

usage() {
    echo "Usage: $0"
    echo "Options:"
    echo "  --model|-m                    Model path lub model stub"
    echo "  --image_type|-i               Type model: snowscat synthetic"
    echo "  --video                       URL of the video file"
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
    --gpu_mem_usage | -gmu)
        GMU=$2
        shift 2
        ;;
    --image_width)
        ImageWidth=$2
        shift 2
        ;;
    --image_height)
        ImageHeight=$2
        shift 2
        ;;
    --text_only | -t)
        TextOnly="On"
        shift 1
        ;;
    --video | -v)
        if [[ -n "$2" && ! "$2" =~ ^- ]]; then # Assign URL if provided
            video=$2
            shift 2
        else
            video="https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4" # video set to default video URL file
            shift 1
        fi
        ;;
    --random_image_size | -ris)
        RandomImageSize="On"
        shift 1
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
video=${video}

if [[ -n $HELP ]]; then
    usage
fi

if [[ $ImageType == "snowscat" ]] && [[ ! $(md5sum /tmp/snowscat-H3oXiq7_bII-unsplash.jpg | cut -d" " -f1) == "3ad5657658d1a7684e35541778d30204" ]]; then
    python3 -c "import requests; from PIL import Image; url = 'https://huggingface.co/alkzar90/ppaine-landscape/resolve/main/assets/snowscat-H3oXiq7_bII-unsplash.jpg'; filename = f'"'/tmp/{url.split("'"/"'")[-1]}'"'; r = requests.get(url, allow_redirects=True); open(filename,'wb').write(r.content);"
    ImageWidth=${ImageWidth:-"1200"}
    ImageHeight=${ImageHeight:-"600"}
elif [[ $ImageType == "synthetic" ]]; then
    ImageWidth=${ImageWidth:-"250"}
    ImageHeight=${ImageHeight:-"250"}
fi

if [[ -n "$video" ]]; then
    filename=$(basename "$video")
    echo "Downloading Video $filename from $video"
    wget -O /tmp/$filename "$video"
    videofile="/tmp/$filename"
    GMU=${GMU:-"0.6"}
else
    GMU=${GMU:-"0.9"}
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

if false; then
    export VLLM_GRAPH_RESERVED_MEM=0.05 # default 0.1
    export VLLM_GRAPH_PROMPT_RATIO=0.3  # 0.3 default
    export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=true
    export PT_HPU_METRICS_GC_DETAILS=1
    export VLLM_HPU_LOG_STEP_CPU_FALLBACKS=1
fi

if [[ -n "$SKIPWARMUP" ]]; then
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

EXTRAARGS=" "
if [[ -n "$MultiPrompt" ]]; then
    EXTRAARGS=" --multiple_prompts"
fi

if [[ "$model" == *"Qwen2"* ]]; then
    #export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false; unset VLLM_QWEN_SPLIT_GRAPHS
    export VLLM_QWEN_SPLIT_GRAPHS=true
    unset PT_HPUGRAPH_DISABLE_TENSOR_CACHE
fi

if [[ -n "$video" ]]; then
    ARGS="-m $model -v $videofile --iter $iter $EXTRAARGS"
elif [[ -n "$TextOnly" ]]; then
    ARGS="-m $model -t --iter $iter $EXTRAARGS"
else
    ARGS="-m $model -i $ImageType --iter $iter $EXTRAARGS"
    if [ -n "$ImageWidth" ] && [ -n "$ImageHeight" ]; then
        ARGS="$ARGS --image_width $ImageWidth --image_height $ImageHeight"
    fi
    if [ -n "$RandomImageSize" ]; then
        ARGS="$ARGS --random_img_size"
    fi
fi
if [[ -n "$GMU" ]]; then ARGS="$ARGS --gpu_mem_usage $GMU"; fi

cmd="python offline_inferece.py $ARGS"
echo $cmd && eval $cmd
