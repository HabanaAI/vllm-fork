# set -ex

usage() {
    echo "Usage: $0"
    echo "Options:"
    echo " --model|-m                       Model path lub model stub"
    echo " --image|-i                       run with image-embed prompts"
    echo " --video                          URL of the video file"
    echo " --iter                           number of iterations(Default:1)"
    echo " --num_prompts                    number of batches of prompt"
    echo " --multiple_prompts               run 4x image in a single prompt (bs=4)"
    echo " --image_width/--image_height     width/height of image"
    echo " --video                          run with inputs"
    echo " --text-only                      run with only text inputs"
    echo " --mix_prompt_lenght              run with randomized prompt lenght in each iter (--iter)"
    echo " --shuffle_images                 randomly shuffle the source images"
    echo " --mix_prompts                    run both image and text modality together"
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
    --num_prompts)
        NumPrompts=$2
        shift 2
        ;;
    --image | -i)
        ImageInput="On"
        shift 1
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
    --mix_prompt_lenght)
        MixPromptLenght="On"
        shift 1
        ;;
    --shuffle_images)
        ShuffleImages="On"
        shift 1
        ;;
    --iter)
        iter=$2
        shift 2
        ;;
    --two_images_prompt)
        TwoImagesPrompt="on"
        shift 1
        ;;
    --help)
        usage
        ;;
    esac
done

#Set Default values
iter=${iter:-1}
ImageInput=${ImageInput:-"on"}
ImageWidth=${ImageWidth:-"1200"}
ImageHeight=${ImageHeight:-"600"}
video=${video}

if [[ -n $HELP ]]; then
    usage
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
    EXTRAARGS="$EXTRAARGS --multiple_prompts"
fi

if [[ -n "$NumPrompts" ]]; then
    EXTRAARGS="$EXTRAARGS --prompts $NumPrompts"
fi

if [[ -n "$TwoImagesPrompt" ]]; then
    EXTRAARGS="$EXTRAARGS --two_images_prompt --limit_mm_image 2"
fi

if [[ -n "$MixPromptLenght" ]]; then
    EXTRAARGS="$EXTRAARGS --mix_prompt_lenght"
fi

if [[ -n "$ShuffleImages" ]]; then
    EXTRAARGS="$EXTRAARGS --shuffle_images"
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
    ARGS="-m $model -i --iter $iter $EXTRAARGS"
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
