# set -ex
usage() {
    echo "Usage: $0"
    echo "Options:"
    echo " --model|-m                       Model path lub model stub"
    echo " --image|-i                       run with image-embed prompts"
    echo " --video                          URL of the video file"
    echo " --iter                           number of iterations(Default:1)"
    echo " --multiple_prompts               run 4x image in a single prompt (bs=4)"
    echo " --image_width/--image_height     width/height of image"
    echo " --video                          run with inputs"
    echo " --text-only                      run with only text inputs"
    echo " --mix_prompt_lenght              run with randomized prompt lenght in each iter (--iter)"
    echo " --shuffle_images                 randomly shuffle the source images"
    echo " --mix_prompts                    run both image and text modality together"
    echo " --max_num_seq                    setting the max_num_seq for vllm engine"
    echo " --awq                            run with hpu-awq quantization"
    echo " --inc                            run with inc fp8 quantization"
    echo " --tp                             mi,ber pf tensor parallel size"
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
    --image | -i)
        ImageInput="On"
        shift 1
        ;;
    --num_prompts)
        NumPrompts=$2
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
    --use_vllm_v1)
        VLLMV1="On"
        shift 1
        ;;
    --max_num_seq | -mns)
        MaxNumSeq=$2
        shift 2
        ;;
    --awq)
        AWQ="On"
        shift 1
        ;;
    --inc)
        INC="On"
        shift 1
        ;;
    --tp)
        TP=$2
        shift 2
        ;;
    --help)
        usage
        ;;
    *)
        echo "unknown arg $i"
        exit
        ;;
    esac
done

#Set Default values
iter=${iter:-1}
ImageInput=${ImageInput:-"on"}
MaxNumSeq=${MaxNumSeq:-"5"}
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

if [[ -n "$VLLMV1" ]]; then
    echo "INFO: vllm V1 is enabled"
    export VLLM_USE_V1=1
else
    echo "INFO: vllm V1 is disabled"
    export VLLM_USE_V1=0
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

DriverVer=$(hl-smi -Q driver_version -f csv | tail -n 1 | cut -d"-" -f1)
if (($(echo "$(echo $DriverVer | cut -d"." -f1-2 | bc) >= 1.21" | bc -l))); then
    echo "INFO: DriverVer $DriverVer appears on/ahead  1.21. Setting lazy mode via PT_HPU_LAZY_MODE=1."
    export PT_HPU_LAZY_MODE=1
fi

if [[ -n "$SKIPWARMUP" ]]; then
    export VLLM_SKIP_WARMUP=true
else
    export VLLM_SKIP_WARMUP=false
    export VLLM_PROMPT_SEQ_BUCKET_MIN=256
    export VLLM_PROMPT_SEQ_BUCKET_MAX=2048
    export VLLM_PROMPT_BS_BUCKET_MAX=4
    export VLLM_PROMPT_BS_BUCKET_MIN=1
    export VLLM_PROMPT_BS_BUCKET_STEP=2
    export VLLM_DECODE_BS_BUCKET_MIN=1
    export VLLM_DECODE_BS_BUCKET_STEP=2
    export VLLM_DECODE_BS_BUCKET_MAX=4
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
    #export VLLM_QWEN_SPLIT_GRAPHS=true
    unset PT_HPUGRAPH_DISABLE_TENSOR_CACHE
    #export VLLM_FP32_SOFTMAX=true # for accuracy
    #export VLLM_FP32_SOFTMAX_VISION=true # for accuracy, will have perf drop for prefill
fi

# to increase the perf.
#export VLLM_DELAYED_SAMPLING=true


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
if [[ -n "$AWQ" ]]; then ARGS="$ARGS --awq"; fi
if [[ -n "$INC" ]]; then ARGS="$ARGS --inc"; fi
if [[ -n "$TP" ]]; then ARGS="$ARGS --tp $TP"; fi
ARGS="$ARGS --max_num_seq $MaxNumSeq"
cmd="python offline_inferece.py $ARGS"
echo $cmd && eval $cmd
