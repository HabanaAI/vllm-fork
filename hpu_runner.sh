ModelName=Qwen/Qwen2.5-VL-3B-Instruct
logfile=g2-1x-pr1163-3698d03fb_1.21-555.offline-pytest-bf16online-fp8inc_online-NoCustomTransformers.log

echo_exe() {
    echo "$1"
    echo "$1" >>"$2"
    eval "$1" 2>&1 | tee -a "$2"
}

if true; then
    cmd="bash run_offline.sh -m Qwen/Qwen2.5-VL-3B-Instruct -i --random_image_size --iter 5 --image_width 1120 --image_height 1120"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m Qwen/Qwen2.5-VL-3B-Instruct -i --random_image_size --iter 5 --image_width 1120 --image_height 1120 --multiple_prompts"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m Qwen/Qwen2.5-VL-3B-Instruct -i --random_image_size --iter 5 --image_width 224 --image_height 224 --multiple_prompts"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m Qwen/Qwen2.5-VL-3B-Instruct -i --random_image_size --iter 5 --image_width 1120 --image_height 1120 --mix_prompt_lenght"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m Qwen/Qwen2.5-VL-3B-Instruct -i --random_image_size --iter 5 --image_width 224 --image_height 224 --mix_prompt_lenght"
    echo_exe "$cmd" "$logfile"

    cmd="bash run_offline.sh -m  $ModelName -i --iter 3 --image_width 224 --image_height 224 --skip_warmup"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i --iter 3 --image_width 1008 --image_height 1008 --skip_warmup"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i --iter 3 --image_width 1120 --image_height 1120 --skip_warmup"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i  --iter 5  --skip_warmup --num_prompts 10 --multiple_prompts  --image_width 224 --image_height 224 "
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i  --iter 5  --skip_warmup --num_prompts 10 --multiple_prompts  --image_width 1008 --image_height 1008"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i  --iter 5  --skip_warmup --num_prompts 10 --multiple_prompts  --image_width 1120  --image_height 1120"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i  --iter 5  --skip_warmup --num_prompts 20 --multiple_prompts  --image_width 224 --image_height 224"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i  --iter 5  --skip_warmup --num_prompts 20 --multiple_prompts  --image_width 1008 --image_height 1008"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i  --iter 5  --skip_warmup --num_prompts 20 --multiple_prompts  --image_width 1120 --image_height 1120"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i  --iter 5  --skip_warmup --num_prompts 10 --multiple_prompts"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_offline.sh -m  $ModelName -i  --iter 5  --skip_warmup --num_prompts 40 --multiple_prompts  --image_width 1008 --image_height 1008"
    echo_exe "$cmd" "$logfile"

    cmd="VLLM_SKIP_WARMUP=true pytest tests/models/multimodal/generation/test_common.py -s -v -k qwen2_5_vl"
    echo_exe "$cmd" "$logfile"
fi
if true; then
    source ~/helpers/env.sh
    set_intel_proxy
    cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 10"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 100"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 200"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 300"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 400"
    echo_exe "$cmd" "$logfile"

    cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 500"
    echo_exe "$cmd" "$logfile"
    cmd="bash run_online.sh -m $ModelName --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 200"
    echo_exe "$cmd" "$logfile"
    #cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 100 -tp 2"
    #echo_exe "$cmd" "$logfile"
    #cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 100 -tp 4"
    #echo_exe "$cmd" "$logfile"
    cmd="bash run_online.sh -m $ModelName --skip_warmup --hpu -ds sonnet --num-prompts 1000"
    echo_exe "$cmd" "$logfile"

    cmd="QUANT_CONFIG=/root/vllm-fork/calibration/vllm-hpu-extension/calibration/g2/qwen2.5-vl-3b-instruct/maxabs_quant_g2.json bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 100 --fp8"
    echo_exe "$cmd" "$logfile"
    cmd="QUANT_CONFIG=/root/vllm-fork/calibration/vllm-hpu-extension/calibration/g2/qwen2.5-vl-3b-instruct/maxabs_quant_g2.json bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 200 --fp8"
    echo_exe "$cmd" "$logfile"
    cmd="QUANT_CONFIG=/root/vllm-fork/calibration/vllm-hpu-extension/calibration/g2/qwen2.5-vl-3b-instruct/maxabs_quant_g2.json bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 300 --fp8"
    echo_exe "$cmd" "$logfile"
    cmd="QUANT_CONFIG=/root/vllm-fork/calibration/vllm-hpu-extension/calibration/g2/qwen2.5-vl-3b-instruct/maxabs_quant_g2.json bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1 --num-prompts 400 --fp8"
    echo_exe "$cmd" "$logfile"
    cmd="QUANT_CONFIG=/root/vllm-fork/calibration/vllm-hpu-extension/calibration/g2/qwen2.5-vl-3b-instruct/maxabs_quant_g2.json bash run_online.sh -m $ModelName --skip_warmup --hpu -ds sonnet --num-prompts 1000 --fp8"
    echo_exe "$cmd" "$logfile"
fi
