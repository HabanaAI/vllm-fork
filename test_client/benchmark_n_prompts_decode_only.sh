model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
ip_addr=127.0.0.1
port=8868

export PT_HPU_LAZY_MODE=1

export no_proxy=127.0.0.1

mkdir -p benchmark_log

#cd /workspace/vllm-fork
#source pd_xpyd/xpyd_start_proxy.sh 1 2 1 benchmark_decode &
#sleep 10

run_benchmark() {
    local local_input=$1
    local local_output=$2
    local local_max_concurrency=$3
    local local_num_prompts=$4
    local local_backend=$5
    local endpoint=$6
    local request_rate=inf

    if [[ "$local_backend" == "openai-chat" ]]; then
        PREFILL_ENDPOINT='/v1/prefill/chat/completions'
        DECODE_ENDPOINT='/v1/decode/chat/completions'
    else
        PREFILL_ENDPOINT='/v1/prefill/completions'
        DECODE_ENDPOINT='/v1/decode/completions'
    fi

    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, concurrency: $local_max_concurrency, prompt_num: $local_num_prompts, endpoint: $endpoint, backend: $local_backend"

    if [ "$endpoint" = "$PREFILL_ENDPOINT" ]; then
        log_name=benchmark_serving_DeepSeek-R1_cardnumber_1p2d_datatype_fp8_sonnet_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_rate_${request_rate}_prompts_${local_num_prompts}_endpoint_prefill_backend_${local_backend}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    else
        log_name=benchmark_serving_DeepSeek-R1_cardnumber_1p2d_datatype_fp8_sonnet_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_rate_${request_rate}_prompts_${local_num_prompts}_endpoint_decode_backend_${local_backend}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    fi

    python3 ../benchmarks/benchmark_serving.py --backend $local_backend --model $model_path --trust-remote-code --host $ip_addr --port $port --endpoint $endpoint --dataset-name sonnet --dataset-path ../benchmarks/sonnet.txt --sonnet-input-len $local_input --sonnet-output-len $local_output --sonnet-prefix-len 100 --max_concurrency $local_max_concurrency --num-prompts $local_num_prompts --request-rate $request_rate --seed 0 --ignore_eos --burstiness 1000 --save-result --result-filename ${log_name}.json |& tee ${log_name}.log > /dev/null
    mv benchmark_serving_DeepSeek* benchmark_log
}

run_decode_only_benchmark() {
    local local_input_len=$1
    local local_output_len=$2
    local local_num_concurrency=$3
    local local_num_prompts=$(( local_num_concurrency * 5 ))
    local local_backend=$4

    if [[ "$local_backend" == "openai-chat" ]]; then
        PREFILL_ENDPOINT='/v1/prefill/chat/completions'
        DECODE_ENDPOINT='/v1/decode/chat/completions'
    else
        PREFILL_ENDPOINT='/v1/prefill/completions'
        DECODE_ENDPOINT='/v1/decode/completions'
    fi

    # Call the function with the provided output length
    echo "Start Prefill Run - Input: $local_input_len, Output: $local_output_len (Actual: 1), Concurrency: $local_num_concurrency, Prompts: $local_num_prompts, Backend: $local_backend"
    run_benchmark $local_input_len 1 $local_num_concurrency $local_num_prompts $local_backend $PREFILL_ENDPOINT

    echo "Start Decode Run - Input: $local_input_len, Output: $local_output_len, Concurrency: $local_num_concurrency, Prompts: $local_num_prompts, Backend: $local_backend"
    run_benchmark $local_input_len $local_output_len $local_num_concurrency $local_num_prompts $local_backend $DECODE_ENDPOINT
}

run_decode_only_benchmark 5000 1300 64 vllm 
run_decode_only_benchmark 5000 1300 80 vllm
run_decode_only_benchmark 5000 1300 96 vllm
run_decode_only_benchmark 5000 1300 384 vllm
run_decode_only_benchmark 5000 1300 448 vllm
run_decode_only_benchmark 5000 1300 512 vllm

run_decode_only_benchmark 5000 1300 64 openai-chat
run_decode_only_benchmark 5000 1300 80 openai-chat
run_decode_only_benchmark 5000 1300 96 openai-chat
run_decode_only_benchmark 5000 1300 384 openai-chat
run_decode_only_benchmark 5000 1300 448 openai-chat
run_decode_only_benchmark 5000 1300 512 openai-chat

run_decode_only_benchmark 2048 1024 64 vllm
run_decode_only_benchmark 2048 1024 96 vllm
run_decode_only_benchmark 2048 1024 624 vllm
run_decode_only_benchmark 2048 1024 672 vllm
run_decode_only_benchmark 2048 1024 1024 vllm

run_decode_only_benchmark 2048 1024 64 openai-chat
run_decode_only_benchmark 2048 1024 96 openai-chat
run_decode_only_benchmark 2048 1024 624 openai-chat
run_decode_only_benchmark 2048 1024 672 openai-chat
run_decode_only_benchmark 2048 1024 1024 openai-chat

