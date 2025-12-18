model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
ip_addr=127.0.0.1
port=8868

export PT_HPU_LAZY_MODE=1

export no_proxy=127.0.0.1

mkdir -p benchmark_log

PREFILL_ENDPOINT='/v1/prefill/completions'
DECODE_ENDPOINT='/v1/decode/completions'

run_benchmark() {
    local local_input=$1
    local local_output=$2
    local local_max_concurrency=$3
    local local_num_prompts=$3
    local endpoint=$4
    local request_rate=inf

    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, concurrency: $local_max_concurrency, prompt_num: $local_num_prompts, endpoint: $endpoint"

    if [ "$endpoint" = "$PREFILL_ENDPOINT" ]; then
	 log_name=benchmark_serving_DeepSeek-R1_cardnumber_1p2d_datatype_bfloat16_sonnet_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_rate_${request_rate}_prompts_${local_num_prompts}_endpoint_prefill_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    else
       log_name=benchmark_serving_DeepSeek-R1_cardnumber_1p2d_datatype_bfloat16_sonnet_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_rate_${request_rate}_prompts_${local_num_prompts}_endpoint_decode_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    fi

    python3 ../benchmarks/benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port --endpoint $endpoint --dataset-name sonnet --dataset-path ../benchmarks/sonnet.txt --sonnet-input-len $local_input --sonnet-output-len $local_output --sonnet-prefix-len 100 --max_concurrency $local_max_concurrency --num-prompts $local_num_prompts --request-rate $request_rate --seed 0 --ignore_eos --burstiness 1000 --save-result --result-filename ${log_name}.json |& tee ${log_name}.log > /dev/null
    mv benchmark_serving_DeepSeek* benchmark_log
}

run_decode_only_benchmark() {
    local local_input_len=$1
    local local_output_len=$2
    local local_num_concurrency=$3

    # Call the function with the provided output length
    echo "Start Prefill Run - Input: $local_input_len, Output: $local_output_len (Actual: 1), Concurrency: $local_num_concurrency"
    run_benchmark $local_input_len 1 $local_num_concurrency $PREFILL_ENDPOINT

    echo "Start Decode Run - Input: $local_input_len, Output: $local_output_len, Concurrency: $local_num_concurrency"
    run_benchmark $local_input_len $local_output_len $local_num_concurrency $DECODE_ENDPOINT
}

run_decode_only_benchmark 5000 1300 64 
run_decode_only_benchmark 5000 1300 80 
run_decode_only_benchmark 5000 1300 96 
run_decode_only_benchmark 5000 1300 384 
run_decode_only_benchmark 5000 1300 448 
run_decode_only_benchmark 5000 1300 512
