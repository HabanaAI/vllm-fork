#!/bin/bash

# server settings
model="/data/Llama-3.1-70B-Instruct"
model_name="Llama-3.1-70B-Instruct"
qps="inf"
gpu_utils=0.9

#tp_parrallel_list=(2 4 8)
#req_in_out_list=(128_2048 128_4096 500_2000 2048_2048)
max_num_batched_tokens=8192
tp_parrallel_list=(2)
req_in_out_list=(1024_1024 1024_4096 4096_1024 8192_1024)
cons=(1 2 4 8 16 32 64 128 256 512 1024)
for req_in_out in "${req_in_out_list[@]}"; do
    for tp_parrallel in "${tp_parrallel_list[@]}"; do
        for con in "${cons[@]}"; do
            isl=$(echo "$req_in_out" | awk -F'_' '{ print $1 }')
            osl=$(echo "$req_in_out" | awk -F'_' '{ print $2 }')
            # get cons_min and cons_max based on cons
            cons_min=${con}
            cons_max=${con}

            log_name="gaudi3-fp8-${model_name}-${gpu_utils}util-TPparallel${tp_parrallel}-isl${isl}-osl${osl}-con${con}"

            total_len=$((isl + osl))
            block_size=256
            bs_step=128
            block_step=256

            # if total_len is not multiple of block_size, round up to the next multiple of block_size
            total_len=$(((total_len + block_size - 1) / block_size * block_size))
            decode_bs=${cons_max}

            isl_aligned=$(((isl + block_size - 1) / block_size * block_size))
            VLLM_DECODE_BLOCK_BUCKET_MIN=$((isl_aligned * decode_bs / block_size))
            VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len * decode_bs / block_size))
            VLLM_DECODE_BLOCK_BUCKET_MAX=$(((VLLM_DECODE_BLOCK_BUCKET_MAX + block_step - 1) / block_step * block_step))
            prefill_bs=$((max_num_batched_tokens / isl_aligned))

            total_len_aligned=$(((total_len + block_size - 1) / block_size * block_size))
            max_decode_bs=$(((115 * tp_parrallel - 70) * 1024 / 20 * block_size / total_len_aligned))
            echo "max_decode_bs: $max_decode_bs"
            if [[ $max_decode_bs -gt 128 ]]; then
                max_decode_bs=$(((max_decode_bs - 128 + 1) / 128 * 128))
            elif [[ $max_decode_bs -gt 64 ]]; then
                max_decode_bs=$(((max_decode_bs - 64 + 1) / 64 * 64))
            elif [[ $max_decode_bs -gt 32 ]]; then
                max_decode_bs=$(((max_decode_bs - 32 + 1) / 32 * 32))
            fi
            # get max batch_size, if decode_bs is exceeding max_batch_size, set decode_bs to max_batch_size
            if [[ $decode_bs -gt ${max_decode_bs} ]]; then
                decode_bs=${max_decode_bs}
            fi
            
            echo "isl: $isl, osl: $osl, prefill_seqlen: $isl_aligned, prefill_bs: $prefill_bs, decode_bs: $cons_min : $decode_bs, VLLM_DECODE_BLOCK_BUCKET_MIN: $VLLM_DECODE_BLOCK_BUCKET_MIN, VLLM_DECODE_BLOCK_BUCKET_MAX: $VLLM_DECODE_BLOCK_BUCKET_MAX"

            echo "isl: $isl, osl: $osl, prefill_seqlen: $isl_aligned, prefill_bs: $prefill_bs, decode_bs: $cons_min : $decode_bs, VLLM_DECODE_BLOCK_BUCKET_MIN: $VLLM_DECODE_BLOCK_BUCKET_MIN, VLLM_DECODE_BLOCK_BUCKET_MAX: $VLLM_DECODE_BLOCK_BUCKET_MAX" > benchmark_results/${log_name}_serving.log

            #VLLM_DECODE_BS_BUCKET_STEP means below this value, batch size is power by 2
            VLLM_PROMPT_BS_BUCKET_MIN=1 \
            VLLM_PROMPT_BS_BUCKET_STEP=${prefill_bs} \
            VLLM_PROMPT_BS_BUCKET_MAX=${prefill_bs} \
            VLLM_PROMPT_SEQ_BUCKET_MIN=${isl_aligned} \
            VLLM_PROMPT_SEQ_BUCKET_MAX=$((isl_aligned + block_size)) \
            VLLM_DECODE_BS_BUCKET_MIN=${cons_min} \
            VLLM_DECODE_BS_BUCKET_STEP=${decode_bs} \
            VLLM_DECODE_BS_BUCKET_MAX=${decode_bs} \
            VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
            VLLM_DECODE_BLOCK_BUCKET_STEP=${block_step} \
            VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
            QUANT_CONFIG=quant_files/inc_main/meta-llama-3.1-70b-instruct-2/maxabs_quant_g3.json \
            VLLM_DELAYED_SAMPLING=true \
            VLLM_SOFTMAX_CONST_NORM=true \
            VLLM_GRAPH_PROMPT_RATIO=0.1 \
            HABANA_VISIBLE_DEVICES="ALL" \
            PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
            PT_HPU_WEIGHT_SHARING=0 \
            python -m vllm.entrypoints.openai.api_server \
                --port 18080 \
                --model ${model} \
                --tensor-parallel-size ${tp_parrallel} \
                --max-num-seqs ${decode_bs} \
                --disable-log-requests \
                --dtype bfloat16 \
                --block_size ${block_size} \
                --use-v2-block-manager \
                --max-model-len ${max_num_batched_tokens} \
                --max-num-batched-tokens ${max_num_batched_tokens} \
                --distributed_executor_backend mp \
                --gpu_memory_utilization ${gpu_utils} \
                --kv_cache_dtype fp8_inc \
                --quantization 'inc' \
                --weights_load_device "cpu" 2>&1 | tee benchmark_results/${log_name}_serving.log &
            pid=$(($!-1))

            until [[ $ready == true ]]; do
                n=$((n+1))
                if grep -q "Started server process" benchmark_results/${log_name}_serving.log; then
                    break
                fi
                sleep 5s
            done
            sleep 10s
            echo ${pid}

            n_prompts=$((con * 10))
            log_name_run="${log_name}_con${con}_nprompts${n_prompts}"
            start_time=$(date +%s)
            echo "Start to warmup" >> benchmark_results/${log_name}_serving.log
            python /workspace/vllm/benchmarks/benchmark_serving.py \
                --backend vllm \
                --model ${model} \
                --port 18080 \
                --dataset-name "random" --random-input-len $isl --random-output-len $osl \
                --random-range-ratio 1.0 \
                --ignore-eos \
                --max-concurrency "$con" \
                --request-rate "$qps" \
                --num-prompts ${con} \
                --percentile-metrics "ttft,tpot,itl,e2el" 2>&1 | tee benchmark_results/${log_name_run}_warmup.log

            echo "Start to benchmark" >> benchmark_results/${log_name}_serving.log
            python /workspace/vllm/benchmarks/benchmark_serving.py \
                --backend vllm \
                --model ${model} \
                --port 18080 \
                --dataset-name "random" --random-input-len $isl --random-output-len $osl \
                --random-range-ratio 1.0 \
                --ignore-eos \
                --max-concurrency "$con" \
                --request-rate "$qps" \
                --num-prompts ${n_prompts} \
                --percentile-metrics "ttft,tpot,itl,e2el" 2>&1 | tee benchmark_results/${log_name_run}_run.log

            end_time=$(date +%s)
            echo "Time elapsed: $((end_time - start_time))s"

            kill ${pid}
            sleep 10
        done
    done
done
