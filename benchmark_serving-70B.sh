#!/bin/bash

# server settings
model="/data/Llama-3.1-70B-Instruct"
model_name="Llama-3.1-70B-Instruct"
qps="inf"
gpu_utils=0.9

#tp_parrallel_list=(2 4 8)
#req_in_out_list=(128_2048 128_4096 500_2000 2048_2048)
tp_parrallel_list=(8)
req_in_out_list=(128_2048 128_4096 500_2000 2048_2048)
cons=(768 512 384 256 128 64 32 16 8 4 2 1)
for req_in_out in "${req_in_out_list[@]}"; do
    for tp_parrallel in "${tp_parrallel_list[@]}"; do
        isl=$(echo "$req_in_out" | awk -F'_' '{ print $1 }')
        osl=$(echo "$req_in_out" | awk -F'_' '{ print $2 }')
        # get cons_min and cons_max based on cons
        cons_min=($(echo "${cons[@]}" | tr ' ' '\n' | sort -n | head -n 1))
        cons_max=($(echo "${cons[@]}" | tr ' ' '\n' | sort -n | tail -n 1))

        log_name="[test3]gaudi3-fp8-${model_name}-${gpu_utils}util-TPparallel${tp_parrallel}-isl${isl}-osl${osl}"

        total_len=$((isl + osl))
        # if total_len is not multiple of 128, round up to the next multiple of 128
        total_len=$(((total_len + 127) / 128 * 128))
        bs=$((4096 * tp_parrallel * 128 / total_len))
        bs_step=128
        bs=$(((bs + bs_step - 1) / bs_step * bs_step))

        if [ $bs -gt ${cons_max} ]; then
            bs=${cons_max}
        fi
        isl_aligned=$(((isl + 127) / 128 * 128))
        VLLM_DECODE_BLOCK_BUCKET_MIN=$((isl_aligned * bs / 128))
        VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len * bs / 128))
        prefill_bs=$((8192 / $isl_aligned))
        
        echo "isl: $isl, osl: $osl, prefill_seqlen: $isl_aligned, prefill_bs: $prefill_bs, decode_bs: $cons_min : $bs, VLLM_DECODE_BLOCK_BUCKET_MIN: $VLLM_DECODE_BLOCK_BUCKET_MIN, VLLM_DECODE_BLOCK_BUCKET_MAX: $VLLM_DECODE_BLOCK_BUCKET_MAX"

        VLLM_PROMPT_BS_BUCKET_MIN=1 \
        VLLM_PROMPT_BS_BUCKET_MAX=${prefill_bs} \
        VLLM_PROMPT_SEQ_BUCKET_MIN=${isl_aligned} \
        VLLM_PROMPT_SEQ_BUCKET_MAX=${isl_aligned} \
        VLLM_DECODE_BS_BUCKET_MIN=${cons_min} \
        VLLM_DECODE_BS_BUCKET_STEP=128 \
        VLLM_DECODE_BS_BUCKET_MAX=${bs} \
        VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
        VLLM_DECODE_BLOCK_BUCKET_STEP=1024 \
        VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
        QUANT_CONFIG=quant_files/inc_main/meta-llama-3.1-70b-instruct/maxabs_quant_g3.json \
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
            --max-num-seqs ${bs} \
            --disable-log-requests \
            --dtype bfloat16 \
            --block_size 256 \
            --use-v2-block-manager \
            --max-model-len 8192 \
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

        for con in "${cons[@]}"; do
            n_prompts=$((con * 10))
            log_name_run="${log_name}_con${con}_nprompts${n_prompts}"
            start_time=$(date +%s)
            echo "Start to benchmark"
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

            sleep 10
        done

        kill ${pid}
        sleep 10
    done
done
