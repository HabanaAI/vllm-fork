#!/bin/bash

export PT_HPU_LAZY_MODE=1

echo "List of arguments: $@"


# test config
# batch_size	seq_len	num_query_heads	num_kv_heads	head_size	block_size	dtype
# bfloat16
test1=(1  8192 16 16 128 128 'bfloat16')
test2=(1  8192 32 32 128 128 'bfloat16')
test3=(1  8192 32 4  128 128 'bfloat16')
test4=(64 8192 32 4  128 128 'bfloat16')

test5=(1  8192 52 4  128 128 'bfloat16')
test6=(64 8192 52 4  128 128 'bfloat16')

test7=(1  8192 16 2  128 128 'bfloat16')
test8=(64 8192 16 2  128 128 'bfloat16')

test9=( 1  8192 26 2  128 128 'bfloat16')
test10=(64 8192 26 2 128 128 'bfloat16')

test11=(1  8192 8  1 128 128 'bfloat16')
test12=(64 8192 8  1 128 128 'bfloat16')

test13=(1  8192 13  1 128 128 'bfloat16')
test14=(64 8192 13  1 128 128 'bfloat16')

# half
test15=(1  8192 16 16 128 128 'half')
test16=(1  8192 32 32 128 128 'half')
test17=(1  8192 32 4  128 128 'half')
test18=(64 8192 32 4  128 128 'half')

test19=(1  8192 52 4  128 128 'half')
test20=(64 8192 52 4  128 128 'half')

test21=(1  8192 16 2  128 128 'half')
test22=(64 8192 16 2  128 128 'half')

test23=( 1 8192 26 2  128 128 'half')
test24=(64 8192 26 2 128 128 'half')

test25=(1  8192 8  1 128 128 'half')
test26=(64 8192 8  1 128 128 'half')

test27=(1  8192 13  1 128 128 'half')
test28=(64 8192 13  1 128 128 'half')

block_size=(16 128)
dtype=('half' 'bfloat16')

timestamp=$(date +"%Y%m%d%H%M%S")

pushd kernels
if [[ $# -eq 0 || $1 != 'cuda' ]]; then
    # hpu    
    pa_hpu_res="pa_hpu_res_${timestamp}.csv"
    # head line
    echo "device,version,batch_size,seq_len,num_query_heads,num_kv_heads,head_size,block_size,dtype,time(us)" > "$pa_hpu_res"
    for bs in ${block_size[@]}; do
        for dt in ${dtype[@]}; do
            for i in $(seq 1 14); do
              param=test${i}
              log_file="test${i}_hpu.log"
              PREFIX=""
              if [ $dt = 'half' ]; then
                PREFIX="$PREFIX VLLM_PA_SOFTMAX_IMPL='wsum_head_amax'"
              fi
              echo "Add extra env: $PREFIX"
              echo ${log_file}
              eval "${PREFIX} python benchmark_paged_attention_1.21.py --device 'hpu' --batch-size \${$param[0]} --seq-len \${$param[1]} --num-query-heads \${$param[2]} --num-kv-heads \${$param[3]} --head-size \${$param[4]} --block-size ${bs}  --dtype ${dt} 2>&1 |tee ${log_file}"     
              #Kernel running time:
              t=$(grep "Kernel running time:" ${log_file} | awk '{print $4}')
              eval "echo hpu,null,\${$param[0]},\${$param[1]},\${$param[2]},\${$param[3]},\${$param[4]},${bs},${dt},${t} >> $pa_hpu_res"
              rm -f ${log_file}      
            done
        done
    done
else
    # cuda, v1 and v2, block size set as 16
    pa_cuda_res="pa_cuda_res_${timestamp}.csv"
    # head line
    echo "device,version,batch_size,seq_len,num_query_heads,num_kv_heads,head_size,block_size,dtype,time(us)" > "$pa_cuda_res"
    # v1
    for dt in ${dtype[@]}; do
        for i in $(seq 1 14); do
          param=test${i}
          log_file="test${i}_cuda.log"
          eval "python benchmark_paged_attention_1.21.py --version 'v1' --batch-size \${$param[0]} --seq-len \${$param[1]} --num-query-heads \${$param[2]} --num-kv-heads \${$param[3]} --head-size \${$param[4]} --block-size 16  --dtype ${dt} 2>&1 |tee ${log_file}"      
          #Kernel running time:
          t=$(grep "Kernel running time:" ${log_file} | awk '{print $4}')
          eval "echo cuda,v1,\${$param[0]},\${$param[1]},\${$param[2]},\${$param[3]},\${$param[4]},16,${dt},${t} >> $pa_cuda_res"
          rm -f ${log_file}      
        done
        
        # v2
        for i in $(seq 1 14); do
          param=test${i}
          log_file="test${i}_cuda.log"
          eval "python benchmark_paged_attention_1.21.py --version 'v2' --batch-size \${$param[0]} --seq-len \${$param[1]} --num-query-heads \${$param[2]} --num-kv-heads \${$param[3]} --head-size \${$param[4]} --block-size 16  --dtype ${dt} 2>&1 |tee ${log_file}"      
          #Kernel running time:
          t=$(grep "Kernel running time:" ${log_file} | awk '{print $4}')
          eval "echo cuda,v2,\${$param[0]},\${$param[1]},\${$param[2]},\${$param[3]},\${$param[4]},16,${dt},${t} >> $pa_cuda_res"
          rm -f ${log_file}      
        done
    done
fi
popd
