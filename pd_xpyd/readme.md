# Prefill/Decode Disaggregation & Data Parallel Usage Guide

## System setup
In host OS:
```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo 32768 > /proc/sys/vm/nr_hugepages
cat /proc/meminfo | grep Huge
cat /proc/sys/vm/nr_hugepages
```

In docker OS:
```bash
bash setup_env.sh
```

## Prefill/Decode Disaggregation & Data Parallel Usage

0. prepare and modify mooncake.json
```bash
# create a new file with the name "mooncake_$machine.json (mooncake_g1.json, g1 is the machine name)
# refer to existing mooncake json file and do adjustments:
`local_hostname` is the local machine IP
`metadata_server` is the remote machine IP where etcd is launched
`protocal` is rdma or tcp
`master_server_address` is the remote machine IP where mooncake master is launched. high speed network is preferred.
```

1. Adjust 1p_start_prefill.sh to automatically launch/stop etc & mooncake master server. refer to following example:
```bash
if [ -z "$1" ] || [ "$1" == "g10" ] || [ "$1" == "pcie4" ]; then
    if [ "$DEBUG_MODE" == "1" ]; then
    	source "$BASH_DIR"/start_etc_mooncake_master.sh debug
    	echo "source "$BASH_DIR"/start_etc_mooncake_master.sh debug"
    else
	source "$BASH_DIR"/start_etc_mooncake_master.sh
        echo "source "$BASH_DIR"/start_etc_mooncake_master.sh"
    fi
fi
```

2. Adjust dp0_xp2d_start_decode.sh, dp1_xp2d_start_decode.sh for decode node(DP16/2Nodes as an example)
```bash
last line: source "$BASH_DIR"/dp_start_decode.sh g13 16 $TP_SIZE 0 "10.239.129.81" 
parameters are:	machine, DP Size, TP Size, DP Index, DP Host IP
```

3. Adjust proxy server, xpyd_start_proxy.sh
```bash
# Adjust Prefill/Decode IPs
PREFILL_IPS=("10.239.129.9" "10.239.129.67" "10.239.129.21" "10.239.128.165" "10.239.128.244" "10.239.128.153")
DECODE_IPS=("10.239.129.81" "10.239.129.165" "10.239.129.67" "10.239.129.21")
```

4. Start prefill server(s): source 1p_start_prefill.sh $machine (note: the machine with etcd/mooncake master MUST be launched firstly)

5. Start decode servers(s): source dp0_xp2d_start_decode.sh / source dp1_xp2d_start_decode.sh

6. Start proxy server: bash xpyd_start_proxy.sh x y z false 
```bash
# note: x for prefill nodes number, y for decode nodes number, z for decode tp size, false for first token from decode
```

7. Launch client command for inference. --host ip is the proxy server ip


## Benchmark mode:
```bash
# A benchmark mode is designed for easy & quick benchmarking with 1 prefill node for any large decode batch size
benchmark bs1024, 2 decode nodes, tp size 1, as an example:
step1: revise --repeat_d_times to 1023 (1024-1) in xpyd_start_proxy.sh
step2: launch prefill server: source 1p_start_prefill.sh $machine benchmark 
step3: launch decode server(s) as normal
step4: launch proxy server: bash xpyd_start_proxy.sh 1 2 1 false benchmark 
step5: lanch client command with only 1 prompt
python3 benchmarks/benchmark_serving.py --backend vllm --model /mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/ --dataset-name sonnet --request-rate inf --host 10.239.129.9 --port 8868 --sonnet-input-len 2000 --sonnet-output-len 1000 --sonnet-prefix-len 100 --trust-remote-code --max-concurrency 1024 --num-prompts 1 --ignore-eos --burstiness 1000 --dataset-path benchmarks/sonnet.txt --save-result
```

