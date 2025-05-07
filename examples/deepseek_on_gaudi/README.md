# **DeepSeek-R1 Inference Service with Intel® Gaudi® vLLM**
This README provides instructions on how to run DeepSeek-R1 inference on single and multi-node Gaudi setups.
## Single-Node Setup Guide for DeepSeek-R1 671B

### **1. Firmware and Software Stack**

Make sure you use the latest version of the Gaudi software and firmware:

- Driver: 1.21.0 [How to update Gaudi driver](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html)
- Firmware: 1.21.0 [How to update Gaudi firmware](https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html#system-unboxing-main)
- Docker: `vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest`

### **2. Start the Docker Image**

```bash
docker run -it --name deepseek_server --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

### **3. Download and Install vLLM**

```bash
git clone -b "v1.21.0" https://github.com/HabanaAI/vllm-fork.git
pip install -e vllm-fork/
```

### **4. Adjust Workload Parameters**

If required, modify the parameters in `single_16k_len_bf16.sh` or ``single_16k_len_fp8.sh`, based on the workload requirements and the recommended content length of 16k. Update the model_path according to the converted model path:

```bash
max_model_len=16384
max_num_batched_tokens=16384
max_num_seqs=256
input_min=1
input_max=16384
output_max=16384
model_path=deepseek-ai/DeepSeek-R1
```

### **5. Start vLLM**

### **Option 1: Run with runtime dequantize with block-based scale:**

```bash
./single_16k_len_bf16.sh
```

### **Option 2: Run with static quantization:**

#### **a. Get the calibration file:**

Refer to the vllm-hpu-extension calibration [README](https://github.com/HabanaAI/vllm-hpu-extension/blob/v1.21.0/calibration/README.md) to obtain the measurement files.

#### **b. Start vLLM:**

```bash
QUANT_CONFIG=<path to quant config> ./single_16k_len_fp8.sh
```

The vLLM server is ready to serve when the log below appears:

```bash
INFO 04-09 00:49:01 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 32.75 seconds
INFO 04-09 00:49:01 api_server.py:800] Using supplied chat template:
INFO 04-09 00:49:01 api_server.py:800] None
INFO 04-09 00:49:01 api_server.py:937] Starting vLLM API server on http://0.0.0.0:8688
INFO 04-09 00:49:01 launcher.py:23] Available routes are:
INFO 04-09 00:49:01 launcher.py:31] Route: /openapi.json, Methods: HEAD, GET

```

### **6. Check the vLLM Performance**

You can check the vLLM performance with the benchmark_vllm_client.sh script:
- Login the same container with the `docker exec -it deepseek_server /bin/bash` command.
- Copy `benchmark_vllm_client.sh` into the `vllm-fork/benchmarks` folder.
- In the script file, update the model path, vLLM server IP and port if required:

```bash
model_path=deepseek-ai/DeepSeek-R1
ip_addr=127.0.0.1
port=8688
```

- Execute this script in the `vllm-fork/benchmarks` folder:

```bash
pip install datasets
bash benchmark_vllm_client.sh
```

This script calls the standard vLLM benchmark serving tool to check the vLLM throughput.

### **7. Check the vLLM Accuracy**

#### **a. Install lm_eval**

```bash
pip install lm_eval[api]
```

#### **b. Run lm_eval**

Change the model path, vLLM IP address, or port in the command below if required:

```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=deepseek-ai/DeepSeek-R1,base_url=http://127.0.0.1:8688/v1/completions --batch_size 16 --log_samples --output_path ./lm_eval_output
```

## **Multi-Node Setup Guide for DeepSeek-R1 671B**

### **Environment Setup**

#### **1. Identical Firmware and Software Stack**

Make both nodes use the latest version of the Gaudi software and firmware:

- Driver: 1.21.0 [How to update Gaudi driver](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html)
- Firmware: 1.21.0 [How to update Gaudi firmware](https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html#system-unboxing-main)
- Docker: `vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest`

#### **2. Network Configuration**

- Ensure both nodes are connected to the same switch/router.
- Example IP configuration:
  - Node 1: `192.168.1.101`
  - Node 2: `192.168.1.106`
- For the nodes with both the internal/external network segments, you use the external IP address as shown below:
  - Node 1: `10.239.129.238`
  - Node 2: `10.239.129.70`

#### **3. Start the Docker Parameters**

```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all --device=/dev:/dev -v /dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --net=host --ipc=host vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

#### **4. Run HCCL Demo Test**

Make sure the HCCL demo test passes using the assigned IPs on the two nodes (16 HPUs) and get the expected all-reduce throughput.
Follow the instructions provided in the
[HCCL Demo Guide](https://github.com/HabanaAI/hccl_demo?tab=readme-ov-file#running-hccl-demo-on-2-servers-16-gaudi-devices)

##### Example Commands

**Head Node:**

```bash
HCCL_COMM_ID=192.168.1.101:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 0 --size 32m --ranks_per_node 8
```

**Worker Node:**

```bash
HCCL_COMM_ID=192.168.1.101:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 1 --size 32m --ranks_per_node 8
```

The expected throughput should look like the below:

```bash
#########################################################################################
[BENCHMARK] hcclAllReduce(dataSize=33554432, count=8388608, dtype=float, iterations=1000)
[BENCHMARK]     NW Bandwidth   : 205.899352 GB/s
[BENCHMARK]     Algo Bandwidth : 109.812988 GB/s
#########################################################################################
```

### **2. Install VLLM**

```bash
git clone -b "v1.21.0" https://github.com/HabanaAI/vllm-fork.git
pip install -e vllm-fork/
```

### **3. Configure Multi-Node Script**

#### **a. Set the IP address and NIC interface name in `set_header_node_sh` and `set_worker_node_sh`:**

```bash
# set IP address of header node
export VLLM_HOST_IP=192.168.1.101
# set NIC interface name of worker IP address
export GLOO_SOCKET_IFNAME=enx6c1ff7012f87
```

#### **b. Adjust the environment variables if required. Make sure the head node and worker node have the same configuration except for `VLLM_HOST_IP`, `GLOO_SOCKER_IFNAME` and `HCCL_SOCKET_IFNAME`:**

```bash
max_num_batched_tokens=32768
max_num_seqs=512
input_min=768
input_max=20480
output_max=16896
```

#### **c. Apply configuration on both nodes.**

Run the following command on both head and worker nodes:

Header node:

```bash
source set_header_node.sh
```

Worked node:

```bash
source set_worker_node.sh
```

### **4. Start Ray Cluster**

#### **a. Start Ray on the head node:**

```bash
ray start --head --node-ip-address=<HEADER_NODE_IP> --port=<PORT>
```

I. e.

```bash
ray start --head --node-ip-address=192.168.1.101 --port=8850
```

#### **b. Start Ray on the worker node:**

```bash
ray start --address='HEADER_NODE_IP:port'
```

I. e.

```bash
ray start --address='192.168.1.101:8850'
```

If you see the error message like "ray.exceptions.RaySystemError: System error: No module named 'vllm'", set the variable below:

```bash
echo 'PYTHONPATH=$PYTHONPATH:/workspace/vllm-fork' | tee -a /etc/environment
source /etc/environment

```

`/workspace/vllm-fork` is the vLLM source code folder. Update to your folder path.
### 5. Start vLLM on Head Node

See the example command below. It may take several hours to finish the warm-up for 32k context length on 2 nodes:

```bash
python -m vllm.entrypoints.openai.api_server \
    --host 192.168.1.101 \
    --port 8688 \
    --model deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 16 \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-log-requests \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_inc \
    --use-v2-block-manager \
    --num_scheduler_steps 1\
    --block-size $block_size \
    --max-model-len $max_num_batched_tokens \
    --distributed_executor_backend ray \
    --gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --trust_remote_code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
```
