# Running DeepseekV3ForCausalLM Models on the deepseek_r1 Branch
**[中文](README.zh.md)**

This guide provides step-by-step instructions for deploying and running DeepseekV3ForCausalLM architecture models with the vLLM serving framework on Intel® Gaudi® HPUs. It covers hardware requirements, software prerequisites, model weight download and conversion, environment setup, model serving deployment, and performance/accuracy benchmarking on single-node and multi-node 8*Gaudi servers.

Verified models:
- moonshotai/Kimi-K2-Instruct
- deepseek-ai/DeepSeek-V3.1

## Table of Contents
- [Running DeepseekV3ForCausalLM Models on the deepseek_r1 Branch](#running-deepseekv3forcausallm-models-on-the-deepseek_r1-branch)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Software Prerequisites](#software-prerequisites)
  - [Model Weights Download and Conversion](#model-weights-download-and-conversion)
    - [Start a Docker Container on the Gaudi Server](#start-a-docker-container-on-the-gaudi-server)
    - [Download the Original Model](#download-the-original-model)
    - [Convert the Model](#convert-the-model)
  - [Single-Node Setup and Serving Deployment](#single-node-setup-and-serving-deployment)
    - [Download and Install vLLM](#download-and-install-vllm)
    - [HCCL Demo Test](#hccl-demo-test)
    - [INC FP8 Quantization](#inc-fp8-quantization)
    - [Parameters of the vLLM Start Script](#parameters-of-the-vllm-start-script)
    - [Launch vLLM Serving with TP=8](#launch-vllm-serving-with-tp8)
    - [Send a Request to Verify Service](#send-a-request-to-verify-service)
  - [Multi-Node Setup and Serving Deployment](#multi-node-setup-and-serving-deployment)
    - [Identical Software Stack](#identical-software-stack)
    - [Network Configuration](#network-configuration)
    - [Start Docker Containers](#start-docker-containers)
    - [HCCL Demo Test](#hccl-demo-test-1)
    - [Install vLLM on Both Nodes](#install-vllm-on-both-nodes)
    - [INC FP8 Quantization](#inc-fp8-quantization-multi-node)
    - [Configure Multi-Node Scripts](#configure-multi-node-scripts)
    - [Start the Ray Cluster](#start-the-ray-cluster)
    - [Start vLLM on the Head Node](#start-vllm-on-the-head-node)
  - [Check vLLM Performance](#check-vllm-performance)
  - [Check Model Accuracy](#check-model-accuracy)
    - [Enter the Running Docker Container](#enter-the-running-docker-container)
    - [Install lm_eval](#install-lm_eval)
    - [Set Proxy or HF Mirror if Needed](#set-proxy-or-hf-mirror-if-needed)
    - [Run lm_eval](#run-lm_eval)
  - [Tool Calling Support](#tool-calling-support)

## Hardware Requirements

* DeepSeek-V3.1
  * 685B parameters, FP8, about 642GB memory. A single 8*Gaudi2 OAM node (768GB total) fits weights plus KV cache for limited context (<=32k).
  * For higher concurrency or longer sequences, use 2 nodes with 8*Gaudi2.

* Kimi-K2-Instruct
  * 1T parameters, FP8, requires 2 nodes with 8*Gaudi2 to hold the weights.

Minimum per-node requirements for high-performance inference:

| Model | Server | CPU per Node | Accelerator per Node | RAM per Node | Storage per Node | Frontend Networking per Node <br>(Management/Storage) | Backend Networking per Node <br>(Compute, RDMA) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V3.1 | 1-node Gaudi2D | 2* 3rd Gen or newer Intel® Xeon® Scalable | 8* HL-225D 96GB OAM | ≥1.5TB | **OS:** ≥480GB SATA/SAS/NVMe SSD <br> **Data:** ≥2TB NVMe SSD | ≥1* 10GbE/25GbE NIC <br> or 1* NVIDIA® 200G BlueField-2 DPU/ConnectX-6 Dx SmartNIC | Not required |
| DeepSeek-V3.1 | 2-node Gaudi2D | 2* 3rd/4th Gen Intel® Xeon® Scalable | 8* HL-225D 96GB OAM | ≥1.5TB | **OS:** ≥480GB SATA/SAS/NVMe SSD <br> **Data:** ≥2TB NVMe SSD | ≥1* 10GbE/25GbE NIC <br> or 1* NVIDIA® 200G BlueField-2 DPU/ConnectX-6 Dx SmartNIC | 4* or 8* NVIDIA® HDR-200G ConnectX-6 Dx or NDR-400G ConnectX-7 |
| Kimi-K2-Instruct | 2-node Gaudi2D | 2* 3rd/4th Gen Intel® Xeon® Scalable | 8* HL-225D 96GB OAM | ≥1.5TB | **OS:** ≥480GB SATA/SAS/NVMe SSD <br> **Data:** ≥2TB NVMe SSD | ≥1* 10GbE/25GbE NIC <br> or 1* NVIDIA® 200G BlueField-2 DPU/ConnectX-6 Dx SmartNIC | 4* or 8* NVIDIA® HDR-200G ConnectX-6 Dx or NDR-400G ConnectX-7 |

### Set CPU to Performance Mode
Change BIOS CPU settings to performance-optimized, and in the OS run:
```bash
sudo echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Software Prerequisites

* Ubuntu 22.04 LTS (example).
* Install Docker on each node: https://docs.docker.com/engine/install/ubuntu/.
* Install Gaudi® driver and software stack (>= 1.23.0) on each node, ensure `habanalabs-container-runtime`: https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html.
* Upgrade Gaudi® firmware to >= 1.23.0: https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html.
* Configure the `habana` container runtime: https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#configure-container-runtime.

## Model Weights Download and Conversion

### Start a Docker Container on the Gaudi Server
Assume original weights are in /mnt/disk4. Allocate ≥1.5TB for DeepSeek-V3.1 and ≥2TB for Kimi-K2-Instruct.
> Note: Ensure the pulled image matches the Gaudi driver/OS. This guide uses the 1.23.0 driver/firmware and Ubuntu 22.04 image below; see the link above for other images.

```bash
docker run -it --name deepseek_server --runtime=habana -e HABANA_VISIBLE_DEVICES=all --device=/dev:/dev -v /dev:/dev -v /mnt/disk4:/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --net=host --ipc=host vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest
```

### Download the Original Model
We assume downloads go to /data/hf_models/.
```bash
sudo apt install git-lfs
git-lfs install

# Option 1: HuggingFace Kimi-K2-Instruct
git clone https://huggingface.co/moonshotai/Kimi-K2-Instruct /data/hf_models/Kimi-K2-Instruct
# Option 2: ModelScope Kimi-K2-Instruct
git clone https://www.modelscope.cn/moonshotai/Kimi-K2-Instruct.git /data/hf_models/Kimi-K2-Instruct

# Option 1: HuggingFace DeepSeek-V3.1
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3.1.git /data/hf_models/DeepSeek-V3.1
# Option 2: ModelScope DeepSeek-V3.1
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-V3.1.git /data/hf_models/DeepSeek-V3.1
```

### Convert the Model
Convert FP8 HuggingFace weights for Gaudi2D. Input folder: /data/hf_models/DeepSeek-V3.1; output: /data/hf_models/DeepSeek-V3.1-G2. Ensure >650GB free; conversion takes ~15 minutes with fast I/O. Do not add trailing slashes to paths.

`-i` option of convert_for_g2.py specifies the path to origin model weights, `-o` option specifies the output folder. Please don't add `/` to the end of input or output path. 

```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
pip install torch safetensors numpy --extra-index-url https://download.pytorch.org/whl/cpu

# Convert Kimi-K2-Instruct
python scripts/convert_for_g2.py -i /data/hf_models/Kimi-K2-Instruct -o /data/hf_models/Kimi-K2-Instruct-G2

# Convert DeepSeek-V3.1
python scripts/convert_for_g2.py -i /data/hf_models/DeepSeek-V3.1 -o /data/hf_models/DeepSeek-V3.1-G2
```
Example completion message:
```bash
...
processing /data/hf_models/DeepSeek-V3.1-G2/model-00163-of-000163.safetensors
skip model.layers.61.embed_tokens.weight.
skip model.layers.61.enorm.weight.
skip model.layers.61.hnorm.weight.
skip model.layers.61.input_layernorm.weight.
skip model.layers.61.post_attention_layernorm.weight.
skip model.layers.61.shared_head.head.weight.
skip model.layers.61.shared_head.norm.weight.
saving to /data/hf_models/DeepSeek-V3.1-G2/model-00163-of-000163.safetensors
```

## Single-Node Setup and Serving Deployment

### Download and Install vLLM
In the same container in which we convert the model weight, clone the latest code and install it. 
```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
pip install -e vllm-fork/
```

### HCCL Demo Test
Download HCCL Demo, compile and execute the hccl_demo test. Make sure the HCCL demo test passes on 8 HPU. For detailed info, please refer to [HCCL Demo](https://github.com/HabanaAI/hccl_demo)
```bash
git clone https://github.com/HabanaAI/hccl_demo.git
cd hccl_demo
make
HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1000 --ranks_per_node 8
```
Expected pass output (Gaudi PCIe without host NIC scale-out should see ~18GB/s via CPU UPI):
```bash
[BENCHMARK] hcclAllReduce(dataSize=33554432, count=8388608, dtype=float, iterations=1000)

[BENCHMARK]     NW Bandwidth   : 258.259144 GB/s
[BENCHMARK]     Algo Bandwidth : 147.576654 GB/s
```

### INC FP8 Quantization

To run DeepSeek-V3.1 with INC FP8 quantization on a single node:

#### 1. Calibrate DeepSeek-V3.1
Calibrate the model and it generates measurement files in scripts/nc_workspace_measure_kvcache.
```bash
cd vllm-fork
bash scripts/run_inc_calib.sh --model /data/hf_models/DeepSeek-V3.1-G2 --nprompts 5000
```

#### 2. Configure environment variables (optional)
After downloading measurement files, you need to configure some environment variables to make INC quantization become effective.

##### 2.1 Using start_vllm.sh
If you use `start_vllm.sh`, plan to use FP8 KV cache, and stored measurements at the recommended path (/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache), you can keep defaults. To use a custom path or BF16 KV cache, set `QUANT_CONFIG` and `INC_MEASUREMENT_DUMP_PATH_PREFIX` in start_vllm.sh.

- QUANT_CONFIG

Depends on kv-cache-dtype to use, you should use quantization configuration file accordingly.

These quantization config is located in vllm-fork/scripts/quant_configs.

| KV-Cache-Dtype | vLLM --kv-cache-dtype | QUANT_CONFIG |
| --- | --- | --- |
| BF16 | auto | inc_quant_per_channel_bf16kv.json |
| FP8 | fp8_inc | inc_quant_per_channel_with_fp8kv_config.json |

Example (FP8 KV cache):
```bash
export QUANT_CONFIG=/path/to/vllm-fork/scripts/quant_configs/inc_quant_per_channel_with_fp8kv_config.json
```
Pass `fp8_inc` to `--kv-cache-dtype`.

- INC_MEASUREMENT_DUMP_PATH_PREFIX

The environment variable `INC_MEASUREMENT_DUMP_PATH_PREFIX` specifies the root directory where measurement statistics were saved.
The final path is constructed by joining this root directory with the `dump_stats_path` defined in the quantization JSON file specified by the `QUANT_CONFIG` environment variable.

If we download the measurements to `/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache`, we got below files:
```bash
user:vllm-fork$ ls -l ./scripts/nc_workspace_measure_kvcache
-rw-r--r-- 1 user Software-SG 1949230 May 15 08:05 inc_measure_output_hooks_maxabs_0_8.json
-rw-r--r-- 1 user Software-SG  254451 May 15 08:05 inc_measure_output_hooks_maxabs_0_8_mod_list.json
-rw-r--r-- 1 user Software-SG 1044888 May 15 08:05 inc_measure_output_hooks_maxabs_0_8.npz
...
```
Then, we export `INC_MEASUREMENT_DUMP_PATH_PREFIX=/path/to/vllm-fork`, and INC will parse the full as below:

```
dump_stats_path (from config): "scripts/nc_workspace_measure_kvcache/inc_measure_output"
Resulting full path: "/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache/inc_measure_output_hooks_maxabs_0_8.npz"
```

#### 3. Verify INC enablement
`Preparing model with INC` should appear in vLLM server logs.

### Parameters of the vLLM Start Script
Check supported parameters:
```bash
bash start_vllm.sh -h
```
Output:
```
Start vllm server for a huggingface model on Gaudi.

Syntax: bash start_vllm.sh <-w> [-u:p:l:b:c:sq] [-h]
options:
w  Weights of the model, could be model id in huggingface or local path
u  URL of the server, str, default=0.0.0.0
p  Port number for the server, int, default=8688
l  max_model_len for vllm, int, default=16384, maximal value for single node: 32768
b  max_num_seqs for vllm, int, default=64
c  Cache HPU recipe to the specified path, str, default=None
s  Skip warmup or not, bool, default=false
q  Enable inc fp8 quantization
m  Max number of the prefill sequences, int, default=1 to optimize TTFT
h  Help info
```

### Launch vLLM Serving with TP=8
```bash
bash start_vllm.sh -w /data/hf_models/DeepSeek-V3.1-G2 -q -u 0.0.0.0 -p 8688 -l 16384 -c /data/warmup_cache
```

It takes more than 1 hour to load and warm up the model for the first time. After completion, a typical output would be like below. The warmup time will be accelerated if the warmup cache is reused. vLLM server is ready to serve when the log below appears.
```bash
INFO 04-09 00:49:01 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 32.75 seconds
INFO 04-09 00:49:01 api_server.py:800] Using supplied chat template:
INFO 04-09 00:49:01 api_server.py:800] None
INFO 04-09 00:49:01 api_server.py:937] Starting vLLM API server on http://0.0.0.0:8688
INFO 04-09 00:49:01 launcher.py:23] Available routes are:
INFO 04-09 00:49:01 launcher.py:31] Route: /openapi.json, Methods: HEAD, GET
```
### Send a Request to Verify Service
On bare metal, execute the following command to send a request to the Chat Completions API endpoint using `cURL`: 
```bash
curl http://127.0.0.1:8688/v1/chat/completions \
  -X POST \
  -d '{"model": "/data/hf_models/DeepSeek-V3.1-G2", "messages": [{"role": "user", "content": "List 3 countries and their capitals."}], "max_tokens":128}' \
  -H 'Content-Type: application/json'
```
If the response is normal, proceed to [Check the vLLM Performance](#Check-the-vLLM-performance) and [Check the Model Accuracy](#check-the-model-accuracy) to measure the performance and accuracy.

## Multi-Node Setup and Serving Deployment
vLLM on Gaudi supports multi-node serving. Example: 2 nodes, TP=16.

### Identical Software Stack
Ensure both nodes use:
- Driver: 1.23.0 (https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html)
- Firmware: 1.23.0 (https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html#system-unboxing-main)
- Docker image: vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest
- vLLM branch for DeepSeek-V3.1: https://github.com/HabanaAI/vllm-fork/tree/deepseek_r1
- vLLM HPU extension: https://github.com/HabanaAI/vllm-hpu-extension/tree/deepseek_r1

### Network Configuration
- Ensure both nodes are connected to the same switch/router.
- Example IP configuration:
  - Node 1: `192.168.1.101`
  - Node 2: `192.168.1.106`
- For the nodes with both the internal/external network segments, you may also use the external IP address like
  - Node 1: `10.239.129.238`
  - Node 2: `10.239.129.70`

### Start Docker Containers
Use the command below to start the container on both nodes. Assume that the converted model weight files are in the folder /mnt/disk4. Please make sure that the mapped model weight folders are in the same path. 
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all --device=/dev:/dev -v /dev:/dev -v /mnt/disk4:/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --net=host --ipc=host vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest
```

### HCCL Demo Test
Make sure the HCCL demo test passes using the assigned IPs on the two nodes (16 HPU) and get the expected all-reduce throughput. 
HCCL demo guide document: https://github.com/HabanaAI/hccl_demo?tab=readme-ov-file#running-hccl-demo-on-2-servers-16-gaudi-devices

Head node:
```bash
HCCL_COMM_ID=192.168.1.101:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 0 --size 32m --ranks_per_node 8
```
Worker node:
```bash
HCCL_COMM_ID=192.168.1.101:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 1 --size 32m --ranks_per_node 8
```
Expected:
```
#########################################################################################
[BENCHMARK] hcclAllReduce(dataSize=33554432, count=8388608, dtype=float, iterations=1000)
[BENCHMARK]     NW Bandwidth   : 205.899352 GB/s
[BENCHMARK]     Algo Bandwidth : 109.812988 GB/s
#########################################################################################
```

### Install vLLM on Both Nodes
```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
pip install -e vllm-fork/
```

### INC FP8 Quantization (multi-node)
To run DeepSeek-V3.1 with INC FP8 quantization in multi-nodes case, you need to follow:

#### 1. Calibrate DeepSeek-V3.1 on multi-node.
To deploy DeepSeek-V3.1 with multi-node tensor parallelism (e.g. TP16), you must calibrate the model using the same multi-node configuration. Follow the steps below to generate the measurement files.

> **Note:** Kimi-K2-Instruct calibration by default requires two HPU nodes. Follow the same instructions below, updating the model name and path as needed.

- Start Ray on head node.
```bash
HABANA_VISIBLE_MODULES='0,1,2,3,4,5,6,7'  \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
ray start --head --resources='{"HPU": 8, "TPU": 0}'
```

- Start Ray on worker node.
```bash
HABANA_VISIBLE_MODULES='0,1,2,3,4,5,6,7'  \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
ray start --address='${head_ip}:6379' --resources='{"HPU": 8, "TPU": 0}'
```

- Start calibration on head node.
```bash
cd vllm-fork
bash scripts/run_inc_calib.sh --wd 16 --model /data/hf_models/DeepSeek-V3.1-G2 --nprompts 5000
```

- Copy the calibration output to other node.

Check the caliration files on head node and worker node:
```bash
ls ./scripts/nc_workspace_measure_kvcache #head node
-rw-r--r-- 1 root root 1616977 Jan 23 12:04 inc_measure_output_hooks_maxabs_0_16.json
-rw-r--r-- 1 root root  866890 Jan 23 12:04 inc_measure_output_hooks_maxabs_0_16.npz
-rw-r--r-- 1 root root  206353 Jan 23 12:04 inc_measure_output_hooks_maxabs_0_16_mod_list.json
...
-rw-r--r-- 1 root root 1616883 Jan 23 12:04 inc_measure_output_hooks_maxabs_7_16.json
-rw-r--r-- 1 root root  866890 Jan 23 12:04 inc_measure_output_hooks_maxabs_7_16.npz
-rw-r--r-- 1 root root  206353 Jan 23 12:04 inc_measure_output_hooks_maxabs_7_16_mod_list.json
```

```bash
ls ./scripts/nc_workspace_measure_kvcache #worker node
-rw-r--r-- 1 root root 1617266 Jan 23 11:59 inc_measure_output_hooks_maxabs_8_16.json
-rw-r--r-- 1 root root  866890 Jan 23 11:59 inc_measure_output_hooks_maxabs_8_16.npz
-rw-r--r-- 1 root root  206353 Jan 23 11:59 inc_measure_output_hooks_maxabs_8_16_mod_list.json
...
-rw-r--r-- 1 root root 1617084 Jan 23 11:59 inc_measure_output_hooks_maxabs_15_16.json
-rw-r--r-- 1 root root  866890 Jan 23 11:59 inc_measure_output_hooks_maxabs_15_16.npz
-rw-r--r-- 1 root root  206353 Jan 23 11:59 inc_measure_output_hooks_maxabs_15_16_mod_list.json
```

It is recommanded to copy and combine the calibrations from head node and worker node altoghter. In the runtime, different rank will look for calibration files for its rank.

#### 2. Configure environment variables.

After measurement files are ready, you need to configure some environment variables to make INC quantization become effective.

##### 2.1 Using set_head_node.sh & set_worker_node.sh scripts

If you are using `set_head_node.sh` and `set_worker_node.sh` scripts to start vllm, please configure `QUANT_CONFIG` and `INC_MEASUREMENT_DUMP_PATH_PREFIX` env var in them.

- QUANT_CONFIG

Depends on kv-cache-dtype to use, you should use quantization configuration file accordingly.

These quantization config is located in vllm-fork/scripts/quant_configs.

| KV-Cache-Dtype | vLLM --kv-cache-dtype | QUANT_CONFIG |
| --- | --- | --- |
| BF16 | auto | inc_quant_per_channel_bf16kv.json |
| FP8 | fp8_inc | inc_quant_per_channel_with_fp8kv_config.json |

Example (BF16 KV cache):
```bash
export QUANT_CONFIG=/path/to/vllm-fork/scripts/quant_configs/inc_quant_per_channel_bf16kv.json
```
Pass `auto` to `--kv-cache-dtype`.

- INC_MEASUREMENT_DUMP_PATH_PREFIX

The environment variable `INC_MEASUREMENT_DUMP_PATH_PREFIX` specifies the root directory where measurement statistics were saved.
The final path is constructed by joining this root directory with the `dump_stats_path` defined in the quantization JSON file specified by the `QUANT_CONFIG` environment variable.

If we download the measurements to `/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache`, we got below files:
```bash
user:vllm-fork$ ls -l ./scripts/nc_workspace_measure_kvcache
-rw-r--r-- 1 root root 1136822 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16.json
-rw-r--r-- 1 root root  611732 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16.npz
-rw-r--r-- 1 root root  155379 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16_mod_list.json
...
```
Then, we export `INC_MEASUREMENT_DUMP_PATH_PREFIX=/path/to/vllm-fork`, and INC will parse the full as below:

```
dump_stats_path (from config): "scripts/nc_workspace_measure_kvcache/inc_measure_output"
Resulting full path: "/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache/inc_measure_output_hooks_maxabs_0_16.npz"
```

### Configure Multi-Node Scripts
- Set IP and NIC in set_head_node.sh / set_worker_node.sh:
```bash
export VLLM_HOST_IP=192.168.1.101
export GLOO_SOCKET_IFNAME=enx6c1ff7012f87
```
- Adjust shared env vars (head and workers identical except for VLLM_HOST_IP/GLOO_SOCKET_IFNAME values):
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/data/cache/cache_32k,false,32768
export max_num_batched_tokens=32768
export max_num_seqs=512
```

- Apply configuration
```bash
source set_head_node.sh   # on head
source set_worker_node.sh # on workers
```

### Start the Ray Cluster
Head node:
```bash
ray start --head --node-ip-address=192.168.1.101 --port=8850
```
Worker nodes:
```bash
ray start --address='192.168.1.101:8850'
```
If you see `ray.exceptions.RaySystemError: System error: No module named 'vllm'`, set:
```bash
echo 'PYTHONPATH=$PYTHONPATH:/workspace/vllm-fork' | tee -a /etc/environment
source /etc/environment
```

### Start vLLM on the Head Node
Example (warmup to 32k may take hours on 2 nodes):
```bash
python -m vllm.entrypoints.openai.api_server \
    --host 192.168.1.101 \
    --port 8688 \
    --model /data/hf_models/DeepSeek-V3.1-G2 \
    --tensor-parallel-size 16 \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-log-requests \
    --dtype bfloat16 \
    --kv-cache-dtype $KV_CACHE_DTYPE \
    --use-v2-block-manager \
    --num-scheduler-steps 1 \
    --block-size $block_size \
    --max-model-len $max_num_batched_tokens \
    --distributed-executor-backend ray \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --trust-remote-code
```

## Check vLLM Performance
Use benchmark_vllm_client.sh inside the container (e.g., `docker exec -it deepseek_server /bin/bash`). Copy the script into vllm-fork/benchmarks and update model/IP/port if needed:
```bash
model_path=/data/hf_models/DeepSeek-V3.1-G2
ip_addr=127.0.0.1
port=8688
```
Run:
```bash
pip install datasets
bash benchmark_vllm_client.sh
```
This calls the standard vLLM benchmark with 1k input/output tokens and concurrency 1 and 32.

## Check Model Accuracy
### Enter the Running Docker Container
### Install lm_eval
```bash
pip install lm_eval[api]
```
### Set Proxy or HF Mirror if Needed
```bash
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=127.0.0.1
```
### Run lm_eval
Update model/vLLM IP/port if needed:
```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=/data/hf_models/DeepSeek-V3.1-G2,max_gen_toks=4096,max_length=16384,base_url=http://127.0.0.1:8688/v1/completions --batch_size 16 --log_samples --output_path ./lm_eval_output
```

## Tool Calling Support
vLLM supports user-defined tool calling.

### DeepSeek-V3.1 Models (`deepseek_v31`)
Supported: `deepseek-ai/DeepSeek-V3.1` (use examples/tool_chat_template_deepseekv31.jinja)
```bash
vllm serve ... \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v31 \
    --chat-template ../../examples/tool_chat_template_deepseekv31.jinja
```

### Kimi-K2 Models (`kimi_k2`)
Supported: `moonshotai/Kimi-K2-Instruct`
```bash
vllm serve ... \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2
```
