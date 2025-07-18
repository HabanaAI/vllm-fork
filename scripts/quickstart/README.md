# DeepSeek-R1 Support

This guide provides the step-by-step instructions on deploying and running DeepSeek-R1 671B with vLLM serving framework on Intel® Gaudi® HPUs. It covers the hardware requirements, software prerequisites, model weights downloading and conversion, environment setup, model serving deployment and performance and accuracy benchmarking on single-node and multi-node 8\*Gaudi servers. 

## Table of Contents

- [DeepSeek-R1 Support](#deepseek-r1-support)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Software Prerequisites](#software-prerequisites)
  - [Model Weights Downloading and Conversion](#model-weights-downloading-and-conversion)
    - [Start Docker Container on the Gaudi Server](#start-docker-container-on-the-gaudi-server)
    - [Download the Original Model](#download-the-original-model)
    - [Convert the Model](#convert-the-model)
  - [Single-Node Setup and Serving Deployment](#single-node-setup-and-serving-deployment)
    - [Download and Install vLLM](#download-and-install-vllm)
    - [HCCL Demo Test](#hccl-demo-test)
    - [The parameters of starting vLLM server script](#the-parameters-of-starting-vllm-server-script)
    - [Launch vLLM Serving with TP=8](#launch-vllm-serving-with-tp8)
    - [Send Requests to Ensure the Service Functionality](#send-requests-to-ensure-the-service-functionality)
  - [Multi-Node Setup and Serving Deployment](#multi-node-setup-and-serving-deployment)
    - [Identical Software Stack](#identical-software-stack)
    - [Network Configuration](#network-configuration)
    - [Start Dcoker Container Parameters](#start-docker-container-parameters)
    - [HCCL Demo Test](#hccl-demo-test)
    - [Install vLLM on Both Nodes](#install-vllm-on-both-nodes)
    - [Configure Multi-Node Script](#configure-multi-node-script)
    - [Start Ray Cluster](#start-ray-cluster)
    - [Start vLLM on Head Node](#start-vllm-on-head-node)
  - [Check the vLLM Performance](#check-the-vllm-performance)
  - [Check the Model Accuracy](#check-the-model-accuracy)
    - [Enter the Running Docker Container in the Above Way](#enter-the-running-docker-container-in-the-above-way)
    - [Install lm\_eval](#install-lm_eval)
    - [Set Proxy or HF Mirror if Required](#set-proxy-or-hf-mirror-if-required)
    - [Run lm\_eval](#run-lm_eval)

## Hardware Requirements

* DeepSeek-R1 671B has 671B parameters with FP8 precision and takes up about 642GB memory. Single-node 8\*Gaudi2 OAM (768GB memory in total) is enough to accommodate the model weights and required KV cache with limited context length (<=32k). 

* To support higher concurrency and longer token lengths, 2-node 8\*Gaudi2 servers are recommended. 

The following table outlines the mininum requirements for each hardware component of each node to achieve high-performance inference.

| Servers | CPU per Node | Accelerators per Node | RAM per Node | Storage per Node | Frontend Networking per Node <br>(In-band Management/Storage) | Backend Networking per Node <br>(Compute, w/ RDMA) |
| -------------- | --------------------------------------------------- | -------------------- | ------------- | ------------------------------------------------------ | -- | -- |
| 1-node Gaudi2D | 2\* 3rd or newer Gen Intel® Xeon® Scalable Processors | 8\* HL-225D 96GB OAM | Mininum 1.5TB | **OS:** At least 480GB SATA/SAS/NVMe SSD, <br> **Data:** At least 2TB NVMe SSD | At least 1\* 10GbE/25GbE NIC <br> or 1\* NVIDIA® 200G BlueField-2 DPU/ConnectX-6 Dx SmartNIC | Not Required |
| 2-node Gaudi2D | 2\* 3rd or 4th Gen Intel® Xeon® Scalable Processors | 8\* HL-225D 96GB OAM | Mininum 1.5TB | **OS:** At least 480GB SATA/SAS/NVMe SSD, <br> **Data:** At least 2TB NVMe SSD | At least 1\* 10GbE/25GbE NIC <br> or 1\* NVIDIA® 200G BlueField-2 DPU/ConnectX-6 Dx SmartNIC | 4\* or 8\* NVIDIA® HDR-200G ConnectX-6 Dx SmartNIC/HCA or NDR-400G ConnectX-7 SmartNIC/HCA |

### Set CPU to Performance Mode
Please change the CPU setting to be performance optimization mode in BIOS setup and execute the command below in OS to make sure get the best CPU performance. 
```
sudo echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Software Prerequisites

* This guide uses deployment on Ubuntu 22.04 LTS as an example.

* Refer to [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/) to install Docker on each node.

* Refer to [Driver and Software Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install the Gaudi® driver and software stack (>= 1.20.1) on each node. Make sure `habanalabs-container-runtime` is installed.

* Refer to [Firmware Upgrade](https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html) to upgrade the Gaudi® firmware to >=1.20.1 version on each node.

* Refer to [Configure Container Runtime](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#configure-container-runtime) to configure the `habana` container runtime on each node.



## Model Weights Downloading and Conversion

### Start Docker Container on the Gaudi Server
Assume that the original DeepSeek-R1 model weight files are downloaded and converted in the folder /mnt/disk4 which has at least 1.5TB disk space. 
> [!NOTE]
> * Make sure the pulled docker image aligns with the corresponding Gaudi driver and OS version. The default one used in this guide is for Gaudi driver/firmware 1.20.1 and Ubuntu 22.04, referring to [Use Intel(R)Gaudi Containers](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#use-intel-gaudi-containers) for other images.

```bash
docker run -it --name deepseek_server --runtime=habana -e HABANA_VISIBLE_DEVICES=all -v /mnt/disk4:/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

### Download the Original Model

The original DeepSeek-R1 model is available both on [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1) and [ModelScope](https://www.modelscope.cn/deepseek-ai/DeepSeek-R1). We assume that the model is downloaded in the folder, "/data/hf_models/". 

```bash
sudo apt install git-lfs
git-lfs install

# Option1: Download from HuggingFace
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1 /data/hf_models/DeepSeek-R1
# Option2: Download from ModelScope
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1 /data/hf_models/DeepSeek-R1
```

### Convert the Model
To serve DeepSeek-R1 model with Gaudi2D, the original HuggingFace FP8 model weights should be converted to channel-wise FP8 model weights using the command below on Gaudi server. We assume that the original model has been downloaded in the /data/hf_models/DeepSeek-R1 folder and the converted model will be saved in the /data/hf_models/DeepSeek-R1-Gaudi folder. Please make sure that the new folder has enough disk space (>650GB). If the disk I/O is fast enough, it will take about 15 minutes to finish the conversion.

```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
pip install compress_pickle torch safetensors numpy --extra-index-url https://download.pytorch.org/whl/cpu

python scripts/convert_block_fp8_to_channel_fp8.py --model_path /data/hf_models/DeepSeek-R1 --qmodel_path /data/hf_models/DeepSeek-R1-Gaudi --input_scales_path scripts/DeepSeek-R1-BF16-w8afp8-static-no-ste_input_scale_inv.pkl.gz
```

The conversion is finished when the message below is shown. The converted model weight files are saved in your specified folder, like /data/hf_models/DeepSeek-R1-Gaudi, and we will use this converted model to host vLLM service. 

```bash
INFO:__main__:[160/163] Saving 649 tensors to /data/hf_models/DeepSeek-R1-Gaudi/model-00160-of-000163.safetensors
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 163/163 [11:30<00:00,  4.24s/it]
Saving tensor mapping to /data/hf_models/DeepSeek-R1-Gaudi/model.safetensors.index.json
Conversion is completed.

```

## Single-Node Setup and Serving Deployment
### Download and Install vLLM
In the same container in which we convert the model weight, clone the latest code and install it. 
```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-hpu-extension.git
pip install -e vllm-fork/
pip install -e vllm-hpu-extension/
```
### HCCL Demo Test
Download HCCL Demo, compile and execute the hccl_demo test. Make sure the HCCL demo test passes on 8 HPU. For detailed info, pelease refer to [HCCL Demo](https://github.com/HabanaAI/hccl_demo)
```bash
git clone https://github.com/HabanaAI/hccl_demo.git
cd hccl_demo
make
HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1000 --ranks_per_node 8
```

The hccl test is passed if the message below is shown. For Gaudi PCIe system without Host NIC scale-out, the communication need go throughput CPU UPI and NW Bandwidth should be about 18GB/s. 
```bash 
#########################################################################################
[BENCHMARK] hcclAllReduce(dataSize=33554432, count=8388608, dtype=float, iterations=1000)
[BENCHMARK]     NW Bandwidth   : 258.259144 GB/s
[BENCHMARK]     Algo Bandwidth : 147.576654 GB/s
#########################################################################################
```


### The parameters of starting vLLM server script
There are some system environment variables which need be set to get the best vLLM performance. We provide the sample script to set the recommended environment variables.

The script file "start_vllm.sh" is used to start vLLM service. You may execute the command below to check its supported parameters.
```bash
bash start_vllm.sh -h
```

The command output is like below. 
```bash
Start vllm server for a huggingface model on Gaudi.

Syntax: bash start_vllm.sh <-w> [-u:p:l:b:c:s:q] [-h]
options:
w  Weights of the model, could be model id in huggingface or local path
u  URL of the server, str, default=0.0.0.0
p  Port number for the server, int, default=8688
l  max_model_len for vllm, int, default=16384, maximal value for single node: 32768
b  max_num_seqs for vllm, int, default=128
c  Cache HPU recipe to the specified path, str, default=None
s  Skip warmup or not, bool, default=false
q  Enable inc fp8 quantization
h  Help info
```

#### INC FP8 Quantization

To run DeepSeek-R1 with INC FP8 quantization in single-node case, you need to follow:

1. Download corresponding measurement files according to target model and tp-size.

|Model|TP-Size|Measurement Files|
|---|---|---|
|DeepSeek-R1-0528|8|Yi30/ds-r1-0528-default-pile-g2-0529|
|DeepSeek-R1|8|Yi30/inc-woq-2282samples-514-g2|

For example, if you want to run DeepSeek-R1-0528, with tp-size 8, you can download measurement files with:
```bash
cd vllm-fork
huggingface-cli download Yi30/ds-r1-0528-default-pile-g2-0529  --local-dir ./scripts/nc_workspace_measure_kvcache
```

2. Configure environment variables.

After downloading measurement files, you need to configure some environment variables to make INC quantization become effective.

##### Using start_vllm.sh script

If you are using `start_vllm.sh` script to start vllm, please configure `QUANT_CONFIG` and `INC_MEASUREMENT_DUMP_PATH_PREFIX` env var in start_vllm.sh

- QUANT_CONFIG

Depends on kv-cache-dtype to use, you should use quantization configuration file accordingly.

These quantization config is located in vllm-fork/scripts/quant_configs.

|KV-Cache-Dtype|QUANT_CONFIG|
|---|---|
|BF16|inc_quant_per_channel_bf16kv.json|
|FP8|inc_quant_per_channel_with_fp8kv_config.json|

For example, if you want to use BF16 kv cache, you should set QUANT_CONFIG with:
```
export QUANT_CONFIG=/path/to/vllm-fork/scripts/quant_configs/inc_quant_per_channel_bf16kv.json
```

- INC_MEASUREMENT_DUMP_PATH_PREFIX

The environment variable `INC_MEASUREMENT_DUMP_PATH_PREFIX` specifies the root directory where measurement statistics were saved.
The final path is constructed by joining this root directory with the `dump_stats_path` defined in the quantization JSON file specified by the `QUANT_CONFIG` environment variable.

If we download the measurements to `/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache`, we got below files:

```bash
user:vllm-fork$ pwd
/path/to/vllm-fork
user:vllm-fork$ ls -l  ./scripts/nc_workspace_measure_kvcache
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

##### Manually start vllm

If you want to start vllm manually or use your own script, please set below environment variables.

|Env Var Name|Mandatory for INC|Value|Explanation|
|---|---|---|---|
|INC_MEASUREMENT_DUMP_PATH_PREFIX|Yes|The root directory where measurement statistics were saved.|See above section for detail|
|QUANT_CONFIG|Yes|Quantization config file to use, which is under `vllm-fork/scripts/quant_configs` folder|See above section for detail|
|VLLM_REQUANT_FP8_INC|Yes|1|Enables requantization of FP8 weights with block-wise scaling using INC.|
|VLLM_ENABLE_RUNTIME_DEQUANT|Yes|1|Enables runtime dequantization of FP8 weights with block-wise scaling.|
|VLLM_MOE_N_SLICE|Yes|1|Specifies the number of slices for the MoE part.|
|VLLM_HPU_MARK_SCALES_AS_CONST|No|false(recommended) or true|Marks the scaling values of the quantized model as constant.|

3. Check if INC quantization enabled successfully

If INC quantization is enabled successfully, `Preparing model with INC` should be observed in vllm server log.

### Launch vLLM Serving with TP=8
```bash
bash start_vllm.sh -w /data/hf_models/DeepSeek-R1-Gaudi -u 0.0.0.0 -p 8688 -b 128 -l 16384 -c /data/warmup_cache
```

It takes more than 1 hour to load and warm up the model for the first time. After completion, a typical output would be like below. The warmup time will be accelerated if the warmup cache is re-used. vLLM server is ready to serve when the log below appears.
```bash
INFO 04-09 00:49:01 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 32.75 seconds
INFO 04-09 00:49:01 api_server.py:800] Using supplied chat template:
INFO 04-09 00:49:01 api_server.py:800] None
INFO 04-09 00:49:01 api_server.py:937] Starting vLLM API server on http://0.0.0.0:8688
INFO 04-09 00:49:01 launcher.py:23] Available routes are:
INFO 04-09 00:49:01 launcher.py:31] Route: /openapi.json, Methods: HEAD, GET
```

### Send Requests to Ensure the Service Functionality

On bare metal, execute the following command to send a request to the Chat Completions API endpoint using `cURL`: 

```bash
curl http://127.0.0.1:8688/v1/chat/completions \
  -X POST \
  -d '{"model": "/data/hf_models/DeepSeek-R1-Gaudi", "messages": [{"role": "user", "content": "List 3 countries and their capitals."}], "max_tokens":128}' \
  -H 'Content-Type: application/json'
```

If it reponses normally, refer to [Check the vLLM Performance](#Check-the-vLLM-performance) and [Check the Model Accuracy](#check-the-model-accuracy) to measure the performance and accuracy.


## Multi-Node Setup and Serving Deployment

vLLM on Gaudi supports multi-node serving. This section uses 2-node and TP16 as an example for environment setup and deployment.

### Identical Software Stack
Ensure both nodes have the same software stack, including:
- Driver: 1.20.1 (how to update Gaudi driver: https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html)
- Firmware: 1.20.1 (how to update Gaudi firmware: https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html#system-unboxing-main)
- Docker: vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
- vLLM branch for DeepSeek-R1 671B: https://github.com/HabanaAI/vllm-fork/tree/deepseek_r1
- vLLM HPU extention for DeepSeek-R1 671B: https://github.com/HabanaAI/vllm-hpu-extension/tree/deepseek_r1

### Network Configuration
- Ensure both nodes are connected to the same switch/router.
- Example IP configuration:
  - Node 1: `192.168.1.101`
  - Node 2: `192.168.1.106`
- For the nodes with both the internal/external network segments, you may also use the external IP address like
  - Node 1: `10.239.129.238`
  - Node 2: `10.239.129.70`


### Start Docker Container Parameters
Use the command below to start the container on both nodes. Assume that the converted model weight files are in the folder /mnt/disk4. Please make sure that the mapped model weight folders are in the same path. 
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all --device=/dev:/dev -v /dev:/dev -v /mnt/disk4:/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --net=host --ipc=host vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

### HCCL Demo Test
Make sure the HCCL demo test passes using the assigned IPs on the two nodes (16 HPU) and get the expected all-reduce throughput. 
HCCL demo guide document: https://github.com/HabanaAI/hccl_demo?tab=readme-ov-file#running-hccl-demo-on-2-servers-16-gaudi-devices
#### Example Commands:
**Head Node:**
```bash
HCCL_COMM_ID=192.168.1.101:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 0 --size 32m --ranks_per_node 8
```

**Worker Node:**
```bash
HCCL_COMM_ID=192.168.1.101:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 1 --size 32m --ranks_per_node 8
```

The expected throughput is below. 
```
#########################################################################################
[BENCHMARK] hcclAllReduce(dataSize=33554432, count=8388608, dtype=float, iterations=1000)
[BENCHMARK]     NW Bandwidth   : 205.899352 GB/s
[BENCHMARK]     Algo Bandwidth : 109.812988 GB/s
#########################################################################################
```


### Install VLLM on Both Nodes
```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-hpu-extension.git
pip install -e vllm-fork/
pip install -e vllm-hpu-extension/
```

### Configure Multi-Node Script
#### Set the IP address and NIC interface name in set_header_node_sh and set_worker_node_sh. 
```bash
# set IP address of header node
export VLLM_HOST_IP=192.168.1.101
# set NIC interface name of worker IP address
export GLOO_SOCKET_IFNAME=enx6c1ff7012f87
```

#### Adjust environment variables if required. Make sure the head node and worker node to have the same configuration except for VLLM_HOST_IP, GLOO_SOCKER_IFNAME and HCCL_SOCKET_IFNAME. 
```bash
# warmup cache folder
export PT_HPU_RECIPE_CACHE_CONFIG=/data/cache/cache_32k_1k_20k_16k,false,32768

# vllm parameters
max_num_batched_tokens=32768
max_num_seqs=512
input_min=768
input_max=20480
output_max=16896
```

#### INC FP8 Quantization

To run DeepSeek-R1 with INC FP8 quantization in multi-nodes case, you need to follow:

1. Download corresponding measurement files to both head and worker node according to target model and tp-size.

|Model|TP-Size|Measurement Files|
|---|---|---|
|DeepSeek-R1-0528|16|Yi30/ds-r1-0528-default-pile-g2-ep16-0610|
|DeepSeek-R1|16|Yi30/ds-r1-default-pile-g2-ep16-0610|

For example, if you want to run DeepSeek-R1-0528, with tp-size 16, you can download measurement files with:
```bash
cd vllm-fork
huggingface-cli download Yi30/ds-r1-0528-default-pile-g2-ep16-0610  --local-dir ./scripts/nc_workspace_measure_kvcache
```

For case running DeepSeek-R1-0528 with tp-size 16, we have downloaded and kept the measurement files of `Yi30/ds-r1-0528-default-pile-g2-ep16-0610` to `scripts/measure_kvcache/ds-r1-0528-g2-tp16`. You can just copy to target folder:
```bash
cd vllm-fork
cp -r ./scripts/measure_kvcache/ds-r1-0528-g2-tp16 ./scripts/nc_workspace_measure_kvcache
```

2. Configure environment variables.

After downloading measurement files, you need to configure some environment variables to make INC quantization become effective.

##### Using start_vllm.sh script

If you are using `set_head_node.sh` and `set_worker_node.sh` scripts to start vllm, please configure `QUANT_CONFIG` and `INC_MEASUREMENT_DUMP_PATH_PREFIX` env var in them.

- QUANT_CONFIG

Depends on kv-cache-dtype to use, you should use quantization configuration file accordingly.

These quantization config is located in vllm-fork/scripts/quant_configs.

|KV-Cache-Dtype|QUANT_CONFIG|
|---|---|
|BF16|inc_quant_per_channel_bf16kv.json|
|FP8|inc_quant_per_channel_with_fp8kv_config.json|

For example, if you want to use BF16 kv cache, you should set QUANT_CONFIG with:
```
export QUANT_CONFIG=/path/to/vllm-fork/scripts/quant_configs/inc_quant_per_channel_bf16kv.json
```

- INC_MEASUREMENT_DUMP_PATH_PREFIX

The environment variable `INC_MEASUREMENT_DUMP_PATH_PREFIX` specifies the root directory where measurement statistics were saved.
The final path is constructed by joining this root directory with the `dump_stats_path` defined in the quantization JSON file specified by the `QUANT_CONFIG` environment variable.

If we download the measurements to `/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache`, we got below files:

```bash
user:vllm-fork$ pwd
/path/to/vllm-fork
user:vllm-fork$ ls -l  ./scripts/nc_workspace_measure_kvcache
-rw-r--r-- 1 root root    1136822 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16.json
-rw-r--r-- 1 root root     611732 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16.npz
-rw-r--r-- 1 root root     155379 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16_mod_list.json
...
```

Then, we export `INC_MEASUREMENT_DUMP_PATH_PREFIX=/path/to/vllm-fork`, and INC will parse the full as below:

```
dump_stats_path (from config): "scripts/nc_workspace_measure_kvcache/inc_measure_output"
Resulting full path: "/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache/inc_measure_output_hooks_maxabs_0_16.npz"
```
##### Manually start vllm

If you want to start vllm manually or use your own script, please set below environment variables.

|Env Var Name|Mandatory for INC|Value|Explanation|
|---|---|---|---|
|INC_MEASUREMENT_DUMP_PATH_PREFIX|Yes|The root directory where measurement statistics were saved.|See above section for detail|
|QUANT_CONFIG|Yes|Quantization config file to use, which is under `vllm-fork/scripts/quant_configs` folder|See above section for detail|
|VLLM_REQUANT_FP8_INC|Yes|1|Enables requantization of FP8 weights with block-wise scaling using INC.|
|VLLM_ENABLE_RUNTIME_DEQUANT|Yes|1|Enables runtime dequantization of FP8 weights with block-wise scaling.|
|VLLM_MOE_N_SLICE|Yes|1|Specifies the number of slices for the MoE part.|
|VLLM_HPU_MARK_SCALES_AS_CONST|No|false(recommended) or true|Marks the scaling values of the quantized model as constant.|

3. Check if INC quantization enabled successfully

If INC quantization is enabled successfully, `Preparing model with INC` should be observed in vllm server log.

#### Apply Configuration on Both Nodes
Run the following command on both head and worker nodes:
header node
```bash
source set_header_node.sh
```
worker node
```bash
source set_worker_node.sh
```


---

### Start Ray Cluster

#### Start Ray on Head Node
ray start --head --node-ip-address=HEADER_NODE_ip --port=PORT
```bash
ray start --head --node-ip-address=192.168.1.101 --port=8850
```

#### Start Ray on Worker Node

ray start --address='HEADER_NODE_IP:port'
```bash
ray start --address='192.168.1.101:8850'
```

If you meet the error message like "ray.exceptions.RaySystemError: System error: No module named 'vllm'", please set the variable below. "/workspace/vllm-fork" is your vLLM source code folder. 
"/workspace/vllm-fork" is your vLLM source code folder and pelase update to your folder path. 
```bash
echo 'PYTHONPATH=$PYTHONPATH:/workspace/vllm-fork' | tee -a /etc/environment
source /etc/environment

```

---

### Start vLLM on Head Node
The example command is like below. It may take several hours to finish the warm up for 32k context length on 2 nodes.
```bash
python -m vllm.entrypoints.openai.api_server \
    --host 192.168.1.101 \
    --port 8688 \
    --model /data/hf_models/DeepSeek-R1-Gaudi \
    --tensor-parallel-size 16 \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-log-requests \
    --dtype bfloat16 \
    --kv-cache-dtype $KV_CACHE_DTYPE \
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

## Check the vLLM Performance
You may check the vLLM performance with benchmark_vllm_client.sh. 
- Login the same container with the command like "docker exec -it deepseek_server /bin/bash"
- Please copy benchmark_vllm_client.sh into the folder "vllm-fork/benchmarks". 
- Update the model path, vLLM server IP and port in the script file if required. 
```bash
model_path=/data/hf_models/DeepSeek-R1-Gaudi
ip_addr=127.0.0.1
port=8688
```

- Execute this scrpt in the folder "vllm-fork/benchmarks". 
```bash
pip install datasets
bash benchmark_vllm_client.sh
```
This script call the standard vLLM benchmark serving tool to check the vLLM throughput. The input and output token length are both 1k. concurrency 1 and 32 are used. 

## Check the Model Accuracy
### Enter the Running Docker Container in the Above Way
### Install lm_eval
```bash
pip install lm_eval[api]
```
### Set Proxy or HF mirror if Required:
```bash
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=127.0.0.1
```
### Run lm_eval
Change the model path, vLLM IP address or port in the command below if required. 
```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=/data/hf_models/DeepSeek-R1-Gaudi,base_url=http://127.0.0.1:8688/v1/completions --batch_size 16 --log_samples --output_path ./lm_eval_output
```


