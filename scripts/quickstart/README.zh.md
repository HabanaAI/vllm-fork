# 在 deepseek_r1 分支上运行 DeepseekV3ForCausalLM 架构模型

本指南提供了在英特尔® Gaudi® HPU 上使用 vLLM 服务框架部署和运行 DeepseekV3ForCausalLM 架构模型的分步说明。它涵盖了硬件要求、软件先决条件、模型权重下载和转换、环境设置、模型服务部署以及在单节点和多节点 8*Gaudi 服务器上的性能和精度基准测试。

已验证模型：
- deepseek-ai/DeepSeek-R1-0528
- deepseek-ai/DeepSeek-R1
- moonshotai/Kimi-K2-Instruct
- deepseek-ai/DeepSeek-V3.1

## 目录

- [在 deepseek_r1 分支上运行 DeepseekV3ForCausalLM 架构模型](#在-deepseek_r1-分支上运行-deepseekv3forcausallm-架构模型)
  - [目录](#目录)
  - [硬件要求](#硬件要求)
  - [软件先决条件](#软件先决条件)
  - [模型权重下载与转换](#模型权重下载与转换)
    - [在 Gaudi 服务器上启动 Docker 容器](#在-gaudi-服务器上启动-docker-容器)
    - [下载原始模型](#下载原始模型)
    - [转换模型](#转换模型)
  - [单节点设置与服务部署](#单节点设置与服务部署)
    - [下载并安装 vLLM](#下载并安装-vllm)
    - [HCCL-Demo 测试](#hccl-demo-测试)
    - [INC FP8 量化](#inc-fp8-量化)
    - [启动 vLLM 服务器脚本的参数](#启动-vllm-服务器脚本的参数)
    - [以 TP=8 启动 vLLM 服务](#以-tp8-启动-vllm-服务)
    - [发送请求以确保服务功能正常](#发送请求以确保服务功能正常)
  - [多节点设置与服务部署](#多节点设置与服务部署)
    - [一致的软件栈](#一致的软件栈)
    - [网络配置](#网络配置)
    - [启动 Docker 容器参数](#启动-docker-容器参数)
    - [HCCL demo 测试](#hccl-demo-测试-1)
    - [在两个节点上安装 vLLM](#在两个节点上安装-vllm)
    - [配置多节点脚本](#配置多节点脚本)
    - [启动 Ray 集群](#启动-ray-集群)
    - [在头节点上启动 vLLM](#在头节点上启动-vllm)
  - [检查 vLLM 性能](#检查-vllm-性能)
  - [检查模型精度](#检查模型精度)
    - [按上述方式进入正在运行的 Docker 容器](#按上述方式进入正在运行的-docker-容器)
    - [安装 lm\_eval](#安装-lm_eval)
    - [如果需要，设置代理或 HF 镜像](#如果需要设置代理或-hf-镜像)
    - [运行 lm\_eval](#运行-lm_eval)

## 硬件要求

* DeepSeek-R1-0528、DeepSeek-R1 或 DeepSeek-V3.1

  * DeepSeek-R1-0528 、 DeepSeek-R1 或 DeepSeek-V3.1 拥有 671B 参数，采用 FP8 精度，约占 642GB 内存。单节点 8*Gaudi2 OAM（总共 768GB 内存）足以容纳模型权重和有限上下文长度（<=32k）所需的 KV 缓存。

  * 为支持更高的并发性和更长的令牌长度，推荐使用 2 节点 8*Gaudi2 服务器。

* Kimi-K2-Instruct

  * Kimi-K2-Instruct 拥有 1T 参数，采用 FP8 精度，需要 2 节点 8*Gaudi2 服务器来容纳模型权重。

下表概述了每个节点的每个硬件组件为实现高性能推理所需的最低要求。

| 模型                                       | 服务器         | 每节点 CPU                                           | 每节点加速器           | 每节点 RAM    | 每节点存储                                                                 | 每节点前端网络 <br>（带内管理/存储）                                   | 每节点后端网络 <br>（计算，带 RDMA）                                                                                     |
| :----------------------------------------- | :------------- | :-------------------------------------------------- | :--------------------- | :------------ | :------------------------------------------------------------------------- | :--------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| DeepSeek-R1-0528/DeepSeek-R1/DeepSeek-V3.1 | 1 节点 Gaudi2D | 2\* 第三代或更新代英特尔® 至强® 可扩展处理器          | 8\* HL-225D 96GB OAM   | 最低 1.5TB    | **操作系统:** 至少 480GB SATA/SAS/NVMe SSD, <br> **数据:** 至少 2TB NVMe SSD | 至少 1\* 10GbE/25GbE NIC <br> 或 1\* NVIDIA® 200G BlueField-2 DPU/ConnectX-6 Dx SmartNIC | 不需要                                                                                                                   |
| DeepSeek-R1-0528/DeepSeek-R1/DeepSeek-V3.1 | 2 节点 Gaudi2D | 2\* 第三代或第四代英特尔® 至强® 可扩展处理器         | 8\* HL-225D 96GB OAM   | 最低 1.5TB    | **操作系统:** 至少 480GB SATA/SAS/NVMe SSD, <br> **数据:** 至少 2TB NVMe SSD | 至少 1\* 10GbE/25GbE NIC <br> 或 1\* NVIDIA® 200G BlueField-2 DPU/ConnectX-6 Dx SmartNIC | 4\* 或 8\* NVIDIA® HDR-200G ConnectX-6 Dx SmartNIC/HCA 或 NDR-400G ConnectX-7 SmartNIC/HCA                                |
| Kimi-K2-Instruct                           | 2 节点 Gaudi2D | 2\* 第三代或第四代英特尔® 至强® 可扩展处理器         | 8\* HL-225D 96GB OAM   | 最低 1.5TB    | **操作系统:** 至少 480GB SATA/SAS/NVMe SSD, <br> **数据:** 至少 2TB NVMe SSD | 至少 1\* 10GbE/25GbE NIC <br> 或 1\* NVIDIA® 200G BlueField-2 DPU/ConnectX-6 Dx SmartNIC | 4\* 或 8\* NVIDIA® HDR-200G ConnectX-6 Dx SmartNIC/HCA 或 NDR-400G ConnectX-7 SmartNIC/HCA                                |

### 将 CPU 设置为性能模式
请在 BIOS 设置中将 CPU 设置更改为性能优化模式，并在操作系统中执行以下命令以确保获得最佳 CPU 性能。

sudo echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor


## 软件先决条件

* 本指南以在 Ubuntu 22.04 LTS 上部署为例。

* 参考 [在 Ubuntu 上安装 Docker Engine](https://docs.docker.com/engine/install/ubuntu/) 在每个节点上安装 Docker。

* 参考 [驱动程序和软件安装](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) 在每个节点上安装 Gaudi® 驱动程序和软件栈（>= 1.20.1）。确保安装了 `habanalabs-container-runtime`。

* 参考 [固件升级](https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html) 在每个节点上将 Gaudi® 固件升级到 >=1.20.1 版本。

* 参考 [配置容器运行时](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#configure-container-runtime) 在每个节点上配置 `habana` 容器运行时。

## 模型权重下载与转换

### 在 Gaudi 服务器上启动 Docker 容器
假设原始模型权重文件在文件夹 /mnt/disk4 中下载和转换，该文件夹对于 DeepSeek-R1-0528、DeepSeek-R1 或 DeepSeek-V3.1 应至少有 1.5TB 磁盘空间，对于 Kimi-K2-Instruct 应至少有 2TB 磁盘空间。
> [!NOTE]
> * 确保拉取的 docker 镜像与相应的 Gaudi 驱动程序和操作系统版本对齐。本指南中使用的默认镜像是针对 Gaudi 驱动程序/固件 1.20.1 和 Ubuntu 22.04 的，其他镜像请参考 [使用英特尔(R)Gaudi 容器](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#use-intel-gaudi-containers)。

```bash
docker run -it --name deepseek_server --runtime=habana -e HABANA_VISIBLE_DEVICES=all --device=/dev:/dev -v /dev:/dev -v /mnt/disk4:/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --net=host --ipc=host vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```


### 下载原始模型

原始的 DeepSeek-R1 模型可在 [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1) 和 [ModelScope](https://www.modelscope.cn/deepseek-ai/DeepSeek-R1) 上获取。我们假设模型已下载到文件夹 "/data/hf_models/" 中。

```bash
sudo apt install git-lfs
git-lfs install

# 选项1：从 HuggingFace 下载 DeepSeek-R1
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1 /data/hf_models/DeepSeek-R1
# 选项2：从 ModelScope 下载 DeepSeek-R1
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1 /data/hf_models/DeepSeek-R1

#选项1：从 HuggingFace 下载 DeepSeek-R1-0528
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528 /data/hf_models/DeepSeek-R1-0528
#选项2：从 ModelScope 下载 DeepSeek-R1-0528
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-0528.git /data/hf_models/DeepSeek-R1-0528

#选项1：从 HuggingFace 下载 Kimi-K2-Instruct
git clone https://huggingface.co/moonshotai/Kimi-K2-Instruct /data/hf_models/Kimi-K2-Instruct
#选项2：从 ModelScope 下载 Kimi-K2-Instruct
git clone https://www.modelscope.cn/moonshotai/Kimi-K2-Instruct.git /data/hf_models/Kimi-K2-Instruct

#选项1：从 HuggingFace 下载 DeepSeek-V3.1
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3.1.git /data/hf_models/DeepSeek-V3.1
#选项2：从 ModelScope 下载 DeepSeek-V3.1
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-V3.1.git /data/hf_models/DeepSeek-V3.1
```


### 转换模型
要在 Gaudi2D 上服务 DeepSeek-R1 模型，应使用以下命令在 Gaudi 服务器上转换原始的 HuggingFace FP8 模型权重。我们假设原始模型已下载到 /data/hf_models/DeepSeek-R1 文件夹中，转换后的模型将保存到 /data/hf_models/DeepSeek-R1-G2 文件夹中。请确保新文件夹有足够的磁盘空间（对于 DeepSeek-R1 >650GB）。如果磁盘 I/O 足够快，完成转换大约需要 15 分钟。

convert_for_g2.py 的 `-i` 选项指定原始模型权重的路径，`-o` 选项指定输出文件夹。请不要在输入或输出路径的末尾添加 `/`。

```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
pip install torch safetensors numpy --extra-index-url https://download.pytorch.org/whl/cpu

# 转换 DeepSeek-R1
python scripts/convert_for_g2.py -i /data/hf_models/DeepSeek-R1 -o /data/hf_models/DeepSeek-R1-G2

# 转换 DeepSeek-R1-0528
python scripts/convert_for_g2.py -i /data/hf_models/DeepSeek-R1-0528 -o /data/hf_models/DeepSeek-R1-0528-G2

#转换 Kimi-K2-Instruct
python scripts/convert_for_g2.py -i /data/hf_models/Kimi-K2-Instruct -o /data/hf_models/Kimi-K2-Instruct-G2

#转换 DeepSeek-V3.1
python scripts/convert_for_g2.py -i /data/hf_models/DeepSeek-V3.1 -o /data/hf_models/DeepSeek-V3.1-G2
```

当显示以下消息时，转换完成。转换后的模型权重文件保存在您指定的文件夹中，例如 /data/hf_models/DeepSeek-R1-G2，我们将使用此转换后的模型来托管 vLLM 服务。

```bash
...
processing /data/hf_models/DeepSeek-R1/model-00163-of-000163.safetensors
skip model.layers.61.embed_tokens.weight.
skip model.layers.61.enorm.weight.
skip model.layers.61.hnorm.weight.
skip model.layers.61.input_layernorm.weight.
skip model.layers.61.post_attention_layernorm.weight.
skip model.layers.61.shared_head.head.weight.
skip model.layers.61.shared_head.norm.weight.
saving to /data/hf_models/DeepSeek-R1-G2/model-00163-of-000163.safetensors
```

## 单节点设置与服务部署
### 下载并安装 vLLM
在转换模型权重的同一容器中，克隆最新代码并安装。
```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
pip install -e vllm-fork/
```

### HCCL demo 测试
下载 HCCL demo，编译并执行 hccl_demo 测试。确保 HCCL Demo 测试在 8 个 HPU 上通过。详细信息，请参考 [HCCL Demo](https://github.com/HabanaAI/hccl_demo)
```bash
git clone https://github.com/HabanaAI/hccl_demo.git
cd hccl_demo
make
HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1000 --ranks_per_node 8
```


如果显示以下消息，则 hccl 测试通过。对于没有主机 NIC 横向扩展的 Gaudi PCIe 系统，通信需要通过 CPU UPI，NW 带宽应约为 18GB/s。
```bash
[BENCHMARK] hcclAllReduce(dataSize=33554432, count=8388608, dtype=float, iterations=1000)

[BENCHMARK]     NW Bandwidth   : 258.259144 GB/s
[BENCHMARK]     Algo Bandwidth : 147.576654 GB/s
```

### INC FP8 量化

要在单节点情况下使用 INC FP8 量化运行 DeepSeek-R1，您需要遵循：

#### 1. 根据目标模型和 tp-size 下载相应的测量文件。

|模型|TP-Size|测量文件|
|---|---|---|
|DeepSeek-R1-0528|8|Yi30/ds-r1-0528-default-pile-g2-0529|
|DeepSeek-R1|8|Yi30/inc-woq-2282samples-514-g2|

例如，如果您想运行 DeepSeek-R1-0528，tp-size 为 8，您可以使用以下命令下载测量文件：
```bash
cd vllm-fork
huggingface-cli download Yi30/ds-r1-0528-default-pile-g2-0529  --local-dir ./scripts/nc_workspace_measure_kvcache
```

##### 1.1 校准 DeepSeek-V3.1 模型
对于 DeepSeek-V3.1，请使用以下命令校准模型。命令完成后，DeepSeek-V3.1 测量文件将在文件夹 "scripts/nc_workspace_measure_kvcache" 中生成。
```bash
cd vllm-fork
bash scripts/run_inc_calib.sh --model /data/hf_models/DeepSeek-V3.1-G2
```

#### 2. 配置环境变量（可选）

下载测量文件后，您需要配置一些环境变量以使 INC 量化生效。

##### 2.1 使用 start_vllm.sh 脚本

如果您想使用 `start_vllm.sh` 脚本启动 vllm，计划使用 FP8 kv 缓存，并且已将测量文件下载到推荐路径（/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache），您可以跳过本节，直接使用默认的环境变量值。

如果您想使用自定义路径或使用 BF16 kv 缓存，请在 start_vllm.sh 中重新配置 `QUANT_CONFIG` 和 `INC_MEASUREMENT_DUMP_PATH_PREFIX` 环境变量。

- QUANT_CONFIG

根据要使用的 kv-cache-dtype，您应相应地使用量化配置文件。

这些量化配置位于 vllm-fork/scripts/quant_configs。

|KV-Cache-Dtype|vLLM --kv-cache-dtype 参数|QUANT_CONFIG|
|---|---|---|
|BF16|auto|inc_quant_per_channel_bf16kv.json|
|FP8|fp8_inc|inc_quant_per_channel_with_fp8kv_config.json|

例如，如果您想使用 FP8 kv 缓存，您应该设置 QUANT_CONFIG：

export QUANT_CONFIG=/path/to/vllm-fork/scripts/quant_configs/inc_quant_per_channel_with_fp8kv_config.json

并将 "fp8_inc" 传递给 vLLM --kv-cache-dtype 参数。

- INC_MEASUREMENT_DUMP_PATH_PREFIX

环境变量 `INC_MEASUREMENT_DUMP_PATH_PREFIX` 指定保存测量统计信息的根目录。
最终路径是通过将此根目录与由 `QUANT_CONFIG` 环境变量指定的量化 JSON 文件中定义的 `dump_stats_path` 连接起来构建的。

如果我们将测量下载到 `/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache`，我们会得到以下文件：

```bash
user:vllm-fork$ pwd
/path/to/vllm-fork
user:vllm-fork$ ls -l  ./scripts/nc_workspace_measure_kvcache
-rw-r--r-- 1 user Software-SG 1949230 May 15 08:05 inc_measure_output_hooks_maxabs_0_8.json
-rw-r--r-- 1 user Software-SG  254451 May 15 08:05 inc_measure_output_hooks_maxabs_0_8_mod_list.json
-rw-r--r-- 1 user Software-SG 1044888 May 15 08:05 inc_measure_output_hooks_maxabs_0_8.npz
...
```

然后，我们导出 `INC_MEASUREMENT_DUMP_PATH_PREFIX=/path/to/vllm-fork`，INC 将解析完整路径如下：


dump_stats_path (来自配置): "scripts/nc_workspace_measure_kvcache/inc_measure_output"
结果完整路径: "/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache/inc_measure_output_hooks_maxabs_0_8.npz"


##### 2.2 手动启动 vllm

如果您想手动启动 vllm 或使用自己的脚本，请设置以下环境变量。

|环境变量名称|INC 是否必需|值|解释|
|---|---|---|---|
|INC_MEASUREMENT_DUMP_PATH_PREFIX|是|保存测量统计信息的根目录。|详见上文|
|QUANT_CONFIG|是|要使用的量化配置文件，位于 `vllm-fork/scripts/quant_configs` 文件夹下|详见上文|
|VLLM_REQUANT_FP8_INC|是|1|启用使用 INC 对 FP8 权重进行块级缩放的重新量化。|
|VLLM_ENABLE_RUNTIME_DEQUANT|是|1|启用对 FP8 权重进行块级缩放的运行时反量化。|
|VLLM_MOE_N_SLICE|是|1|指定 MoE 部分的切片数量。|
|INC_FORCE_NAIVE_SCALING|是|1|设置为 1 时，INC 将使用朴素缩放，这可以有更好的精度。如果设置为 0，INC 将使用硬件对齐的缩放，这具有更好的性能但精度较差。|
|VLLM_HPU_MARK_SCALES_AS_CONST|否|false（推荐）或 true|将量化模型的缩放值标记为常量。|

#### 3. 检查 INC 量化是否成功启用

如果 INC 量化成功启用，应在 vllm 服务器日志中观察到 `Preparing model with INC`。

### 启动 vLLM 服务器脚本的参数
有一些系统环境变量需要设置以获得最佳 vLLM 性能。我们提供了示例脚本来设置推荐的环境变量。

脚本文件 "start_vllm.sh" 用于启动 vLLM 服务。您可以执行以下命令检查其支持的参数。
```bash
bash start_vllm.sh -h
```

命令输出如下所示。
```bash
在 Gaudi 上为 huggingface 模型启动 vllm 服务器。

语法: bash start_vllm.sh <-w> [-u:p:l:b:c:sq] [-h]
选项:
w  模型的权重，可以是 huggingface 中的模型 ID 或本地路径
u  服务器的 URL，字符串，默认=0.0.0.0
p  服务器的端口号，整数，默认=8688
l  vllm 的 max_model_len，整数，默认=16384，单节点最大值：32768
b  vllm 的 max_num_seqs，整数，默认=128
c  将 HPU 配方缓存到指定路径，字符串，默认=None
s  是否跳过预热，布尔值，默认=false
q  启用 inc fp8 量化
h  帮助信息
```

### 以 TP=8 启动 vLLM 服务
```bash
bash start_vllm.sh -w /data/hf_models/DeepSeek-R1-G2 -q -u 0.0.0.0 -p 8688 -b 128 -l 16384 -c /data/warmup_cache
```

注意：对于 DeepSeek-V3.1，如果客户端使用非思维模式，请删除文件 "start_vllm.sh" 中的参数 "--enable-reasoning --reasoning-parser deepseek_r1"。

首次加载和预热模型需要超过 1 小时。完成后，典型输出如下所示。如果重用预热缓存，预热时间将加快。当出现以下日志时，vLLM 服务器已准备好服务。
```bash
INFO 04-09 00:49:01 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 32.75 seconds
INFO 04-09 00:49:01 api_server.py:800] Using supplied chat template:
INFO 04-09 00:49:01 api_server.py:800] None
INFO 04-09 00:49:01 api_server.py:937] Starting vLLM API server on http://0.0.0.0:8688
INFO 04-09 00:49:01 launcher.py:23] Available routes are:
INFO 04-09 00:49:01 launcher.py:31] Route: /openapi.json, Methods: HEAD, GET
```

### 发送请求以确保服务功能正常

在裸机上，执行以下命令使用 `cURL` 向 Chat Completions API 端点发送请求：

```bash
curl http://127.0.0.1:8688/v1/chat/completions \
  -X POST \
  -d '{"model": "/data/hf_models/DeepSeek-R1-G2", "messages": [{"role": "user", "content": "列出 3 个国家和它们的首都。"}], "max_tokens":128}' \
  -H 'Content-Type: application/json'
```

如果响应正常，请参考 [检查 vLLM 性能](#检查-vllm-性能) 和 [检查模型精度](#检查模型精度) 来测量性能和精度。

## 多节点设置与服务部署

vLLM on Gaudi 支持多节点服务。本节以 2 节点和 TP16 为例进行环境设置和部署。

### 一致的软件栈
确保两个节点具有相同的软件栈，包括：
- 驱动程序：1.20.1（如何更新 Gaudi 驱动程序：https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html）
- 固件：1.20.1（如何更新 Gaudi 固件：https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html#system-unboxing-main）
- Docker：vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
- 用于 DeepSeek-R1 671B 的 vLLM 分支：https://github.com/HabanaAI/vllm-fork/tree/deepseek_r1
- 用于 DeepSeek-R1 671B 的 vLLM HPU 扩展：https://github.com/HabanaAI/vllm-hpu-extension/tree/deepseek_r1

### 网络配置
- 确保两个节点连接到相同的交换机/路由器。
- 示例 IP 配置：
  - 节点 1：`192.168.1.101`
  - 节点 2：`192.168.1.106`
- 对于具有内部/外部网络段的节点，您也可以使用外部 IP 地址，例如
  - 节点 1：`10.239.129.238`
  - 节点 2：`10.239.129.70`

### 启动 Docker 容器参数
在两个节点上使用以下命令启动容器。假设转换后的模型权重文件在文件夹 /mnt/disk4 中。请确保映射的模型权重文件夹位于同一路径中。
```bash
docker run -it --runtime=habana -e HABana_VISIBLE_DEVICES=all --device=/dev:/dev -v /dev:/dev -v /mnt/disk4:/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --net=host --ipc=host vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

### HCCL demo 测试
确保 HCCL demo 测试使用指定的 IP 在两个节点（16 HPU）上通过，并获得预期的 all-reduce 吞吐量。
HCCL demo 指南文档：https://github.com/HabanaAI/hccl_demo?tab=readme-ov-file#running-hccl-demo-on-2-servers-16-gaudi-devices
#### 示例命令：
**头节点：**
```bash
HCCL_COMM_ID=192.168.1.101:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 0 --size 32m --ranks_per_node 8
```

**工作节点：**
```bash
HCCL_COMM_ID=192.168.1.101:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 1 --size 32m --ranks_per_node 8
```

预期吞吐量如下。
```
[BENCHMARK] hcclAllReduce(dataSize=33554432, count=8388608, dtype=float, iterations=1000)

[BENCHMARK]     NW Bandwidth   : 205.899352 GB/s
[BENCHMARK]     Algo Bandwidth : 109.812988 GB/s
```

### 在两个节点上安装 VLLM
```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
pip install -e vllm-fork/
```

### 配置多节点脚本
#### 在 set_head_node_sh 和 set_worker_node_sh 中设置 IP 地址和 NIC 接口名称。
```bash
#设置头节点的 IP 地址
export VLLM_HOST_IP=192.168.1.101
#设置头节点 IP 地址的 NIC 接口名称
export GLOO_SOCKET_IFNAME=enx6c1ff7012f87
```

#### 如果需要，调整环境变量。确保头节点和工作节点具有相同的配置，除了 VLLM_HOST_IP、GLOO_SOCKER_IFNAME 和 HCCL_SOCKET_IFNAME。
```bash
#预热缓存文件夹
export PT_HPU_RECIPE_CACHE_CONFIG=/data/cache/cache_32k_1k_20k_16k,false,32768

# vllm 参数
max_num_batched_tokens=32768
max_num_seqs=512
input_min=768
input_max=20480
output_max=16896
```


#### INC FP8 量化

要在多节点情况下使用 INC FP8 量化运行 DeepSeek-R1，您需要遵循：

##### 1. 根据目标模型和 tp-size 将相应的测量文件下载到头节点和工作节点。

|模型|TP-Size|测量文件|
|---|---|---|
|DeepSeek-R1-0528|16|Yi30/ds-r1-0528-default-pile-g2-ep16-0610|
|DeepSeek-R1|16|Yi30/ds-r1-default-pile-g2-ep16-0610|
|Kimi-K2-Instruct|16|Yi30/miki-k2-pile-g2-tp16-2nd-0717|

例如，如果您想运行 DeepSeek-R1-0528，tp-size 为 16，您可以使用以下命令下载测量文件：
```bash
cd vllm-fork
huggingface-cli download Yi30/ds-r1-0528-default-pile-g2-ep16-0610  --local-dir ./scripts/nc_workspace_measure_kvcache
```


对于使用 tp-size 16 运行 DeepSeek-R1-0528 的情况，我们已经下载并将 `Yi30/ds-r1-0528-default-pile-g2-ep16-0610` 的测量文件保存到 `scripts/measure_kvcache/ds-r1-0528-g2-tp16`。您可以复制到目标文件夹：
```bash
cd vllm-fork
cp -r ./scripts/measure_kvcache/ds-r1-0528-g2-tp16 ./scripts/nc_workspace_measure_kvcache
```


###### 1.1 根据目标模型校准 DeepSeek-V3.1 模型到头节点和工作节点。
对于 DeepSeek-V3.1，请使用以下命令校准模型。命令完成后，DeepSeek-V3.1 测量文件将在文件夹 "vllm-fork/scripts/nc_workspace_measure_kvcache" 中生成。生成测量文件后，您可以将它们复制到其他工作节点的文件夹 vllm-fork/scripts/nc_workspace_measure_kvcache" 中。
```bash
cd vllm-fork
bash scripts/run_inc_calib.sh --model /data/hf_models/DeepSeek-V3.1-G2
```


##### 2. 配置环境变量。

下载测量文件后，您需要配置一些环境变量以使 INC 量化生效。

###### 2.1 使用 set_head_node.sh & set_worker_node.sh 脚本

如果您使用 `set_head_node.sh` 和 `set_worker_node.sh` 脚本启动 vllm，请在它们中配置 `QUANT_CONFIG` 和 `INC_MEASUREMENT_DUMP_PATH_PREFIX` 环境变量。

- QUANT_CONFIG

根据要使用的 kv-cache-dtype，您应相应地使用量化配置文件。

这些量化配置位于 vllm-fork/scripts/quant_configs。

|KV-Cache-Dtype|vLLM --kv-cache-dtype 参数|QUANT_CONFIG|
|---|---|---|
|BF16|auto|inc_quant_per_channel_bf16kv.json|
|FP8|fp8_inc|inc_quant_per_channel_with_fp8kv_config.json|

例如，如果您想使用 BF16 kv 缓存，您应该设置 QUANT_CONFIG：

export QUANT_CONFIG=/path/to/vllm-fork/scripts/quant_configs/inc_quant_per_channel_bf16kv.json

并将 "auto" 传递给 vLLM --kv-cache-dtype 参数。

- INC_MEASUREMENT_DUMP_PATH_PREFIX

环境变量 `INC_MEASUREMENT_DUMP_PATH_PREFIX` 指定保存测量统计信息的根目录。
最终路径是通过将此根目录与由 `QUANT_CONFIG` 环境变量指定的量化 JSON 文件中定义的 `dump_stats_path` 连接起来构建的。

如果我们将测量下载到 `/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache`，我们会得到以下文件：

```bash
user:vllm-fork$ pwd
/path/to/vllm-fork
user:vllm-fork$ ls -l  ./scripts/nc_workspace_measure_kvcache
-rw-r--r-- 1 root root    1136822 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16.json
-rw-r--r-- 1 root root     611732 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16.npz
-rw-r--r-- 1 root root     155379 Jul  4 13:30 inc_measure_output_hooks_maxabs_0_16_mod_list.json
```


然后，我们导出 `INC_MEASUREMENT_DUMP_PATH_PREFIX=/path/to/vllm-fork`，INC 将解析完整路径如下：


dump_stats_path (来自配置): "scripts/nc_workspace_measure_kvcache/inc_measure_output"
结果完整路径: "/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache/inc_measure_output_hooks_maxabs_0_16.npz"

###### 2.2 手动启动 vllm

如果您想手动启动 vllm 或使用自己的脚本，请设置以下环境变量。

|环境变量名称|INC 是否必需|值|解释|
|---|---|---|---|
|INC_MEASUREMENT_DUMP_PATH_PREFIX|是|保存测量统计信息的根目录。|详见上文|
|QUANT_CONFIG|是|要使用的量化配置文件，位于 `vllm-fork/scripts/quant_configs` 文件夹下|详见上文|
|VLLM_REQUANT_FP8_INC|是|1|启用使用 INC 对 FP8 权重进行块级缩放的重新量化。|
|VLLM_ENABLE_RUNTIME_DEQUANT|是|1|启用对 FP8 权重进行块级缩放的运行时反量化。|
|VLLM_MOE_N_SLICE|是|1|指定 MoE 部分的切片数量。|
|INC_FORCE_NAIVE_SCALING|是|1|设置为 1 时，INC 将使用朴素缩放，这可以有更好的精度。如果设置为 0，INC 将使用硬件对齐的缩放，这具有更好的性能但精度较差。|
|VLLM_HPU_MARK_SCALES_AS_CONST|否|false（推荐）或 true|将量化模型的缩放值标记为常量。|

##### 3. 检查 INC 量化是否成功启用

如果 INC 量化成功启用，应在 vllm 服务器日志中观察到 `Preparing model with INC`。

#### 在两个节点上应用配置
在头节点和工作节点上运行以下命令：
头节点
```bash
source set_head_node.sh
```

工作节点
```bash
source set_worker_node.sh
```

---

### 启动 Ray 集群

#### 在头节点上启动 Ray
ray start --head --node-ip-address=头节点_ip --port=端口
```bash
ray start --head --node-ip-address=192.168.1.101 --port=8850
```


#### 在工作节点上启动 Ray

ray start --address='头节点_IP:端口'
```bash
ray start --address='192.168.1.101:8850'
```


如果您遇到错误消息如 "ray.exceptions.RaySystemError: System error: No module named 'vllm'"，请设置以下变量。这里假设"/workspace/vllm-fork" 是您的 vLLM 源代码文件夹，如果不是，请更新为您的文件夹路径。
```bash
echo 'PYTHONPATH=$PYTHONPATH:/workspace/vllm-fork' | tee -a /etc/environment
source /etc/environment
```


### 在头节点上启动 vLLM
示例命令如下所示。在 2 个节点上完成 32k 上下文长度的预热可能需要几个小时。
```bash
python -m vllm.entrypoints.openai.api_server \
    --host 192.168.1.101 \
    --port 8688 \
    --model /data/hf_models/DeepSeek-R1-G2 \
    --tensor-parallel-size 16 \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-log-requests \
    --dtype bfloat16 \
    --kv-cache-dtype $KV_CACHE_DTYPE \
    --use-v2-block-manager \
    --num-scheduler-steps 1\
    --block-size $block_size \
    --max-model-len $max_num_batched_tokens \
    --distributed-executor-backend ray \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
```

## 检查 vLLM 性能
您可以使用 benchmark_vllm_client.sh 检查 vLLM 性能。
- 使用类似 "docker exec -it deepseek_server /bin/bash" 的命令登录同一容器。
- 请将 benchmark_vllm_client.sh 复制到文件夹 "vllm-fork/benchmarks" 中。
- 如果需要，在脚本文件中更新模型路径、vLLM 服务器 IP 和端口。
```bash
model_path=/data/hf_models/DeepSeek-R1-G2
ip_addr=127.0.0.1
port=8688
```

- 在文件夹 "vllm-fork/benchmarks" 中执行此脚本。
```bash
pip install datasets
bash benchmark_vllm_client.sh
```
此脚本调用标准的 vLLM 基准服务工具来检查 vLLM 吞吐量。输入和输出令牌长度均为 1k。使用了并发数 1 和 32。

## 检查模型精度
### 按上述方式进入正在运行的 Docker 容器
### 安装 lm_eval
```bash
pip install lm_eval[api]
```

### 如果需要，设置代理或 HF 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=127.0.0.1
```

### 运行 lm_eval
如果需要，更改以下命令中的模型路径、vLLM IP 地址或端口。
```bash
lm_eval --model local-completions --tasks gsm8k --model_args model=/data/hf_models/DeepSeek-R1-G2,max_gen_toks=4096,max_length=16384,base_url=http://127.0.0.1:8688/v1/completions --batch_size 16 --log_samples --output_path ./lm_eval_output
```