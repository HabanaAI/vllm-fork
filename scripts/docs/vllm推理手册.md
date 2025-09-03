# Gaudi2E 推理手册 – v1.21 版本

## 1.0 环境部署

### 1.1 BIOS 设置以及操作系统设置

#### 1.1.1 请在 BIOS 里按照服务器或者主板说明书进行如下的设置

- 设置 CPU 为性能模式（performance mode）
- 开启 CPU P-state
- 关闭 CPU C6 状态

#### 1.1.2 进入 Linux OS 后，在主机上设置

- 在 GRUB 里设置 CPU 为性能模式（以 Ubuntu 为例）

打开文件 `/etc/default/grub`  
给变量 `GRUB_CMDLINE_LINUX_DEFAULT` 增加参数 `cpufreq.default_governor=performance`  
例如：

```
GRUB_CMDLINE_LINUX_DEFAULT="cpufreq.default_governor=performance intel_idle.max_cstate=0"
```

执行命令 `update-grub` 使命令生效，然后重启 OS。

查看 CPU 是否是 performance 模式。

```
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

如果输出为 performance，则说明 CPU 已经设置为性能模式。

- 关闭 NUMA balancing

```bash
echo 0 > /proc/sys/kernel/numa_balancing
```

- 设置 hugepages

```bash
sudo sysctl -w vm.nr_hugepages=15000
echo "vm.nr_hugepages=15000" | sudo tee -a /etc/sysctl.conf
```

### 1.2 镜像

#### 1.2.1 基础镜像及网络配置

在 Host 使用如下命令启动最新的容器（以 1.21.3 docker image 为例）：

```bash
docker run -it --name gaudi_server --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice --net=host --ipc=host \
    vault.habana.ai/gaudi-docker/1.21.3/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

若服务器配置了高速互联网卡（如 Mellanox CX6 / CX7）并连接至交换机，需要在容器内安装 libfabric 及 hccl_ofi_wrapper 库来使能 4 卡以上的通信互联。  
进入容器后请参考该链接执行：  
[Host NIC Scale Out Setup](https://github.com/HabanaAI/hccl_demo?tab=readme-ov-file#host-nic-scale-out-setup)

建议将如下内容写入容器 `~/.bashrc` 以自动应用上述通信库：

```bash
export LIBFABRIC_ROOT=/opt/libfabric
export LD_LIBRARY_PATH=$LIBFABRIC_ROOT/lib:$LD_LIBRARY_PATH
```

Gaudi2 通过 HCCL Demo 来验证通信功能：

```bash
cd /root && git clone https://github.com/HabanaAI/hccl_demo.git
cd hccl_demo && make -j
HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 10000 --ranks_per_node 8
```

当出现带宽的结果时则证明多卡间高速互联功能已开启（带宽数值随高速网卡配置变化）。

### 1.3 模型权重文件下载

为容器设置正确的网络或代理设置，确保容器可以正常访问网络（如 github）。  
也可以在容器外下载模型权重文件，然后在启动容器时，把模型权重所在的目录映射进容器。

您可以在 HuggingFace 或 ModelScope 网站上下载需要的模型权重文件。  
例如从 ModelScope 下载 Qwen2-72B 模型权重文件：

```bash
sudo apt install git-lfs
git-lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2-72B-Instruct /models/Qwen2-72B-Instruct
```

### 1.4 安装 vLLM

为容器设置正确的网络设置，确保容器可以正常访问 github。  
使用如下命令在镜像环境安装 vLLM v1.21.0：

```bash
git clone -b aice/v1.21.0 https://github.com/HabanaAI/vllm-fork
pip config set global.index-url  https://mirrors.aliyun.com/pypi/simple/
pip install -r vllm-fork/requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e vllm-fork
```

可选：如果需要使用像 Qwen-VL、GLM-4V 这样的多模态模型，请安装 Pillow-SIMD 来提升性能：

```bash
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

## 2.0 vLLM 配置

### 2.1 环境变量配置

为方便用户部署 LLM 服务，提供了集成环境变量配置和 LLM 在线部署启动的一站式脚本 `start_gaudi_vllm_server.sh`。

进入 `vllm-fork/script`，执行如下命令获取 vLLM Gaudi 服务启动脚本的参数信息：

```bash
bash start_gaudi_vllm_server.sh -h
```

命令输出如下：

```
Start vllm server for a huggingface model on Gaudi.

Syntax: bash start_gaudi_vllm_server.sh <-w> [-n:m:u:p:d:i:o:t:l:b:e:c:r:sfza] [-h]
options:
w  Weights of the model, could be model id in huggingface or local path
n  Number of HPU to use, [1-8], default=1
m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None
u  URL of the server, str, default=127.0.0.1
p  Port number for the server, int, default=30001
d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'
i  Input range, str, format='input_min,input_max', default='4,16384'
o  Output range, str, format='output_min,output_max', default='4,2048'
t  max_num_batched_tokens for vllm, int, default=20480
l  max_model_len for vllm, int, default=20480
b  max_num_seqs for vllm, int, default=32
e  number of scheduler steps, int, default=1
r  reduce warmup graphs, str, format='max_bs, max_seq', default=None
c  Cache HPU recipe to the specified path, str, default=None
s  Skip warmup or not, bool, default=false
f  Enable profiling or not, bool, default=false
z  Disable zero-padding, bool, default=false
a  Disable FusedFSDPA, bool, default=false
h  Help info
```

比较重要的参数包括：

- `-w` 指定模型权重所在目录
- `-n` 指定需要几张 Gaudi 加速卡，该选项映射到模型并行（TP）数。若模型使用 GQA 架构，请确保 n 不超过模型配置中的 `num_key_value_heads` 数值；此外该脚本会针对 MoE 模型自动使能专家并行（EP），EP 数等同于 TP 数。
- `-b` 服务支持的最大并发
- `-i` 指定单个访问请求输入的长度范围，比如 `-i 4,16384` 表示输入长度在 4 个 token 到 16k 个 token。用户可根据实际业务部署需求更改上限值。
- `-o` 指定单个访问请求输出的长度范围，比如 `-o 4,2048` 表示输出长度在 4 个 token 到 2k 个 token。用户可根据实际业务部署需求更改上限值。
- `-l` 指定单个访问请求最大上下文长度，需大于输入+输出的最大值
- `-t` 指定模型运行时单次处理的多个访问请求最大输入总长度
- `-d` 指定模型精度
- `-p` 指定服务的 HTTP 端口
- `-c` 执行模型预热的缓存目录
- `-r` 减少预热 graph 时间, 通过给定热点并发数及热点最大输入长度参数：'max_bs, max_seq' 来打开该功能，默认关闭。具体使用说明参考 [2.2 模型预热](#22-模型预热)。
- `-m` 指定使用卡的 module ID，可以通过命令 `hl-smi -Q index,module_id -f csv` 查询得到，请在指定 module ID 时，尽量确保使用相同 NUMA node 里的 module ID。Module 的 NUMA 信息可以通过命令 `hl-smi topo -c` 查询得到。

服务启动命令和参数配置可参考[第三章节](#30-大模型服务启动示例)。

### 2.2 模型预热

以下启动命令表示启动 vLLM 服务在 Qwen2-72B-Instruct 模型上，使用 Gaudi 的 4 张卡，module ID 分别是 0、1、2、3，输入长度范围是 800 到 1024 token，输出范围是 400 到 512 token，推理精度是 BF16，vLLM 服务侦听在 30001 端口上。

```bash
bash start_gaudi_vllm_server.sh \
    -w "/models/Qwen2-72B-Instruct" \
    -n 4 \
    -m 0,1,2,3 \
    -b 128 \
    -i 800,1024 \
    -o 400,512 \
    -l 8192 \
    -t 8192 \
    -d bfloat16 \
    -p 30001 \
    -c /data/warmup_cache
```

该模型在该配置下大约需要 10 分钟左右预热完成。当出现如下信息则 vLLM 服务可用。

```
INFO 03-25 09:01:25 launcher.py:27 Route: /v1/score, Methods: POST
INFO 03-25 09:01:25 launcher.py:27 Route: /v2/rerank, Methods: POST
INFO 03-25 09:01:25 launcher.py:27 Route: /v2/rerank, Methods: POST
INFO 03-25 09:01:25 launcher.py:27 Route: /invocations, Methods: POST
INFO: Started server process [1167]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:30001 (Press CTRL+C to quit)
```

由于 vLLM 首次预热较为花费时间，推荐您指定预热缓存目录 `-c /data/warmup_cache`，第一次预热的 recipe 文件会保存在该目录中。当后续以相同配置启动 vLLM 服务时，指定相同的预热缓存目录，可以大量地避免 Gaudi 重新编译，节省 vLLM 预热的启动时间。

若对服务启动时间有严格要求，可在第二次带上预热缓存目录的启动任务命令中加上 `-s` 选项来跳过预热阶段，该选项会用部分初始访问请求来做性能预热，可能会观测到轻微的性能损失，在一段预热时间后恢复到正常水平。

在一些线上业务部署 vLLM 的场景中，用户可能对响应延迟要求较高，要求服务能在较短的时间内（如 3 分钟）重启，建议部署时使用 `-r` 选项来减少 graph 预热时间。该选项要求用户根据在线推理业务实际运行情况，给定运行时热点并发数以及热点最长输入（通常小于服务启动时给定的最大并发数和最长输入），服务启动时仅会覆盖这些热点配置进行预热。如已对 QwQ-32B 模型预热果如下配置：输入长度范围是 4 到 38912 token，输出范围是 4 到 2048 token，推理精度是 BF16，最大模型长度为 40960，最大并发数 24，cache 目录指定为 `/data/warmup_cache`，常见的在线业务场景下推理并发数是 8，大部分输入的长度不超过 24000，可使用如下命令来再次启动服务：

```bash
bash start_gaudi_vllm_server.sh \
    -w "/models/QwQ-32B" \
    -n 8 \
    -b 24 \
    -i 4,38912 \
    -o 4,2048 \
    -l 40960 \
    -t 40960 \
    -d bfloat16 \
    -p 30001 \
    -r 8,24000 \
    -c /data/warmup_cache
```

该命令可在较短时间将服务启动完毕，且没有明显性能损失。

## 3.0 大模型服务启动示例

### 3.1 DeepSeek-R1 FP8（8 卡部署）

#### 3.1.1 下载和转换模型权重

对于大于 300B 的 FP8 模型，需要使用 8 卡部署推理服务，请确保 Gaudi2 服务器的高速互联网卡已连接至交换机（参考 1.2.1 章节）。

由于 Gaudi2 采用 torch.float8_e4m3fnuz 格式，DeepSeek-R1 FP8 的模型权重需要在 Gaudi2 服务器上做一次 FP8 格式转换。请确保有 1.5TB 以上的硬盘空间，用于保存下载的原生模型权重和转换以后的模型权重文件。现已支持 DeepSeek-R1 671B 和 DeepSeek-R1 0528，以下以 DeepSeek-R1 0528 为例说明启动步骤。DeepSeek-R1 671B 除了模型权重不同外，其他步骤、参数基本相同。

如下命令在 Host 环境启动容器，假设 `/mnt/disk4` 有足够的硬盘空间用来保存模型权重。请为容器设置正确的网络设置，可以在容器内正常访问互联网资源。

```bash
docker run -it --name deepseek_server --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    --device=/dev:/dev -v /dev:/dev -v /mnt/disk4:/data \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK \
    --ulimit memlock=-1:-1 --net=host --ipc=host \
    vault.habana.ai/gaudi-docker/1.21.3/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

下载模型权重（假设模型权重下载在 `/data/hf_models` 目录）：

```bash
sudo apt install git-lfs
git-lfs install
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-0528.git /data/hf_models/DeepSeek-R1-0528
```

模型权重转换：

```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
pip install torch safetensors numpy --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/convert_for_g2.py -i /data/hf_models/DeepSeek-R1-0528 -o /data/hf_models/DeepSeek-R1-0528-G2
```

模型转换时间大约为 15 分钟。

#### 3.1.2 安装和启动 vLLM

安装 vLLM：

```bash
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-fork.git
git clone -b "deepseek_r1" https://github.com/HabanaAI/vllm-hpu-extension.git
pip install -e vllm-fork
pip install -e vllm-hpu-extension
```

下载模型需要的 measurement 文件  
DeepSeek-R1 0528 和 DeepSeek-R1 671B 需要的文件分别如下表所列，该文件存放于 huggingface 仓库，可通过配置代理来获取：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

| 模型             | Measurement 文件名                   |
| ---------------- | ------------------------------------ |
| DeepSeek-R1-0528 | Yi30/ds-r1-0528-default-pile-g2-0529 |
| DeepSeek-R1 671B | Yi30/inc-woq-2282samples-514-g2      |

例如，下载 DeepSeek-R1-0528 的 measurement 文件命令如下：

```bash
cd vllm-fork
huggingface-cli download Yi30/ds-r1-0528-default-pile-g2-0529  --local-dir ./scripts/nc_workspace_measure_kvcache
```

查看 vLLM 启动参数：

```bash
cd vllm-fork/quickstart
bash start_vllm.sh -h
```

命令输出如下所示：

```
Start vllm server for a huggingface model on Gaudi.

Syntax: bash start_vllm.sh <-w> [-u:p:l:b:c:sq] [-h]
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

您可以使用如下命令启动 vLLM，服务监听在 127.0.0.1:8688 上，支持最大并发 128，16k 上下文长度，预热缓存在 `/data/warmup_cache` 目录。

```bash
bash start_vllm.sh -w /data/hf_models/DeepSeek-R1-0528-G2 -q -u 127.0.0.1 -p 8688 -b 128 -l 16384 -c /data/warmup_cache
```

首次预热启动时间约为一个小时。建议参考 2.2 节模型预热说明，设置 cache 目录存储 recipe 文件。当后续使用同样参数启动服务时，可用 skip_warmup 来跳过预热阶段节省启动时间。推理服务启动完毕后，您可以在镜像环境中发送如下命令，测试服务是否工作正常：

```bash
curl http://127.0.0.1:8688/v1/chat/completions \
  -X POST \
  -d '{"model": "/data/hf_models/DeepSeek-R1-0528-G2", "messages": [{"role": "user", "content": "List 3 countries and their capitals."}], "max_tokens":128}' \
  -H 'Content-Type: application/json'
```

### 3.2 DeepSeek-R1 蒸馏模型

#### 3.2.1 启动容器和下载模型权重

请用如下命令启动容器，假设 `/mnt/disk4` 有足够的硬盘空间用来保存模型权重，或者模型权重已经保存在该目录下。请为容器设置正确的网络设置，可以在容器内正常访问互联网资源。

```bash
docker run -it --name deepseek_r1_distill_server --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -v /mnt/disk4:/models \
    --cap-add=sys_nice --net=host --ipc=host --workdir=/workspace --privileged \
    vault.habana.ai/gaudi-docker/1.21.3/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

下载模型权重（假设模型权重下载在 `/data/hf_models` 目录）：

```bash
sudo apt install git-lfs
git-lfs install
git clone https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B.git /data/hf_models/DeepSeek-R1-Distill-Llama-70B
git clone https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B.git /data/hf_models/DeepSeek-R1-Distill-Qwen-32B
git clone https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B.git /data/hf_models/DeepSeek-R1-Distill-Llama-8B
```

#### 3.2.2 安装和启动 vLLM

安装 vLLM  
[参照 1.4 章节](#14-安装-vllm) 使用 vLLM aice/v1.21.0 版本安装到容器里。

启动 vLLM  
进入启动脚本目录，启动 vLLM。如下命令表示在 module ID 0,1,2,3（`-n 4 -m 0,1,2,3`）上使用 4 卡跑 Deepseek-R1-Distill-Llama-70B 模型，最大支持并发 128（`-b 128`），支持输入 token 长度 800 到 4096（`-i 800,4096`），输出 token 长度范围 800 到 2048（`-o 800,2048`），精度 BF16（`-d bfloat16`），服务端口 30001（`-p 30001`），预热缓存目录 `/data/70B_warmup_cache`（`-c /data/70B_warmup_cache`）：

```bash
cd vllm-fork/scripts
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/DeepSeek-R1-Distill-Llama-70B" \
    -n 4 \
    -m 0,1,2,3 \
    -b 128 \
    -i 800,4096 \
    -o 800,2048 \
    -l 8192 \
    -t 8192 \
    -d bfloat16 \
    -p 30001 \
    -c /data/70B_warmup_cache
```

DeepSeek-R1-Distill-Qwen-32B 模型 2 卡最大并发 32 部署可使用如下命令启动：

```bash
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/DeepSeek-R1-Distill-Qwen-32B" \
    -n 2 \
    -m 0,1 \
    -b 32 \
    -i 800,4096 \
    -o 800,2048 \
    -l 8192 \
    -t 8192 \
    -d bfloat16 \
    -p 30001 \
    -c /data/32B_warmup_cache
```

DeepSeek-R1-Distill-Llama-8B 模型单卡最大并发 32 部署可使用如下命令启动：

```bash
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/DeepSeek-R1-Distill-Llama-8B" \
    -n 1 \
    -m 0 \
    -b 32 \
    -i 800,4096 \
    -o 800,2048 \
    -l 8192 \
    -t 8192 \
    -d bfloat16 \
    -p 30001 \
    -c /data/8B_warmup_cache
```
