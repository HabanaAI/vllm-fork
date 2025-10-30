# Gaudi2E 推理手册 – v1.22 版本

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
git clone -b aice/v1.22.0 https://github.com/HabanaAI/vllm-fork
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -r vllm-fork/requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e vllm-fork --no-build-isolation
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
Start a vLLM server for a huggingface model on Gaudi.

Usage: bash start_gaudi_vllm_server.sh <-w> [-t:m:a:d:q:i:p:o:b:g:u:e:l:c:sf] [-h]
Options:
-w  Weights of the model, str, could be model id in huggingface or local path.
    DO NOT change the model name as some of the parameters depend on it.
-t  tensor-parallel-size for vLLM, int, default=1.
    Also used to set EP size if it's enable by --enable-expert-parallel
-m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None
    Used to select HPUs and to set NUMA accordingly. It's recommended to set
    for cases with 4 or less HPUs.
-a  API server URL, str, 'IP:PORT', default=127.0.0.1:30001
-d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'
    Set to 'fp8' if -q or environment variable 'QUANT_CONFIG' is provided.
-q  Path to the quantization config file, str, default=None
    default=./quantization/<model_name_lower>/maxabs_quant_g2.json for -d 'fp8'
    The environment variable 'QUANT_CONFIG' will override this option.
-i  Input range, str, format='input_min,input_max', default='4,1024'
    Make sure the range cover all the possible lengths from the benchmark/client.
-p  Max number of the prefill sequences, int, default=1
    Used to control the max batch size for prefill to balance the TTFT and throughput.
    The default value of 1 is used to optimize the TTFT.
    Set to '' to optimize the throughput for short prompts.
-o  Output range, str, format='output_min,output_max', default='4,2048'
    Make sure the range cover all the possible lengths from the benchmark/client.
-b  max-num-seqs for vLLM, int, default=128
    Used to control the max batch size for decoding phase.
    It is recommended to set this value according to the 'Maximum concurrency'
    reported by a test run.
-g  max-seq-len-to-capture for vLLM, int, default=8192
    Used to control the maximum batched tokens to be captured in HPUgraph.
    Reduce this value could decrease memory usage, but not smaller than 2048.
-u  gpu-memory-utilization, float, default=0.9
    Used to control the GPU memory utilization. Reduce this value if OOM occurs.
-e  Extra vLLM server parameters, str, default=None
    Extra parameters that will pass to the vLLM server.
-l  Limit of the padding ratio, float, [0.0, 0.5], default=0.25
    The padding strategy ensures that padding_size/bucket_size <= limit.
    Smaller limit values result in more aggressive padding and more buckets.
    Set to 0.5 is equivalent to use the exponential bucketing.
    Set to 0.0 is equivalent to use the linear bucketing without padding limits.
-c  Cache HPU recipe to the specified path, str, default=None
    The recipe cache could be reused to reduce the warmup time.
-s  Skip warmup or not, bool, default=false
    Skip warmup to reduce startup time. Used in debug/dev environment only.
    DO NOT use in production environment.
-f  Enable high-level profiler or not, bool, default=false
-h  Help info
```

比较重要的参数包括：

- `-w` 指定模型权重所在目录
- `-t` 指定需要几张 Gaudi 加速卡，该选项映射到模型并行（TP）数。若模型使用 GQA 架构，请确保 n 不超过模型配置中的 `num_key_value_heads` 数值；此外该脚本会针对 MoE 模型自动使能专家并行（EP），EP 数等同于 TP 数。
- `-m` 指定使用卡的 module ID，可以通过命令 `hl-smi -Q index,module_id -f csv` 查询得到，请在指定 module ID 时，尽量确保使用相同 NUMA node 里的 module ID。Module 的 NUMA 信息可以通过命令 `hl-smi topo -c` 查询得到。
- `-a` API 服务器 URL，格式为 'IP:PORT'，默认值=127.0.0.1:30001
- `-d` 指定模型精度，['bfloat16'|'float16'|'fp8'|'awq'|'gptq']，默认值为'bfloat16'，如果提供了-q 或环境变量“QUANT_CONFIG”，则设置为“fp8”。
- `-q` 量化配置文件路径，默认值：无，如果通过 -d 'fp8' 将模型精度设置为 fp8，则默认值：./quantization/<model_name_lower>/maxabs_quant_g2.json，环境变量“QUANT_CONFIG”将覆盖此选项。
- `-i` 指定单个访问请求输入的长度范围，比如 `-i 4,16384` 表示输入长度在 4 个 token 到 16k 个 token。用户可根据实际业务部署需求更改上限值。
- `-p` 预填充序列的最大数量，整数，默认值为 1，用于控制预填充的最大批次大小，以平衡 TTFT 和吞吐量。默认值 1 用于优化 TTFT。设置为 '' 可优化短提示的吞吐量。
- `-o` 指定单个访问请求输出的长度范围，比如 `-o 4,2048` 表示输出长度在 4 个 token 到 2k 个 token。用户可根据实际业务部署需求更改上限值。
- `-b` 服务支持的最大并发，默认值：128，用于控制解码阶段的最大批次大小。建议根据“最大并发量”设置此值。
- `-g` 默认值为 8192，用于控制 HPUgraph 中捕获的最大批量 token 数量。降低此值可以减少内存使用量，但不能小于 2048。
- `-u` 浮点型，默认值：0.9，用于控制 GPU 内存利用率。如果发生 OOM，请降低此值。
- `-e` 额外的 vLLM 服务器参数，字符串，默认为无，将传递给 vLLM 服务器的额外参数。
- `-l` 填充比例的上限，浮点型，[0.0, 0.5]，默认值：0.25，填充策略确保 padding_size/bucket_size <= 上限。下限值越小，填充越激进，存储桶数量也越多。设置为 0.5 相当于使用指数型 bucketing。设置为 0.0 相当于使用没有填充限制的线性 bucketing。
- `-c` 执行模型预热的缓存目录
- `-s` 是否跳过模型预热，布尔值，默认值为 false，跳过预热以减少启动时间。仅适用于调试/开发环境。请勿用于生产环境。

服务启动命令和参数配置可参考[第三章节](#30-大模型服务启动示例)。

### 2.2 模型预热

以下启动命令表示启动 vLLM 服务在 Qwen2-72B-Instruct 模型上，使用 Gaudi 的 4 张卡，module ID 分别是 0、1、2、3，输入长度范围是 800 到 1024 token，输出范围是 400 到 512 token，推理精度是 BF16，vLLM 服务侦听在 30001 端口上。

```bash
bash start_gaudi_vllm_server.sh \
    -w "/models/Qwen2-72B-Instruct" \
    -t 4 \
    -m 0,1,2,3 \
    -a 127.0.0.1:30001 \
    -d bfloat16 \
    -i 800,1024 \
    -o 400,512 \
    -b 128 \
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
[参照 1.4 章节](#14-安装-vllm) 使用 vLLM aice/v1.22.0 版本安装到容器里。

启动 vLLM  
进入启动脚本目录，启动 vLLM。如下命令表示在 module ID 0,1,2,3（`-t 4 -m 0,1,2,3`）上使用 4 卡跑 Deepseek-R1-Distill-Llama-70B 模型，最大支持并发 128（`-b 128`），支持输入 token 长度 800 到 4096（`-i 800,4096`），输出 token 长度范围 800 到 2048（`-o 800,2048`），精度 BF16（`-d bfloat16`），服务端口 30001（`-a 127.0.0.1:30001`），预热缓存目录 `/data/70B_warmup_cache`（`-c /data/70B_warmup_cache`）：

```bash
cd vllm-fork/scripts
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/DeepSeek-R1-Distill-Llama-70B" \
    -t 4 \
    -m 0,1,2,3 \
    -b 128 \
    -i 800,4096 \
    -o 800,2048 \
    -d bfloat16 \
    -a 127.0.0.1:30001 \
    -c /data/70B_warmup_cache
```

DeepSeek-R1-Distill-Qwen-32B 模型 2 卡最大并发 32 部署可使用如下命令启动：

```bash
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/DeepSeek-R1-Distill-Qwen-32B" \
    -t 2 \
    -m 0,1 \
    -b 32 \
    -i 800,4096 \
    -o 800,2048 \
    -d bfloat16 \
    -a 127.0.0.1:30001 \
    -c /data/32B_warmup_cache
```

DeepSeek-R1-Distill-Llama-8B 模型单卡最大并发 32 部署可使用如下命令启动：

```bash
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/DeepSeek-R1-Distill-Llama-8B" \
    -t 1 \
    -m 0 \
    -b 32 \
    -i 800,4096 \
    -o 800,2048 \
    -d bfloat16 \
    -a 127.0.0.1:30001 \
    -c /data/8B_warmup_cache
```

### 3.3 Qwen 系列模型

#### 3.3.1 启动容器和下载模型权重

请用如下命令启动容器，假设 `/mnt/disk4` 有足够的硬盘空间用来保存模型权重，或者模型权重已经保存在该目录下。请为容器设置正确的网络设置，可以在容器内正常访问互联网资源。

```bash
docker run -it --name qwen_server --runtime=habana \
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
git clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git /data/hf_models/Qwen2.5-7B-Instruct
git clone https://www.modelscope.cn/Qwen/QwQ-32B.git /data/hf_models/QwQ-32B
git clone https://www.modelscope.cn/Qwen/Qwen3-8B.git /data/hf_models/Qwen3-8B
git clone https://www.modelscope.cn/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8.git /data/hf_models/Qwen3-Coder-480B-A35B-Instruct-FP8
```

#### 3.3.2 安装和启动 vLLM

安装 vLLM  
[参照 1.4 章节](#14-安装-vllm) 使用 vLLM aice/v1.22.0 版本安装到容器里。

启动 vLLM  
进入启动脚本目录，启动 vLLM。

Qwen2.5-7B-Instruct 模型单卡最大并发 32 部署可使用如下命令启动：

```bash
cd vllm-fork/scripts
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/Qwen2.5-7B-Instruct" \
    -t 1 \
    -m 0 \
    -b 32 \
    -i 800,4096 \
    -o 800,2048 \
    -d bfloat16 \
    -a 127.0.0.1:30001 \
    -c /data/7B_warmup_cache
```

QwQ-32B 模型 2 卡最大并发 32 部署可使用如下命令启动：

```bash
cd vllm-fork/scripts
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/QwQ-32B" \
    -t 2 \
    -m 0,1 \
    -b 32 \
    -i 800,4096 \
    -o 800,2048 \
    -d bfloat16 \
    -a 127.0.0.1:30001 \
    -c /data/32B_warmup_cache
```

Qwen3-8B 模型 2 卡最大并发 32 部署可使用如下命令启动：

```bash
cd vllm-fork/scripts
bash start_gaudi_vllm_server.sh \
    -w "/data/hf_models/Qwen3-8B" \
    -t 2 \
    -m 0,1 \
    -b 32 \
    -i 800,4096 \
    -o 800,2048 \
    -d bfloat16 \
    -a 127.0.0.1:30001 \
    -c /data/8B_warmup_cache
```

对于 Qwen3-Coder-480B-A35B-Instruct-FP8，需要使用脚本来对原始权重进行转换，可以从下面链接获得相关的脚本。

- https://github.com/HabanaAI/vllm-fork/blob/deepseek_r1/scripts/convert_for_g2.py
- https://github.com/HabanaAI/vllm-fork/blob/deepseek_r1/scripts/run_inc_calib.sh
- https://github.com/HabanaAI/vllm-fork/tree/deepseek_r1/scripts/quickstart

获得相应的脚本之后，可以使用下面的命令来进行模型权重转换。

```bash
#转换模型
cd /root/vllm-fork
export HF_ENDPOINT="https://hf-mirror.com"
python scripts/convert_for_g2.py -i /data/hf_models/Qwen3-Coder-480B-A35B-Instruct-FP8 -o /data/hf_models/Qwen3-Coder-480B-A35B-Instruct-FP8-G2
bash scripts/run_inc_calib.sh --model /data/hf_models/Qwen3-Coder-480B-A35B-Instruct-FP8-G2
```

Qwen3-Coder-480B-A35B-Instruct-FP8-G2 模型部署可使用如下命令启动：

```bash
export VLLM_DISABLE_MARK_SCALES_AS_CONST=true
bash scripts/quickstart/start_vllm.sh -w /data/hf_models/Qwen3-Coder-480B-A35B-Instruct-FP8-G2 -q
```

### 3.4 多模态模型

如果要做音频处理，需要安装音频相关的库。

```bash
pip install vllm[audio]
```

#### 3.4.1 Qwen 系列多模态模型

**启动服务**\
**Qwen2-VL**: Support Image and Video inputs

```bash
PT_HPU_LAZY_MODE=1 vllm serve \
    Qwen/Qwen2-VL-7B-Instruct \
    --port 8000 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --limit-mm-per-prompt video=5,image=5 \
    --mm_processor_kwargs max_pixels=1003520
```

**Qwen2-Audio**: Support Audio inputs

```bash
PT_HPU_LAZY_MODE=1 vllm serve \
    Qwen/Qwen2-Audio-7B-Instruct \
    --port 8000 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --limit-mm-per-prompt audio=5 \
    --mm_processor_kwargs max_pixels=1003520
```

**Qwen2.5-VL**: Support Image and Video inputs

```bash
PT_HPU_LAZY_MODE=1 vllm serve \
    Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8000 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --limit-mm-per-prompt video=5,image=5 \
    --mm_processor_kwargs max_pixels=1003520
```

**Qwen2.5-Omni**: Support Image, Video and Audio inputs

```bash
PT_HPU_LAZY_MODE=1 vllm serve \
    Qwen/Qwen2.5-Omni-7B \
    --port 8000 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --limit-mm-per-prompt audio=5,video=5,image=5 \
    --mm_processor_kwargs max_pixels=1003520
```

**Qwen3-VL**: Support Image and Video inputs

```bash
PT_HPU_LAZY_MODE=1 vllm serve \
    Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --limit-mm-per-prompt video=5,image=5 \
    --mm_processor_kwargs max_pixels=1003520,min_pixels=3136
```

- `--limit-mm-per-prompt` 设置每个 prompt 中每种多模态数据的最大个数
- `--mm_processor_kwargs max_pixels=1003520` 限制输入图片最大尺寸。超过的图片会被保持宽高比例缩小。

#### 3.4.2 client 端请求格式样例

多模态 client 端请求格式可以参考脚本 [openai_chat_completion_client_for_multimodal.py](../examples/online_serving/openai_chat_completion_client_for_multimodal.py)

脚本使用命令：

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal.py \
    -c multi-image
python examples/online_serving/openai_chat_completion_client_for_multimodal.py \
    -c video
python examples/online_serving/openai_chat_completion_client_for_multimodal.py \
    -c audio
```

#### 3.4.3 FP8 dynamic quant
**转换模型**\
首先把模型下载到本地 \

```bash
git clone https://github.com/HabanaAI/vllm-hpu-extension.git -b aice/v1.22.0
cd vllm-hpu-extension/scripts
python dynamic_quant_multimodal_for_gaudi2.py \
    -i /data/Qwen3-VL-30B-A3B-Instruct \
    -o /data/Qwen3-VL-30B-A3B-Instruct-FP8-G2-Dynamic
```

**启动服务**\

```bash
PT_HPU_LAZY_MODE=1 vllm serve \
    /data/Qwen3-VL-30B-A3B-Instruct-FP8-G2-Dynamic \
    --port 8000 \
    --host 127.0.0.1 \
    --limit-mm-per-prompt video=5,image=5 \
    --mm_processor_kwargs max_pixels=1003520,min_pixels=3136
```

#### 3.4.4 问题解答

- 如果 server 端出现获取图像音视频超时错误，可以通过设置环境变量`VLLM_IMAGE_FETCH_TIMEOUT` `VLLM_VIDEO_FETCH_TIMEOUT` `VLLM_AUDIO_FETCH_TIMEOUT` 来提高超时时间。默认为 5/30/10
- 过大的输入图像要求更多的设备内存，可以通过设置更小的参数`--gpu-memory-utilization` （默认 0.9）来解决。例如参考脚本`openai_chat_completion_client_for_multimodal.py`中的图像分辨率最高达到 7952x5304,这会导致 server 端推理出错。可以通过设置`--gpu-memory-utilization`至 0.6~0.7 来解决。
