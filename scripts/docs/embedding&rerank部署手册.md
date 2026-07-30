# 1.0 Embedding & Rerank 部署手册

本文档详细介绍如何部署 Embedding & Rerank 服务，包括资源分配、Docker 部署、参数配置、测试验证等方面。文档涵盖两种主要部署方式：
1. **TEI Intel Gaudi 部署**：在 Intel Gaudi 上部署 TEI（Text Embeddings Inference）embedding & rerank 服务，适用于拥有 Intel Gaudi 硬件加速环境的场景。

2. **vLLM CPU 部署**：通过 vLLM 在 CPU 环境下高效部署 Embedding 与 Rerank 服务，适用于无 Intel Gaudi 硬件或需多版本 API（如 /v1/rerank、/v2/rerank）场景。

## 1.1 TEI Intel Gaudi 部署

### 1.1.1 Embedding 模型（单卡部署）

#### 1.1.1.1 TEI 部署指南

#### 资源分配

| 服务          | CPU    | Memory | Intel Gaudi | 网络         | 磁盘空间                                                          |
| ------------- | ------ | ------ | ----------- | ------------ | ----------------------------------------------------------------- |
| TEI embedding | 2 vcpu | 4GB    | 单卡        | Ip: 服务端口 | 宿主机 `${DATA_PATH}` 挂载到容器 `/data`，权重保存在 `/DATA_PATH` |

#### 1.1.1.2 镜像文件

Intel 在 Gaudi 基础镜像基础上编译了 TEI 镜像，可用于部署 RAG 相关组件：

```bash
docker pull ghcr.io/huggingface/text-embeddings-inference:hpu-latest
```

对于特定版本的驱动，如需编译镜像，请使用如下命令编译

```bash
git clone https://github.com/huggingface/text-embeddings-inference.git
cd text-embeddings-inference
# 如果需要修改基础镜像，请修改 Dockerfile-intel
# FROM vault.habana.ai/gaudi-docker/1.21.3/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.21.3-57 AS hpu  #将该镜像替换为驱动要求版本镜像，保存
#build image
ulimit=65536
export platform=hpu

#如果不需要设置代理，请去掉HTTPS_PROXY的设置。
docker build . --ulimit nofile=65535:65535 --build-arg HTTPS_PROXY=${HTTPS_PROXY} -f Dockerfile-intel --build-arg PLATFORM=$platform -t tei_hpu

```

#### 1.1.1.3 部署参数说明

#### 通用环境变量配置

| 变量名                 | 说明           | 示例值                |
| ---------------------- | -------------- | --------------------- |
| EMBEDDING_MODEL_ID     | 模型路径名称   | bge-large-zh-v1.5     |
| DATA_PATH              | 模型数据根目录 | /model/               |
| TEI_EMBEDDING_PORT     | 服务端口       | 12003                 |
| host_ip                | 主机 IP 地址   | 10.239.241.85 or none |
| HABANA_VISIBLE_DEVICES | Intel Gaudi 设备 ID | 0                     |

#### 模型特定参数

- **BGE 系列模型**
  - bge-base-zh-v1.5: `warmup_length=512`, `MAX_WARMUP_BATCH_SIZE=32`
  - bge-large-zh-v1.5: `warmup_length=512`, `MAX_WARMUP_BATCH_SIZE=32`
  - bge-m3: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=16`
- **Qwen3 系列模型**
  - Qwen3-Embedding-0.6B: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=16`
  - Qwen3-Embedding-4B: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=16`
  - Qwen3-Embedding-8B: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=16`

#### 1.1.1.4 Docker 部署步骤

#### 前提条件

确保已安装以下组件:

- Docker Engine
- Habana Docker runtime
- Intel Gaudi 驱动和相关软件栈

#### 部署流程

**步骤 1：准备模型文件**

支持的模型包括：

- bge-base-zh-v1.5
- bge-large-zh-v1.5
- bge-m3
- Qwen3-Embedding-0.6B
- Qwen3-Embedding-4B
- Qwen3-Embedding-8B

您可以在 HuggingFace 或 ModelScope 网站上下载需要的模型权重文件。  
例如从 ModelScope 下载 bge-large-zh-v1.5 模型权重文件：

```bash
sudo apt install git-lfs
git-lfs install
git clone https://www.modelscope.cn/Xorbits/bge-large-zh-v1.5.git /model/bge-large-zh-v1.5
```

以下步骤假设使用 bge-large-zh-v1.5 作为测试模型

**步骤 2：设置部署环境必备变量**

必备变量

```bash
# EMBEDDING_MODEL_ID 为docker 内部使用的模型名称，应该与模型权重目录一致
export EMBEDDING_MODEL_ID=bge-large-zh-v1.5
#DATA_PATH 为host 映射入docker内的物理卷，应配置为存放权重目录
export DATA_PATH=/model/
#Gaudi Device id
export id=0
```

可选变量

```bash
#对外服务端口映射
TEI_EMBEDDING_PORT: 默认12003
host_ip:服务绑定ip，默认是0.0.0.0
warmup_length: 取min(max_model_len, 2048), 默认值512
warmup_batch：默认16，一般不需要调节
client_batch：最大客户端并发值，默认64，一般不需要调节
```

**步骤 3：启动服务**

```bash
 docker run -d --name tei-embedding-serving   --runtime=habana   -p ${TEI_EMBEDDING_PORT:-12003}:80   -v "${DATA_PATH:-./data}:/data"   --shm-size 2g   -e host_ip=${host_ip}   -e HABANA_VISIBLE_DEVICES=${id:-0}   -e OMPI_MCA_btl_vader_single_copy_mechanism=none   -e MAX_WARMUP_SEQUENCE_LENGTH=${warmup_length:-512}   -e MAX_WARMUP_BATCH_SIZE=${warmup_batch:-16}   -e MAX_CLIENT_BATCH_SIZE=${client_batch:-64}   -e MAX_BATCH_REQUESTS=${warmup_batch:-16} tei_hpu:latest --model-id /data/${EMBEDDING_MODEL_ID} --dtype bfloat16 --auto-truncate
```

**步骤 4：确认服务就绪**

通过确认 log 日志来验证服务已经正常启动。

```bash
while true; do
    docker logs tei-embedding-serving 2>/dev/null | grep "Ready"
    if [ "$?" -eq 0 ]; then
        echo "docker is ready exit the loop."
        break
    fi
    echo "within loop"
    sleep 30
done
```

#### 1.1.1.5 Smoke Test

##### 1.1.1.5.1 健康检查

服务启动后，可以通过以下方式验证：

```bash
# 检查容器状态
docker ps | grep tei-embedding-serving
# 查看服务日志
docker logs tei-embedding-serving | grep "Ready"
```

##### 1.1.1.5.2 基本功能测试

TEI embedding 服务支持 openAI/v1/embeddings API 兼容格式，也支持自定义 embed API 格式。

```bash
export no_proxy="localhost, 127.0.0.1, ::1"

# 测试 TEI 自定义 embed API 格式
curl -X POST http://localhost:12003/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs":"这是一个测试文本"}'

 #测试 TEI openAI/v1/embeddings API 兼容格式
curl -X POST http://localhost:12003/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"这是一个测试文本"}'
```

#### 1.1.1.6 其他文档和资料

API swagger 文档链接：

- https://huggingface.github.io/text-embeddings-inference
- https://github.com/huggingface/text-embeddings-inference/

### 1.1.2 Re-ranking 模型（单卡部署）

TEI 部署 rerank 模型步骤与 Embedding 相同，仅需替换模型为 rerank 模型，TEI 会自动识别。

#### 1.1.2.1 模型文件

模型配置方式与 embedding 一样，只要将模型 weights 替换为 rerank 模型，服务可以自动检测。

已验证的模型包括：

- bge-reranker-base
- bge-reranker-large
- bge-reranker-v2-m3

您可以在 HuggingFace 或 ModelScope 网站上下载需要的模型权重文件。  
例如从 ModelScope 下载 bge-reranker-large 模型权重文件：

```bash
sudo apt install git-lfs
git-lfs install
git clone https://www.modelscope.cn/BAAI/bge-reranker-large.git /model/bge-reranker-large
```

以下步骤假设使用 bge-reranker-large 作为测试模型。

#### 1.1.2.2 镜像文件

镜像与 embedding 使用一致，使用 Gaudi 驱动版本对应镜像。

#### 1.1.2.3 部署参数说明

#### 通用参数

| 变量名                 | 说明           | 示例值             |
| ---------------------- | -------------- | ------------------ |
| RERANK_MODEL_ID        | 模型路径名称   | bge-reranker-large |
| DATA_PATH              | 模型数据根目录 | /model/            |
| TEI_RERANKING_PORT     | 服务端口       | 12007              |
| host_ip                | 主机 IP 地址   | 10.239.241.85      |
| HABANA_VISIBLE_DEVICES | Intel Gaudi 设备 ID | 0                     |

#### 模型特定参数

- bge-reranker-base: `warmup_length=512`, `MAX_WARMUP_BATCH_SIZE=8`
- bge-reranker-large: `warmup_length=512`, `MAX_WARMUP_BATCH_SIZE=8`
- bge-reranker-v2-m3: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=8`

#### 1.1.2.4 启动服务

必备变量

```bash
# RERANK_MODEL_ID 为docker 内部使用的模型名称，应该与模型权重目录一致
export RERANK_MODEL_ID=bge-reranker-large
#DATA_PATH 为host 映射入docker内的物理卷，应配置为存放权重目录
export DATA_PATH=/model/
#Gaudi Device id
export id=0
```

可选变量

```bash
#对外服务端口映射
TEI_RERANKING_PORT: 默认12007
host_ip:服务绑定ip，默认是0.0.0.0
warmup_length: 取min(max_model_len, 1024), 默认值512
warmup_batch：默认32，一般不需要调节
client_batch：最大客户端并发值，默认32，一般不需要调节
```

启动 docker 实例

```bash
 docker run -d --name tei-reranking-serving --runtime=habana -p ${TEI_RERANKING_PORT:-12007}:80 -v "${DATA_PATH:-./data}:/data" --shm-size 2g -e host_ip=${host_ip} -e HABANA_VISIBLE_DEVICES=${id:-0}   -e OMPI_MCA_btl_vader_single_copy_mechanism=none   -e MAX_WARMUP_SEQUENCE_LENGTH=${warmup_length:-512}   -e MAX_WARMUP_BATCH_SIZE=${warmup_batch:-32}   -e MAX_CLIENT_BATCH_SIZE=${client_batch:-32}   -e MAX_BATCH_REQUESTS=${warmup_batch:-32}  tei_hpu:latest --model-id /data/${RERANK_MODEL_ID} --dtype bfloat16 --auto-truncate
```

#### 1.1.2.5 健康检查

服务启动后，可以通过以下方式验证：

```bash
# 检查容器状态
docker ps | grep tei-reranking-serving
# 查看服务日志
docker logs tei-reranking-serving
```

#### 1.1.2.6 基本功能测试

```bash
export no_proxy="localhost, 127.0.0.1, ::1"

curl 127.0.0.1:12007/rerank \
    -X POST \
    -d '{"query": "我喜欢什么？", "texts": [
            "昨天没喝水，因为我一直在忙着工作，完全忘记了时间，直到晚上才意识到自己一整天都没有喝水。",
            "你是谁？能不能详细介绍一下你自己，包括你的名字、职业、兴趣爱好以及你目前在做的事情？",
            "他很好，他不仅在工作上表现出色，而且在生活中也是一个非常善良和乐于助人的人，大家都很喜欢他。",
            "今天天气不错，阳光明媚，微风习习，正是出去散步或者进行户外活动的好时机。",
            "我喜欢编程，因为编程不仅能让我解决实际问题，还能让我创造出各种有趣的应用和工具。",
            "你去哪里了？我找了你很久，打了好几个电话你都没有接，是不是发生了什么事情？",
            "我们一起去看电影吧，这部新上映的电影评价很高，剧情也很吸引人，我觉得你一定会喜欢。",
            "这道题很难，我已经尝试了好几种方法都没有解出来，你能不能帮我看看问题出在哪里？",
            "你喜欢什么运动？我最近开始跑步，感觉对身体很有好处，你有没有兴趣一起参加？",
            "明天有空吗？我们可以一起去逛街，顺便吃个饭，聊聊最近的生活和工作。"
        ]}' \
    -H 'Content-Type: application/json'

```

## 1.2 vLLM CPU 部署

vLLM 支持在通用 CPU 环境下高效部署 Embedding 与 Rerank 服务，适用于无 Intel Gaudi 硬件或需多版本 API（如 /v1/rerank、/v2/rerank）场景。

vLLM 对 CPU 的支持相对较新，参考官方文档：https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html

### 1.2.1 构建 vLLM CPU Docker 镜像

由于 vLLM 目前没有提供预编译的 CPU wheels，需要从源码构建 Docker 镜像。

**步骤 1：创建 Dockerfile**

```dockerfile
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ccache curl wget ca-certificates gcc-12 g++-12 \
    libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 \
    libgl1 jq lsof python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12

# 配置 git config
RUN git config --global user.name "Abc"
RUN git config --global user.email "abc@example.com"

# 克隆 vLLM 仓库
RUN git clone https://github.com/vllm-project/vllm.git

# 切换到 vLLM 目录并设置版本
WORKDIR /workspace/vllm
RUN git checkout v0.9.2

# Cherry-pick 指定 commit
RUN git cherry-pick 3fc96443

# 完整的 build & install
RUN pip3 install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN VLLM_TARGET_DEVICE=cpu python3 setup.py develop

# 设置工作目录回到根目录
WORKDIR /workspace

# 创建数据目录
RUN mkdir -p /data

# 设置环境变量
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV CUDA_VISIBLE_DEVICES=""
ENV VLLM_TARGET_DEVICE=cpu
ENV PYTHONPATH="/workspace/vllm"
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# 可配置参数的默认值
ENV MODEL_NAME_ID="bge-reranker-v2-m3"
ENV SERVER_PORT="80"
ENV SERVER_HOST="0.0.0.0"

# 设置默认启动命令，使用 /data 作为模型路径
CMD ["sh", "-c", "vllm serve /data/$MODEL_NAME_ID --port $SERVER_PORT --host $SERVER_HOST"]
```

**步骤 2：构建 Docker 镜像**

```bash
# 构建 vLLM CPU 镜像
docker build -t vllm:cpu .
```

如果需要代理支持，使用以下命令：
```bash
# 带代理的构建命令
docker build $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/--build-arg /') -t vllm:cpu .
```

**注意事项：**
- 构建过程需要从 GitHub 克隆代码库，请确保网络连接正常
- 建议在构建机器上配置足够的内存（建议 16GB+）

**步骤 3：验证镜像构建**

```bash
# 验证 vLLM 安装
docker run --rm vllm:cpu python -c "import vllm; print(vllm.__version__)"

# 测试 vLLM 命令行工具
docker run --rm vllm:cpu vllm --version
```

### 1.2.2 vLLM CPU 服务部署

#### 部署参数说明

| 环境变量名          | 说明                     | 示例值                    |
| ------------------- | ------------------------ | ------------------------- |
| MODEL_NAME_ID       | 模型名称/目录名          | bge-large-zh-v1.5         |
| SERVER_HOST         | 服务监听地址             | 0.0.0.0                   |
| SERVER_PORT         | 宿主机端口（外部访问）   | 12003, 12007              |
| DATA_PATH           | 宿主机模型根目录         | /model                    |

#### 部署步骤


**步骤 1：准备模型文件**

模型文件的准备方法请参考前述 TEI Intel Gaudi 部署章节：
- Embedding 模型文件准备请参考 [1.1.1.4 Docker 部署步骤 - 步骤 1](#1114-docker-部署步骤)
- Rerank 模型文件准备请参考 [1.1.2.1 模型文件](#1121-模型文件)

**步骤 2：启动服务**

**通用启动命令模板**

```bash
docker run --name vllm-serving-${MODEL_NAME_ID:-bge-reranker-v2-m3} -d \
  -e MODEL_NAME_ID=${MODEL_NAME_ID:-bge-reranker-v2-m3} \
  -e SERVER_HOST=${SERVER_HOST:-0.0.0.0} \
  -p ${SERVER_PORT:-12003}:80 \
  -v ${DATA_PATH:-/model}:/data \
  vllm:cpu
```

**启动 Embedding 服务示例**

```bash
# 设置环境变量
export MODEL_NAME_ID=bge-large-zh-v1.5
export SERVER_PORT=12003
export DATA_PATH=/model

# 启动 embedding 服务
docker run --name vllm-serving-${MODEL_NAME_ID} -d \
  -e MODEL_NAME_ID=${MODEL_NAME_ID} \
  -e SERVER_HOST=${SERVER_HOST:-0.0.0.0} \
  -p ${SERVER_PORT}:80 \
  -v ${DATA_PATH}:/data \
  vllm:cpu
```

**启动 Rerank 服务示例**

```bash
# 设置环境变量
export MODEL_NAME_ID=bge-reranker-large
export SERVER_PORT=12007
export DATA_PATH=/model

# 启动 rerank 服务
docker run --name vllm-serving-${MODEL_NAME_ID} -d \
  -e MODEL_NAME_ID=${MODEL_NAME_ID} \
  -e SERVER_HOST=${SERVER_HOST:-0.0.0.0} \
  -p ${SERVER_PORT}:80 \
  -v ${DATA_PATH}:/data \
  vllm:cpu
```

**步骤 3：服务验证**

**健康检查**

```bash
# 检查服务状态（使用对应的端口）
curl http://localhost:${SERVER_PORT}/v1/models | jq '.'

# 检查容器运行状态
docker ps | grep vllm-serving
```

**功能测试**

**注意事项：**
- 确保测试命令中的端口与启动服务时设置的 `${SERVER_PORT}` 一致
- 确保测试命令中的模型路径与启动服务时设置的 `${MODEL_NAME_ID}` 一致
- 模型路径格式为 `/data/${MODEL_NAME_ID}`，需要包含容器内的完整路径

**测试 Embedding 服务**

```bash
export no_proxy="localhost,127.0.0.1,::1"

# 测试 OpenAI 兼容的 embeddings API
# 注意：端口 12003 和模型路径 /data/bge-large-zh-v1.5 需要与启动的服务保持一致
curl -X POST http://localhost:12003/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是一个测试文本",
    "model": "/data/bge-large-zh-v1.5"
  }'
```

**测试 Rerank 服务**

```bash
export no_proxy="localhost,127.0.0.1,::1"

# 测试 v1/rerank API
# 注意：端口 12007 和模型路径 /data/bge-reranker-large 需要与启动的服务保持一致
curl -X POST http://localhost:12007/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/bge-reranker-large",
    "query": "我喜欢什么？",
    "documents": [
      "昨天没喝水，因为我一直在忙着工作。",
      "我喜欢编程，因为编程不仅能让我解决实际问题。",
      "今天天气不错，阳光明媚。"
    ]
  }'

# 测试 v2/rerank API
# 注意：端口 12007 和模型路径 /data/bge-reranker-large 需要与启动的服务保持一致
curl -X POST http://localhost:12007/v2/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/bge-reranker-large",
    "query": "我喜欢什么？",
    "documents": [
      "昨天没喝水，因为我一直在忙着工作。",
      "我喜欢编程，因为编程不仅能让我解决实际问题。",
      "今天天气不错，阳光明媚。"
    ]
  }'
```
