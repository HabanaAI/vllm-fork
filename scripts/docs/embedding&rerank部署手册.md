# 1.0 Embedding & Rerank 部署手册

## 1.1 Embedding 模型（单卡部署）

### 1.1.1 TEI 部署指南

本文档详细介绍如何在 HPU（Habana Processing Unit）上部署 TEI（Text Embeddings Inference）embedding 服务，包括资源分配、Docker 部署、参数配置、测试验证等方面。

#### 资源分配

| 服务          | CPU    | Memory | HPU  | 网络         | 磁盘空间                                                          |
| ------------- | ------ | ------ | ---- | ------------ | ----------------------------------------------------------------- |
| TEI embedding | 2 vcpu | 4GB    | 单卡 | Ip: 服务端口 | 宿主机 `${DATA_PATH}` 挂载到容器 `/data`，权重保存在 `/DATA_PATH` |

### 1.1.2 镜像文件

Intel 在 Gaudi 基础镜像基础上编译了 TEI 镜像，可用于部署 RAG 相关组件：

```bash
docker pull ghcr.io/huggingface/text-embeddings-inference:hpu-latest
```

### 1.1.3 部署参数说明

#### 通用环境变量配置

| 变量名                 | 说明           | 示例值                |
| ---------------------- | -------------- | --------------------- |
| EMBEDDING_MODEL_ID     | 模型路径名称   | bge-large-zh-v1.5     |
| DATA_PATH              | 模型数据根目录 | /model/               |
| TEI_EMBEDDING_PORT     | 服务端口       | 12003                 |
| host_ip                | 主机 IP 地址   | 10.239.241.85 or none |
| HABANA_VISIBLE_DEVICES | HPU 设备 ID    | 2                     |

#### 模型特定参数

- **BGE 系列模型**
  - bge-base-zh-v1.5: `warmup_length=512`, `MAX_WARMUP_BATCH_SIZE=32`
  - bge-large-zh-v1.5: `warmup_length=512`, `MAX_WARMUP_BATCH_SIZE=32`
  - bge-m3: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=16`
- **Qwen3 系列模型**
  - Qwen3-Embedding-0.6B: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=16`
  - Qwen3-Embedding-4B: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=16`
  - Qwen3-Embedding-8B: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=16`

### 1.1.4 Docker 部署步骤

#### 前提条件

确保已安装以下组件:

- Docker Engine
- Habana Docker runtime
- HPU 驱动和相关软件栈

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
 docker run -d --name tei-embedding-serving   --runtime=habana   -p ${TEI_EMBEDDING_PORT:-12003}:80   -v "${DATA_PATH:-./data}:/data"   --shm-size 2g   -e host_ip=${host_ip}   -e HABANA_VISIBLE_DEVICES=${id:-0}   -e OMPI_MCA_btl_vader_single_copy_mechanism=none   -e MAX_WARMUP_SEQUENCE_LENGTH=${warmup_length:-512}   -e MAX_WARMUP_BATCH_SIZE=${warmup_batch:-16}   -e MAX_CLIENT_BATCH_SIZE=${client_batch:-64}   -e MAX_BATCH_REQUESTS=${warmup_batch:-16} ghcr.io/huggingface/text-embeddings-inference:hpu-latest --model-id /data/${EMBEDDING_MODEL_ID} --dtype bfloat16 --auto-truncate
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

### 1.1.5 Smoke Test

#### 1.1.5.1 健康检查

服务启动后，可以通过以下方式验证：

```bash
# 检查容器状态
docker ps | grep tei-embedding-serving
# 查看服务日志
docker logs tei-embedding-serving | grep "Ready"
```

#### 1.1.5.2 基本功能测试

```bash
export no_proxy="localhost, 127.0.0.1, ::1"

# 测试embedding接口
curl -X POST http://localhost:12003/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs":"这是一个测试文本"}'

curl -X POST http://localhost:12003/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"这是一个测试文本"}'
```

### 1.1.7 其他文档和资料

TEI embedding 服务支持 openAI/v1/embeddings API 兼容格式，也支持自定义 embed API 格式。  
API swagger 文档链接：

- https://huggingface.github.io/text-embeddings-inference
- https://github.com/huggingface/text-embeddings-inference/

## 1.2 Re-ranking 模型（单卡部署）

TEI 部署 rerank 模型步骤与 Embedding 相同，仅需替换模型为 rerank 模型，TEI 会自动识别。

### 1.2.1 模型文件

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

### 1.2.2 镜像文件

镜像与 embedding 使用一致，使用 Gaudi 驱动版本对应镜像。

### 1.2.3 部署参数说明

#### 通用参数

| 变量名                 | 说明           | 示例值             |
| ---------------------- | -------------- | ------------------ |
| RERANK_MODEL_ID        | 模型路径名称   | bge-reranker-large |
| DATA_PATH              | 模型数据根目录 | /model/            |
| TEI_RERANKING_PORT     | 服务端口       | 12005              |
| host_ip                | 主机 IP 地址   | 10.239.241.85      |
| HABANA_VISIBLE_DEVICES | HPU 设备 ID    | 2                  |

#### 模型特定参数

- bge-reranker-base: `warmup_length=512`, `MAX_WARMUP_BATCH_SIZE=8`
- bge-reranker-large: `warmup_length=512`, `MAX_WARMUP_BATCH_SIZE=8`
- bge-reranker-v2-m3: `warmup_length=2048`, `MAX_WARMUP_BATCH_SIZE=8`

### 1.2.4 启动服务

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
 docker run -d --name tei-reranking-serving --runtime=habana -p ${TEI_RERANKING_PORT:-12007}:80 -v "${DATA_PATH:-./data}:/data" --shm-size 2g -e host_ip=${host_ip} -e HABANA_VISIBLE_DEVICES=${id:-0}   -e OMPI_MCA_btl_vader_single_copy_mechanism=none   -e MAX_WARMUP_SEQUENCE_LENGTH=${warmup_length:-512}   -e MAX_WARMUP_BATCH_SIZE=${warmup_batch:-32}   -e MAX_CLIENT_BATCH_SIZE=${client_batch:-32}   -e MAX_BATCH_REQUESTS=${warmup_batch:-32}   ghcr.io/huggingface/text-embeddings-inference:hpu-latest --model-id /data/${RERANK_MODEL_ID} --dtype bfloat16 --auto-truncate
```

### 1.2.5 健康检查

服务启动后，可以通过以下方式验证：

```bash
# 检查容器状态
docker ps | grep tei-reranking-serving
# 查看服务日志
docker logs tei-reranking-serving
```

### 1.2.6 基本功能测试

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
