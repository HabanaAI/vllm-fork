# Embedding/Rrank多模型单HPU卡 vLLM推理部署指南

从aice 1.22.0 开始，vllm可以提供在单HPU上运行多模型推理服务。
针对Embedding和rerank类场景，本文档提供了基于 vLLM-fork 的多模型推理服务的完整部署指南，包括系统配置、服务启动、客户端配置和调优参数。

## 已验证模型

Embedding：
bge-large-zh-v1.5
bge-m3
gte-modernbert-base

Rerank

bge-reranker-large
bge-reranker-v2-m3
gte-reranker-modernbert-base

## 快速开始

```bash
# 基本使用（默认max-model-len=512）
python launch_multi_models.py --models /data/models/gte-modernbert-base \
                                      /data/models/gte-reranker-modernbert-base \
                                      --port 8771

# 自定义max-model-len
python launch_multi_models.py --models model1 model2 --max-model-len 8192

# 使用性能预设（推荐用于生产环境）
python launch_multi_models.py --models model1 model2 --env-preset performance
```

## 目录

1. [系统OS配置](#系统os配置)
2. [服务启动](#服务启动)
3. [客户端配置](#客户端配置)
4. [调优参数](#调优参数)

## 系统OS配置

### 硬件要求
- **CPU**: 3 vCPU
- **HPU**: 1卡

### CPU 调优配置

由于Embedding和Rerank 规模较小，为了尽量减少host侧CPU频率的影响，请使用高主频CPU，并确保CPU工作在性能模式，关闭C1E和C6状态。
相关配置可以在BIOS中或者OS进行相关配置。OS内配置命令如下。

```
bash
#打开性能模式
sudo echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#关闭C1E，C6
cpupower idle-set -d 2
cpupower idle-set -d 3

```

### 安装vLLM-fork

请按照vLLM-fork标准方式安装vLLM 1.22.0 版本。
具体步骤，请参考 scripts/README.md 进行。

## 服务启动

### 启动docker
参考1.22.0 标准docker 启动方式

### 方法一：使用增强启动脚本

`launch_multi_models.py` 脚本提供了灵活的配置选项，支持可配置的max-model-len参数和多种环境变量配置方式。

#### 脚本参数说明
- **可配置max-model-len**: 通过`--max-model-len`参数灵活配置最大模型长度
- **多种环境变量配置方式**:
  - 预设配置（default/performance）
  - JSON配置文件
  - 命令行单独设置

#### 基本启动命令

```bash
# 自定义max-model-len
python launch_multi_models.py --models /data/models/gte-modernbert-base /data/models/gte-reranker-modernbert-base \
                              --port 8771 \
                              --max-model-len 4096

# 选用预设配置启动
pythonlaunch_multi_models.py --models /data/models/gte-modernbert-base /data/models/gte-reranker-modernbert-base \
                             --port 8771 --env-preset performance

# 使用JSON配置文件启动，并强制重启
python launch_multi_models.py --models /data/models/gte-modernbert-base /data/models/gte-reranker-modernbert-base \
                              --port 8771 --env-config /data/env_config_example.json --force-restart
```

#### 环境变量配置

##### 使用预设配置

```bash
# 默认预设（保守设置）
python launch_multi_models.py --models model1 model2 --env-preset default

# 性能预设（包含所有调优参数）
python launch_multi_models.py --models model1 model2 --env-preset performance
```

**性能预设包含的环境变量：**

```
VLLM_CONTIGUOUS_PA=false
VLLM_SKIP_WARMUP=false
VLLM_PROMPT_BS_BUCKET_MIN=1
VLLM_PROMPT_BS_BUCKET_STEP=1
VLLM_PROMPT_BS_BUCKET_MAX=8
VLLM_PROMPT_SEQ_BUCKET_MIN=1
VLLM_PROMPT_SEQ_BUCKET_STEP=512
VLLM_PROMPT_SEQ_BUCKET_MAX=4096
VLLM_DECODE_BS_BUCKET_MIN=1
VLLM_DECODE_BS_BUCKET_STEP=1
VLLM_DECODE_BS_BUCKET_MAX=2
VLLM_DECODE_BLOCK_BUCKET_MIN=1
VLLM_DECODE_BLOCK_BUCKET_STEP=1
VLLM_DECODE_BLOCK_BUCKET_MAX=2
PT_HPU_LAZY_MODE=1
```

##### 使用配置文件

```bash
# 从JSON文件加载环境变量配置
python launch_multi_models.py --models model1 model2 --env-config env_config.json
```

配置文件格式（env_config.json）：

```json
{
  "VLLM_CONTIGUOUS_PA": "false",
  "VLLM_SKIP_WARMUP": "false",
  "VLLM_PROMPT_BS_BUCKET_MIN": "1",
  "VLLM_PROMPT_BS_BUCKET_STEP": "1",
  "VLLM_PROMPT_BS_BUCKET_MAX": "8",
  "VLLM_PROMPT_SEQ_BUCKET_MIN": "1",
  "VLLM_PROMPT_SEQ_BUCKET_STEP": "512",
  "VLLM_PROMPT_SEQ_BUCKET_MAX": "4096",
  "VLLM_DECODE_BS_BUCKET_MIN": "1",
  "VLLM_DECODE_BS_BUCKET_STEP": "1",
  "VLLM_DECODE_BS_BUCKET_MAX": "2",
  "VLLM_DECODE_BLOCK_BUCKET_MIN": "1",
  "VLLM_DECODE_BLOCK_BUCKET_STEP": "1",
  "VLLM_DECODE_BLOCK_BUCKET_MAX": "2",
  "PT_HPU_LAZY_MODE": "1"
}
```

##### 单独设置环境变量

```bash
# 通过命令行设置特定环境变量，该配置主要用于测试
python launch_multi_models.py --models model1 model2 \
    --env VLLM_SKIP_WARMUP=false \
    --env PT_HPU_LAZY_MODE=1 \
    --env VLLM_CONTIGUOUS_PA=false
```

#### 高级配置示例

```bash
python launch_multi_models.py \
    --models /data/models/gte-modernbert-base /data/models/gte-reranker-modernbert-base \
    --port 8771 \
    --max-model-len 8192 \
    --env-preset performance \
    --timeout 600 \
    --log-file custom_server.log
```

#### 查看可用预设

```bash
python launch_multi_models.py --list-env-presets
```

#### 停止所有模型

```bash
python launch_multi_models.py --stop-all
```

### 方法二：手动启动服务

#### 基础启动命令

```bash
VLLM_CONTIGUOUS_PA=false VLLM_SKIP_WARMUP=true python3 -m \
    vllm.entrypoints.openai.mm_api_server \
    --models /models/gte-modernbert-base /models/gte-reranker-modernbert-base \
    --port 8771 \
    --device hpu \
    --dtype bfloat16 \
    --gpu-memory-utilization=0.3 \
    --use-v2-block-manager \
    --max-model-len 4096
```

#### 内存分配策略
内存分配公式：`mem_ratio = 7 // len(models) / 10`, 表示使用70%的显存分配给两个模型，剩余的30%预留作临时缓存。
可以根据模型大小再作调整。
- 2个模型：35% GPU内存/模型

### 方法三：API动态管理

#### 查看当前模型列表

```bash
curl http://localhost:8771/v1/models
```

#### 验证模型可用性

```bash
# 测试Embedding模型
curl http://localhost:8771/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "/data/models/gte-modernbert-base", "input": "San Francisco is a Paris"}'

# 测试Rerank模型
curl localhost:8771/rerank \
    -X POST \
    -d '{"query": "我喜欢什么？", "documents": [
            "昨天没喝水，因为我一直在忙着工作，完全忘记了时间，直到晚上才意识到自己一整天都没有喝水。",
            "我喜欢编程，因为编程不仅能让我解决实际问题，还能让我创造出各种有趣的应用和工具。",
            "你去哪里了？我找了你很久，打了好几个电话你都没有接，是不是发生了什么事情？"
        ],
        "model": "/data/models/gte-reranker-modernbert-base" }' \
    -H 'Content-Type: application/json' 
```

## 客户端配置

### 基础API客户端配置

#### 嵌入模型配置

```python
# 嵌入API端点
embed_url = f"{base_url}/v1/embeddings"

# 嵌入请求数据格式
embed_data = {
    "input": "需要嵌入的文本内容",
    "model": "gte-modernbert-base",
    "truncate_prompt_tokens": -1
}
```

#### 重排序模型配置

```python
# 重排序API端点
rerank_url = f"{base_url}/rerank"

# 重排序请求数据格式
rerank_data = {
    "query": "查询问题",
    "documents": ["文档1内容", "文档2内容", "文档3内容", "文档4内容", "文档5内容"],
    "model": "gte-reranker-modernbert-base",
    "truncate_prompt_tokens": -1
}
```

## 调优参数

### 环境变量调优

#### 使用增强启动脚本的环境变量配置

新的启动脚本提供了三种环境变量配置方式，优先级从高到低为：
1. 命令行单独设置（`--env KEY=value`）
2. JSON配置文件（`--env-config file.json`）
3. 预设配置（`--env-preset default/performance`）

#### 性能预设参数
使用`--env-preset performance`启用以下优化参数，适用于Embedding和Rerank场景：

```bash
# 批处理桶配置 - 优化批处理性能
VLLM_PROMPT_BS_BUCKET_MIN=1       # 最小批处理大小
VLLM_PROMPT_BS_BUCKET_STEP=1      # 批处理步长
VLLM_PROMPT_BS_BUCKET_MAX=8       # 最大批处理大小

# 序列长度桶配置 - 优化不同长度序列的处理
VLLM_PROMPT_SEQ_BUCKET_MIN=1      # 最小序列长度
VLLM_PROMPT_SEQ_BUCKET_STEP=512   # 序列长度步长
VLLM_PROMPT_SEQ_BUCKET_MAX=4096   # 最大序列长度,最长不超过8192


# HPU优化配置
PT_HPU_LAZY_MODE=1               # 启用HPU延迟模式
VLLM_CONTIGUOUS_PA=false         # 禁用连续PA
VLLM_SKIP_WARMUP=false           # 不禁用预热（启用预热以优化性能）
```

#### 不同模型特别调优

针对gte和bge 中小于1B的模型，请将PT_HPU_LAZY_MODE 设置为0，其他参数类型的模型，请使用默认PT_HPU_LAZY_MODE 为1，即可

```
python
python launch_multi_models.py --models /data/models/gte-modernbert-base /data/models/gte-reranker-modernbert-base \
                              --port 8771 --env-preset performance --env "PT_HPU_LAZY_MODE=0"

```
