# MinerU Gaudi 部署指南

本指南提供在 MinerU v2.5.4 上使用 Intel Gaudi 作为硬件加速器通过 VLLM 后端进行部署的详细步骤。
本部署包括如下模块：
部署minerU vllm gaudi server
部署minerU api 服务
minerU CLI和API 访问方式参考用例

## 前提条件

- 已安装 Intel Gaudi 软件栈
- 支持 Gaudi 对应驱动版本的基础Docker镜像

## 环境准备

### 1. 获取gaudi vllm 源代码和MinerU 源代码

```bash
git clone https://github.com/HabanaAI/vllm-fork.git -b aice/v1.22.0
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
git checkout release-2.5.4
```

### 2. 启动docker
假设host 目录/mnt/disk1 包含 vllm-fork 和MinerU 代码，并选用0号卡部署

```
docker run -it --name minerU-server --runtime=habana -e HABANA_VISIBLE_DEVICES=0 \ 
           -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v /mnt/disk1:/data  \
           --cap-add=sys_nice --net=host --ipc=host --workdir=/workspace \ 
           vault.habana.ai/gaudi-docker/1.21.3/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.21.3-57
```

### 3. 安装vllm 和minerU
注意本步骤需要docker能够访问公网

```bash
cd /data/vllm-fork
 VLLM_TARGET_DEVICE=hpu pip install .

cd /data/MinerU
pip install .[core] -i https://mirrors.aliyun.com/pypi/simple

```

## 环境配置

### 1. 环境变量

部署需要特定的环境变量来优化 Gaudi 性能。创建或执行 `env.sh` 文件：

```bash
#!/bin/bash
export MAX_NUM_SEQS=16
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=True
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0.4
export VLLM_MULTIMODAL_BUCKETS="64,192,384,960,1600,2496,3136,4096,5504,8064,9216"
export MINERU_MODEL_SOURCE=local
export VLLM_CONFIGURE_LOGGING=0
export VLLM_USE_V1=0
```

**关键环境变量说明：**
- `MAX_NUM_SEQS=16`：批处理的最大序列数
- `PT_HPU_LAZY_MODE=1`：启用 HPU 执行的延迟模式
- `VLLM_SKIP_WARMUP=True`：跳过预热以减少启动时间
- `VLLM_GRAPH_RESERVED_MEM=0.2`：为图操作保留 20% 内存
- `VLLM_GRAPH_PROMPT_RATIO=0.4`：为提示处理分配 40% 内存
- `VLLM_MULTIMODAL_BUCKETS`：多模态处理的预定义存储桶大小
- `VLLM_USE_V1`：不使用VLLM V1 engine 避免customer logits 调用出错

### 2. 启动服务

启动 VLLM 服务器和 MinerU API 服务：

```bash
# 在端口 30000 启动 VLLM 服务器
mineru-vllm-server --host 0.0.0.0 --port 30000 --gpu-memory-utilization 0.7 2>&1 | tee -a server.log >/dev/null &

# 声明mineru-vllm-server url 环境变量传递给api server使用
export MINERU_VL_SERVER=http://0.0.0.0:30000
# 在端口 8007 启动 MinerU API 服务器
mineru-api --host 0.0.0.0 --port 8007 2>&1 | tee -a api.log >/dev/null &
```

### 3. 处理 PDF 文档

可以使用两种方式处理 PDF 文档：
minerU CLI

```bash
export MINERU_VL_SERVER=http://0.0.0.0:30000
mineru -p "input.pdf"  -o "output" -b vlm-http-client -u ${MINERU_VL_SERVER}
```

minerU API 参考

```bash
curl -vvv -X POST "http://0.0.0.0:8007/file_parse" \
  -H "Accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/data/tianfeng/minerU/data_set/input/fund.pdf;type=application/pdf" \
  -F "output_dir=./out" \
  -F "lang_list[0]=zh" \
  -F "backend=vlm-http-client" \
  -F "parse_method=ocr" \
  -F "formula_enable=true" \
  -F "table_enable=true" \
  -F "server_url=http://0.0.0.0:30000" \
  -F "return_md=true" \
  -F "return_middle_json=true" \
  -F "return_model_output=true" \
  -F "return_content_list=true" \
  -F "return_images=true" \
  -F "response_format_zip=true" \
  -F "start_page_id=1" \
  -F "end_page_id=10" \
  --output result.zip
```
