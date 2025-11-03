# MinerU Gaudi 部署指南

本指南提供在 MinerU v2.5.4 上使用 Intel Gaudi 作为硬件加速器通过 pipeline后端/VLLM 后端进行部署的详细步骤。

## 前提条件

- 已安装 Intel Gaudi 软件栈
- 支持 Gaudi 对应驱动版本的基础Docker镜像


## 使用MinerU v2.5.4

### 1. 启动docker
```
docker run -it --name minerU-server --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
           -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v /mnt/disk1:/data  \
           --cap-add=sys_nice --net=host --ipc=host --workdir=/workspace \
           vault.habana.ai/gaudi-docker/1.21.3/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.21.3-57
```

### 2. 源码安装 MinerU
为容器设置正确的网络设置，确保容器可以正常访问 github。

```bash
git clone https://gitee.com/intel-china/aisolution-mineru.git
cd aisolution-mineru
git checkout release-2.5.4-hpu
pip install -e .[core] -i https://mirrors.aliyun.com/pypi/simple
```

### 3. 下载模型和更新minerU.json

模型下载请参考https://opendatalab.github.io/MinerU/usage/model_source/#1-download-models-to-local-storage。
选取modelscope或者huggingface下载模型，以下以modelscope为例

```bash
mineru-models-download -s modelscope
```

根据模型下载位置，更新mineru.json。可参考
https://github.com/opendatalab/MinerU/blob/master/mineru.template.json

需要更新的配置为models-dir，这个可以指定到模型目录，如果不指定会使用modelscope默认/root/.cache/modelscope

```json
    "models-dir": {
        "pipeline": "",
        "vlm": ""
    },
```

### 4. 使用MinerU vlm-vllm-engine backend

#### 4.1 为vlm-vllm-engine backend安装vllm
```bash
git clone https://github.com/HabanaAI/vllm-fork.git -b aice/v1.22.0
pip install -e . -i https://mirrors.aliyun.com/pypi/simple
git clone https://github.com/vllm-project/vllm.git
cp -r vllm/vllm/v1/sample/logits_processor vllm-fork/vllm/v1/sample/logits_processor
```

#### 4.2 在Gaudi上运行vlm-vllm-engine backend

#### 4.2.1 设置环境变量

部署需要特定的环境变量来优化 Gaudi 性能。创建或执行 `env.sh` 文件：

```bash
#!/bin/bash
export MAX_NUM_SEQS=16
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=True
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0.4
export VLLM_MULTIMODAL_BUCKETS="64,192,384,512,640,768,896,1024,1152,1280,1408,1536,1664,2496, 3136, 4096, 5504, 6272, 7104, 8192, 9216"
export MINERU_MODEL_SOURCE=local
export VLLM_CONFIGURE_LOGGING=0
export VLLM_USE_V1=0
export VLLM_FP32_SOFTMAX=true
```

**关键环境变量说明：**
- `MAX_NUM_SEQS=16`：批处理的最大序列数
- `PT_HPU_LAZY_MODE=1`：启用 HPU 执行的延迟模式
- `VLLM_SKIP_WARMUP=True`：跳过预热以减少启动时间
- `VLLM_GRAPH_RESERVED_MEM=0.2`：为图操作保留 20% 内存
- `VLLM_GRAPH_PROMPT_RATIO=0.4`：为提示处理分配 40% 内存
- `VLLM_MULTIMODAL_BUCKETS`：多模态处理的预定义存储桶大小
- `VLLM_USE_V1`：不使用VLLM V1 engine 避免customer logits 调用出错
- `VLLM_FP32_SOFTMAX` ： SDPA doftmax计算使用FP32精度

#### 4.2.2 使用命令行方式运行Mineru
```bash
mineru -p <input_path> -o <output_path>  -b vlm-vllm-engine 
```

#### 4.2.3 使用 http-client/server 方式运行Mineru

##### 启动 vllm server 和 api server
```bash
# 在端口 30000 启动 VLLM 服务器
mineru-vllm-server --host 0.0.0.0 --port 30000 --gpu-memory-utilization 0.7 2>&1 | tee -a server.log >/dev/null &
```
##### 使用 minerU CLI 处理文档
```bash
export MINERU_VL_SERVER=http://0.0.0.0:30000
mineru -p "input.pdf"  -o "output" -b vlm-http-client -u ${MINERU_VL_SERVER}
```

##### 使用 minerU API 处理文档
```bash
# 声明mineru-vllm-server url 环境变量传递给api server使用
export MINERU_VL_SERVER=http://0.0.0.0:30000
# 在端口 8007 启动 MinerU API 服务器
mineru-api --host 0.0.0.0 --port 8007 2>&1 | tee -a api.log >/dev/null &
```
minerU API 参考

```bash
curl -vvv -X POST "http://0.0.0.0:8007/file_parse" \
  -H "Accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/data/test.pdf;type=application/pdf" \
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


### 5. 使用MinerU Pipeline backend

#### 5.1 为Pipeline backend安装optimum-habana
```bash
$ git clone https://github.com/huggingface/optimum-habana
$ cd optimum-habana && git checkout 48a2dae1709b50630c6fc93fdf76c52fdfb82566
$ pip install -e . -i https://mirrors.aliyun.com/pypi/simple
```

#### 5.2 为支持hpu修改已安装的doclayout_yolo和ultralytics代码
```bash
vim /usr/local/lib/python3.10/dist-packages/doclayout_yolo/engine/predictor.py
vim /usr/local/lib/python3.10/dist-packages/ultralytics/engine/predictor.py
```
参照下面修改setup_model()函数中的device参数:

```bash
    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        if self.args.device == "hpu":
            device = self.args.device
        else:
            device = select_device(self.args.device, verbose=verbose)
        self.model = AutoBackend(
            weights=model or self.args.model,
            #device=select_device(self.args.device, verbose=verbose),
            device=device,
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=False,
            verbose=verbose,
        )
```

```bash
vim /usr/local/lib/python3.10/dist-packages/doclayout_yolo/nn/autobackend.py
vim /usr/local/lib/python3.10/dist-packages/ultralytics/nn/autobackend.py
```
参照如下修改warmup()，为其增加if self.device  == "hpu"分支:
```bash
    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        #if any(warmup_types) and (self.device.type != "cpu" or self.triton):
        if self.device  == "hpu":
            im = torch.empty(*imgsz, dtype=torch.bfloat16 if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup
        elif any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup
```

#### 5.3 使用命令行方式运行Mineru
```bash
$ MINERU_DEVICE_MODE=hpu mineru -p ./test.pdf -o ./ -d hpu  -b pipeline -m ocr
```