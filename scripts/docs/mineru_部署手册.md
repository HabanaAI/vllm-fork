## 启动容器
使用下面命令启动容器
```bash
$ docker pull vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
$ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

## 源码安装Mineru
```bash
$ git clone https://github.com/yingjie-han/MinerU.git
$ git checkout release-2.1.0-hpu
$ pip install -e .[core]
```

## 源码安装optimum-habana
```bash
$ git clone https://github.com/huggingface/optimum-habana
$ cd optimum-habana && git checkout 48a2dae1709b50630c6fc93fdf76c52fdfb82566
$ pip install -e .
```

## 模型源配置
MinerU 默认在首次运行时自动从 HuggingFace 下载所需模型。若无法访问 HuggingFace，可通过以下方式切换模型源：

#### 切换至 ModelScope 源

```bash
mineru -p <input_path> -o <output_path> --source modelscope
```

或设置环境变量：

```bash
export MINERU_MODEL_SOURCE=modelscope
mineru -p <input_path> -o <output_path>
```

#### 使用本地模型

##### 1. 下载模型到本地

```bash
mineru-models-download --help
```

或使用交互式命令行工具选择模型下载：

```bash
mineru-models-download
```

下载完成后，模型路径会在当前终端窗口输出，并自动写入用户目录下的 mineru.json。

##### 2. 使用本地模型进行解析

```bash
mineru -p <input_path> -o <output_path> --source local
```

或通过环境变量启用：

```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```


## 在CPU上运行Mineru命令行
```bash
$ mineru -p ./test.pdf -o ./ -m ocr
```

## 在Gaudi上运行Mineru命令行

### 需要修改已安装的doclayout_yolo和ultralytics中的如下代码：
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

### 在hpu上用命令行运行mineru
```bash
$ MINERU_DEVICE_MODE=hpu mineru -p ./test.pdf -o ./ -d hpu -m ocr
```
