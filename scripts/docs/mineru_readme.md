## Run Docker Image
Use the following commands to run a Docker image.
```bash
$ docker pull vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
$ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
```

## Install MinerU from source
```bash
$ git clone https://github.com/yingjie-han/MinerU.git
$ git checkout release-2.1.0-hpu
$ pip install -e .[core]
```

## Install optimum-habana from source
```bash
$ git clone https://github.com/huggingface/optimum-habana
$ cd optimum-habana && git checkout 48a2dae1709b50630c6fc93fdf76c52fdfb82566
$ pip install -e .
```

## Model Source Configuration

MinerU automatically downloads required models from HuggingFace on first run. If HuggingFace is inaccessible, you can switch model sources:

#### Switch to ModelScope Source

```bash
mineru -p <input_path> -o <output_path> --source modelscope
```

Or set environment variable:

```bash
export MINERU_MODEL_SOURCE=modelscope
mineru -p <input_path> -o <output_path>
```

#### Using Local Models

##### 1. Download Models Locally

```bash
mineru-models-download --help
```

Or use interactive command-line tool to select models:

```bash
mineru-models-download
```

After download, model paths will be displayed in current terminal and automatically written to `mineru.json` in user directory.

##### 2. Parse Using Local Models

```bash
mineru -p <input_path> -o <output_path> --source local
```

Or enable via environment variable:

```bash
export MINERU_MODEL_SOURCE=local
mineru -p <input_path> -o <output_path>
```


## Runing pipeline on CPU
```bash
$ mineru -p ./test.pdf -o ./ -m ocr
```

## Runing pipeline on Gaudi

### modify code in doclayout_yolo and ultralytics for hpu
```bash
vim /usr/local/lib/python3.10/dist-packages/doclayout_yolo/engine/predictor.py
vim /usr/local/lib/python3.10/dist-packages/ultralytics/engine/predictor.py
```
Change the device parameter in setup_model() as following:

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
Add the  hpu device branch in warmup() as following:
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

### Runing pipeline on hpu
```bash
$ MINERU_DEVICE_MODE=hpu mineru -p ./test.pdf -o ./ -d hpu -m ocr
```
