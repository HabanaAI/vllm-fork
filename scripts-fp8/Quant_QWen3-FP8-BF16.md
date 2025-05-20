
## 0. Prerequisites

- Driver: 1.20.1 (how to update Gaudi driver: https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html)
- Firmware: 1.20.1 (how to update Gaudi firmware: https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html#system-unboxing-main)
- Docker: vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

## 1. Installation

- VLLM
```bash
git clone recursive https://github.com/vllm-fork.git -b aice/v1.20.1

cd vllm-fork
pip install -e .
```
- INC
```bash
pip install git+https://github.com/intel/neural-compressor.git@qwen-fp8


- VLLM-HPU-EXT
```bash
git clone https://github.com/vllm-hpu-extension-fork.git -b aice/v1.20.1
cd vllm-hpu-extension
pip install -e . 
```

## 2. FP8


- Calibration 

```bash
cd vllm-fork/scripts-fp8
export OFFICIAL_MODEL=/path/to/qwen/model
bash /run_qwen.sh calib ${OFFICIAL_MODEL}
```

```
- Online Serving 

```bash
cd vllm-fork/scripts-fp8
bash 01-benchmark-online-30B-fp8.sh --model_path </path/to/qwen/model> --tp_size <number of cards>
ex.
bash 01-benchmark-online-30B-fp8.sh --model_path /workspace/HF_models/Qwen3-30B-A3B  --tp_size 8
```

Please refer to https://github.com/HabanaAI/vllm-fork/tree/dev/qwen3/scripts for other benchmarks.


## 3. BF16

export BF16_OR_VLLM_FP8_STATIC_MOE_ENABLED=True