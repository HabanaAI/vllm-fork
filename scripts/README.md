## How to deploy
```bash
docker run -d -it --runtime=habana --name qwen3-vllm-1.20  -v `pwd`:/workspace/vllm/  -v /data:/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest /bin/bash

docker exec -it qwen3-vllm-1.20 /bin/bash
cd /workspace/vllm/

git clone -b dev/qwen3 https://github.com/HabanaAI/vllm-fork.git;
cd vllm-fork; pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;
pip install lm_eval datasets numpy==1.26.4
```

## Run with BF16

### Evaluate accuracy
```bash
cd scripts;
bash 02-accuracy_mmlupro.sh --model_path ${model_path} --tp_size 1
bash 03-accuracy_gsm8k.sh --model_path ${model_path} --tp_size 1
```

### Benchmark
```bash
cd scripts;
bash 01-benchmark-online-30B.sh --model_path ${model_path}
bash 01-benchmark-online-32B.sh --model_path ${model_path}
bash 01-benchmark-online-235B.sh --model_path ${model_path}
bash 01-benchmark-online-235B-tp8.sh --model_path ${model_path}
```

## Run with FP8

### Calibration 

```bash
cd vllm-fork/scripts
export OFFICIAL_MODEL=/path/to/qwen/model
# --tokenizer is optional
QUANT_CONFIG=inc_meaure_g3_235B_A22B.json bash run_qwen.sh calib ${OFFICIAL_MODEL} --tokenizer ${OFFICIAL_MODEL} --tp_size 8 --ep_size 8
```

### Quantization 
```bash
cd vllm-fork/scripts
export OFFICIAL_MODEL=/path/to/qwen/model
# --tokenizer is optional
QUANT_CONFIG=inc_quant_g3_235B_A22B.json bash run_qwen.sh quant ${OFFICIAL_MODEL} --tokenizer ${OFFICIAL_MODEL} --tp_size 8 --ep_size 8
```

### Evaluate accuracy
```bash
cd scripts;
QUANT_CONFIG=inc_quant_g3_235B_A22B.json bash 02-accuracy_mmlupro_fp8.sh --model_path ${model_path} --tp_size 8 --ep_size 8
QUANT_CONFIG=inc_quant_g2_235B_A22B.json bash 03-accuracy_gsm8k_fp8.sh --model_path ${model_path} --tp_size 8 --ep_size 8
```

### Benchmark
```bash
cd scripts;
bash 01-benchmark-online_fp8_30B.sh  --model_path ${model_path}
bash 01-benchmark-online_fp8_32B.sh  --model_path ${model_path}
bash 01-benchmark-online_fp8_235B.sh  --model_path ${model_path}
```
