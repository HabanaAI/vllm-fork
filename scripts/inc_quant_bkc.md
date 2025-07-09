## 0. Prerequisites

- Docker Synapse 1.21, vault.habana.ai/gaudi-docker/1.21.0/ubuntu24.04/habanalabs/pytorch-installer-2.6.0:latest

## 1. Installation

- vLLM

```bash
git clone -b aice-121-qwen https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
pip install -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation
```

### 2 FP8 KV + Per-Channel Quantization

- Get calibration files

```bash
cd vllm-fork/scripts
pip install -U "huggingface_hub[cli]"
huggingface-cli download Yi30/DeepSeek-R1-Distill-Qwen-32B-pile-512-g2-tp1-0707-post  --local-dir nc_workspace_measure_kvache_post
```

- Running the Benchmark

```bash
cd vllm-fork/scripts
export MODEL=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
export QUANT_CONFIG=inc_quant_post.json
PT_HPU_LAZY_MODE=1 \
VLLM_SKIP_WARMUP=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
python ./run_example_tp_qwen.py \
    --model $MODEL \
    --tokenizer $MODEL \
    --osl 32 --max_model_len 2048 \
    --max_num_seqs 1 \
    --tp_size 1 --ep_size 1  \
    --inc --fp8_kv_cache
```