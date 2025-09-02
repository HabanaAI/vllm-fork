- QWEN3-30B-3A 1 card
- scalar w/ h2d: hang
- scalar w/o h2d:
- const 


- const
warmup scenario warmup_prompt_bs1_seq8192_ctx0_multimodalF_graphsT took 11.28 seconds
- scalar w/ runtime patching
warmup scenario warmup_prompt_bs1_seq8192_ctx0_multimodalF_graphsT took 293.59 seconds
- scalar w/o runtime patching
warmup scenario warmup_prompt_bs1_seq8192_ctx0_multimodalF_graphsT took 289.47 seconds


- Image: artifactory-kfs.habana-labs.com/docker-local/1.23.0/ubuntu24.04/habanalabs/pytorch-installer-2.8.0:1.23.0-248

# Installation
``` bash
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout ds-example

pip install -r requirements/hpu.txt
VLLM_TARGET_DEVICE=hpu python3 setup.py develop --user
```

# Reproduce CMD
```bash
cd /examples/offline_inference/deepseek
bash reproduce_cmd.sh
```