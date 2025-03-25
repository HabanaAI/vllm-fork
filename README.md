# install

```
docker run -d -it --runtime=habana --name vllm-gaudi1.20  -v `pwd`:/workspace/  -v /scratch-1:/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface vault.habana.ai/gaudi-docker/1.20.0/ubuntu24.04/habanalabs/pytorch-installer-2.6.0:latest /bin/bash

docker exec -it vllm-gaudi1.20 /bin/bash

apt update; apt install python3-is-python
pip install datasets lm_eval compress_pickle
```

```
git clone https://github.com/HabanaAI/vllm-fork.git;
cd vllm;  pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;
```

# run benchmark

## 70B

### quick example test
```

```

### step 3. run benchmark
```
bash benchmark-staticfp8-i1k-o1k-ep8-bestperf.sh
```

## 405B

### 1. quick example test
```
python run_example_405B.py
```

### 2. acc check
```
VLLM_DELAYED_SAMPLING=true \
VLLM_SKIP_WARMUP=true \
VLLM_SOFTMAX_CONST_NORM=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
QUANT_CONFIG=quant_files/inc/meta-llama-3.1-405b-instruct-v2/maxabs_quant_g3.json \
lm_eval --model vllm  --model_args "pretrained=/data/Llama-3.1-405B-Instruct,tensor_parallel_size=8,distributed_executor_backend=ray,trust_remote_code=true,max_model_len=4096" --tasks gsm8k --num_fewshot 5 --limit 64 --batch_size 1
```

### 3. run benchmark
```
bash benchmark_serving-405B.sh
```