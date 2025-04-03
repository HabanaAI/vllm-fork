# Enable LLAMA4 on vllm hpu

## deploy
```
docker run -d -it --runtime=habana --name llama4-vllm-1.21 -v /software:/software -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface vault.habana.ai/gaudi-docker/1.20.1/ubuntu24.04/habanalabs/pytorch-installer-2.6.0:latest /bin
/bash

docker exec -it llama4-vllm-1.21 /bin/bash

pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;

# install dependencies for llama4
pip install pydantic msgspec cachetools cloudpickle psutil zmq blake3 py-cpuinfo aiohttp openai uvloop fastapi uvicorn watchfiles partial_json_parser python-multipart gguf llguidance prometheus_client numba compressed_tensors

```

## run example
```
python llama4-scripts/test_vllm.py --model_id /software/stanley/models/llama4-final-v2/Llama-4-Scout-17B-16E-Instruct/ 2>&1 | tee llama4-scripts/llama4_vllm.log
```
