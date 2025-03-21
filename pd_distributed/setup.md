# vLLM Disaggregated Prefill with MooncakeStore

## Setup docker 
```bash
docker run -it -d --runtime=habana --name xpyd -v `pwd`:/workspace/ -v /mnt/disk9:/software/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/software/data/ artifactory-kfs.habana-labs.com/docker-local/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.20.0-521 /bin/bash
```

## Installation
### Run inside docker
```bash
docker exec -it xpyd bash
```

### Setup HTTP Proxy
```bash
export http_proxy=http://child-prc.intel.com:913
export https_proxy=http://child-prc.intel.com:913
export no_proxy=10.112.*,localhost,127.0.0.1
```


### Install etcd 
```bash
apt update
apt install sudo etcd -y
cd /workspace
git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git
cd etcd-cpp-apiv3
mkdir build && cd build
cmake ..
make -j$(nproc) && make install
cp /usr/local/lib/libetcd-cpp-api.so /usr/local/include/etcd
```

### Install mooncake
```bash
cd /workspace
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh
mkdir build
cd build
cmake ..
make -j
make install
```

### Install vLLM
#### 1. Clone vLLM
```bash
cd /workspace/
git clone git@github.com:habanaai/vllm-fork.git vllm
```
#### 2. Build
##### 2.1 Build from source
```bash
cd vllm
git checkout dev/pd_dp
pip install -r requirements-hpu.txt
pip install modelscope quart
VLLM_TARGET_DEVICE=hpu python3 setup.py develop
```

## Configuration
### Prepare configuration file to Run Example over TCP
 - Please change the IP addresses and ports in the following guide according to your env.
- Prepare a _**mooncake.json**_ file for both Prefill and Decode instances
```json
{
    "local_hostname": "192.168.0.137",
    "metadata_server": "etcd://192.168.0.137:2379",
    "protocol": "tcp",
    "device_name": "",
    "master_server_address": "192.168.0.137:50001"
}
```

## Run Example
###  1. Start the etcd server
- Please change the IP addresses and ports in the following guide according to your env.
```bash
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379
```

### 2. Start the mooncake_master server
```bash
mooncake_master --port 50001
```

### 3. Run multiple vllm instances
#### Setup environments
```bash
export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/
export VLLM_MLA_DISABLE_REQUANTIZATION=1 
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_EP_SIZE=1
export VLLM_SKIP_WARMUP=True
export VLLM_LOGGING_LEVEL=DEBUG
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MAX_MODEL_LEN=8192
```

#### kv_producer role
```bash
cd /workspace/vllm
MOONCAKE_CONFIG_PATH=./pd_distributed/mooncake.json python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8100 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 -tp 1 --disable-async-output-proc --max-num-seqs 32 --enforce-eager  --trust-remote-code  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
```

#### kv_consumer role
```bash
cd /workspace/vllm
MOONCAKE_CONFIG_PATH=./pd_distributed/mooncake.json python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8200 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 -tp 1 --disable-async-output-proc --max-num-seqs 32 --enforce-eager --trust-remote-code  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
```

#### Start the proxy server
```bash
cd /workspace/vllm
python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py --model $MODEL_PATH --prefill 127.0.0.1:8100 --decode 127.0.0.1:8200  --port 8123
```

- The `--model` parameter specifies the model to use, also specifies the tokenizer used by the proxy server.
- The `--port` parameter specifies the vllm service port on which to listen.
- The `--prefill` or `-p` specifies the ip and port of the vllm prefill instances.
- The `--decode` or `-d` specifies the ip and port of the vllm decode instances.

```bash
# If you want to dynamically adjust the instances of p-nodes and d-nodes during runtime, you need to configure this environment variables.
export ADMIN_API_KEY="xxxxxxxx"

# Then use this command to add instances into prefill group or decode group
curl -X POST "http://localhost:8000/instances/add" -H "Content-Type: application/json" -H "X-API-Key: $ADMIN_API_KEY" -d '{"type": "prefill", "instance": "localhost:8300"}'

curl -X POST "http://localhost:8000/instances/add" -H "Content-Type: application/json" -H "X-API-Key: $ADMIN_API_KEY" -d '{"type": "decode", "instance": "localhost:8301"}'

# Use this command to get the proxy status
curl localhost:8000/status | jq
```

**_Be sure to change the IP address in the commands._**

## Test with openai compatible request
```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "${MODEL_PATH}",
  "prompt": "San Francisco is a",
  "max_tokens": 1000
}'
```
- If you are not testing on the proxy server, please change the `localhost` to the IP address of the proxy server.