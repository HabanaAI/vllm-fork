# Guide to Setting Up PD Disaggregation with Mooncake

> **Note**: This document does not cover MLA data parallel setup.

## Mooncake Installation

### Install via pip

```bash
apt install etcd -y
pip3 install mooncake-transfer-engine==0.3.0
```

### Install from Source

```bash
# Install required packages
apt install git wget curl net-tools sudo iputils-ping etcd -y

# Clone the Mooncake repository
git clone https://github.com/kvcache-ai/Mooncake.git -b v0.3.0
cd Mooncake

# Install dependencies
bash dependencies.sh

# Build and install Mooncake
mkdir build
cd build
cmake ..
make -j
make install
```

## PD Disaggregation Usage

### 1. Prepare and Modify `mooncake.json`

Create and configure the `mooncake.json` file:

```json
{
    "local_hostname": "192.168.0.137",
    "metadata_server": "etcd://192.168.0.137:2379",
    "protocol": "tcp",
    "device_name": "",
    "master_server_address": "192.168.0.137:50001"
}
```

- Update `metadata_server` with the etcd address.
- Update `master_server_address` with the Mooncake store master address. Use a high-speed network for optimal KV cache data transfer performance.
- Update `local_hostname` with the node's IP address.

> **Note**: The `mooncake.json` configuration differs for prefill and decode instances. Update the IP addresses accordingly based on the specific setup.

### 2. Start the etcd Server on the Master Node

```bash
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379 > etcd.log 2>&1 &
```

### 3. Start the `mooncake_master` Server on the Master Node

```bash
mooncake_master --enable_gc true --port 50001
```

### 4. Start the Prefill Instance

```bash
export MOONCAKE_CONFIG_PATH=./mooncake.json
export PT_HPU_LAZY_MODE=1 

python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1 \
    --port 8100 \
    -tp 8 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
```

### 5. Start the Decode Instance

```bash
export MOONCAKE_CONFIG_PATH=./mooncake.json
export PT_HPU_LAZY_MODE=1 

python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1 \
    --port 8200 \
    -tp 8 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
```

### 6. Start the Proxy Server

```bash
python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model deepseek-ai/DeepSeek-R1 \
    --prefill $prefill_node_ip:8100 \
    --decode $decode_node_ip:8200 \
    --port 8000
```

- Set `prefill_node_ip` and `decode_node_ip` based on the actual machine addresses.

## Testing

### Test with OpenAI-Compatible Request

```bash
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "prompt": "San Francisco is a",
    "max_tokens": 1000
}'
```

- If not testing on the proxy server, replace `localhost` with the proxy server's IP address.

### Test Accuracy with `lm_eval`

```bash
pip install lm_eval
pip install lm-eval[api]

lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args model=deepseek-ai/DeepSeek-R1,base_url=http://localhost:8000/v1/completions,num_concurrent=1 \
    --batch_size 1 \
    --log_samples \
    --output_path ./lm_eval_output \
    --num_fewshot 3
```
