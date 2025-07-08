# LMCache Examples
Please Note: HPU integration for LMCache will be upstreamed. After that, the following test cases can be used.

This folder demonstrates how to use LMCache for disaggregated prefilling, CPU offloading and KV cache sharing.

## 1. Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using lm or redis on a single node.

### Prerequisites
- HPU implementation to be upstreamed
- At least 2 HPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

### Usage

Run
`cd disagg_prefill_lmcache_v1`
to get into `disagg_prefill_lmcache_v1` folder, and then run

```bash
bash disagg_example_nixl.sh
```

to run disaggregated prefill and benchmark the performance.

### Components

#### Server Scripts
- `disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `../disagg_prefill_lmcache_v1/disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_prefill_lmcache_v1/disagg_example.sh` - Main script to run the example through lm/redis remote server

#### Configuration
- `disagg_prefill_lmcache_v1/configs/lmcache-config-lm.yaml` - Configuration for prefiller/decoder server through lm server
- `disagg_prefill_lmcache_v1/configs/lmcache-config-redis.yaml` - Configuration for prefill/decoder server through redis server

#### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server

## 2. CPU Offload Examples

- `python cpu_offload_lmcache.py -v v1` - CPU offloading implementation for vLLM v1

## 3. KV Cache Sharing

The `kv_cache_sharing_lmcache_v1.py` example demonstrates how to share KV caches between vLLM v1 instances.

## 4. Disaggregated Prefill in vLLM v0

The `disaggregated_prefill_lmcache_v0.py` provides an example of how to run disaggregated prefill in vLLM v0.
