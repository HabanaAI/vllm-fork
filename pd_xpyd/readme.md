Note: this doc didn't cover mla data parallel!!!

# <a name="s1">setup docker and install mooncake (single node)</a>

Please refer Chaojun's https://github.com/chaojun-zhang/vllm/blob/c9154592820bf1375a030e46e0334b83ac36287b/pd_distributed/setup.md 

0. make sure proxy works in intel env!
1. install mooncake [link](#https://github.com/kvcache-ai/Mooncake/?tab=readme-ov-file#-quick-start)

```
apt install git wget curl net-tools sudo iputils-ping etcd  -y

git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake

bash dependencies.sh

mkdir build
cd build
cmake ..
make -j
make install
```

## How to Run

0. prepare and modify mooncake.json

modify `metadata_server` as etcd address
modify `master_server_address` as mooncake store master address, better use high speed network for this since this will influence kv cache data transer performance.

modify `local_hostname` as node ip

1. start etcd server on master node
```
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379  >etcd.log 2>&1 &
```

2. start Mooncake Store on master node
```
mooncake_master -enable_gc true -port 50001
```

3. start Prefill and Decode instance
refer `start_prefill.sh`, `start_decode.sh`
nencessary env/paras are:
```
# some mooncake components will install to here
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

MOONCAKE_CONFIG_PATH=./pd_xpyd/mooncake.json

# for prefill
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}' 
--port 8100

# for decode
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
--port 8200
```

4. start proxy server
refer `start_proxy.sh`, modify parameters accordingly.


# enbale RDMA in mooncake (multiple Gaudi2 vm nodes)

Please refer to [setup docker and install mooncake (single node)](#s1) for general guideline of installing this vllm fork and mooncake for PD distribution.

To speed up KV transfer between prefill and encode nodes, this section guides you how to enable RDMA in mooncake's transfer engine in multiple Gaudi2 vm nodes.

## build and install vllm and mooncake

0. first of all, login to your gaudi dev vm and try to install everything underneath your home directory so that artifacts can be shared in later assigned multiple vm nodes.

1. setup qnpu environment with pytorch 1.20.0 (assuming you are familiar with Gaudi qnpu)
```
# only 1.20.0 is verified for now. for latest 1.22, it doesn't work with deepseek moe.
qnpu-init.sh --pytorch -v 1.20.0 -b 521 -m pytorch-stack.xml pt-int-1.20
# verify pytorch install by importing torch in non-lazy mode
PT_HPU_LAZY_MODE=0 python
>>> import torch
```

2. install mooncake to $HOME/.local
```
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local -DCMAKE_CXX_FLAGS="-I $HOME/.local/include -DGLOG_USE_GLOG_EXPORT" -DCMAKE_C_FLAGS="-I $HOME/.local/include"

GO111MODULE=on make -j
make install

```
Depending on version and your local env, you may get some dependency issues, like glog, gflags and go version. You can follow <a href="https://intel.sharepoint.com/sites/aisepytorchgaudi/_layouts/OneNote.aspx?id=%2Fsites%2Faisepytorchgaudi%2FSiteAssets%2FAISE%20PyTorch%20Gaudi%20Notebook&wd=target%28General.one%7C1D4489F0-A2A4-43A9-8704-58AC434C6690%2FMooncake%20RDMA%20Setup%20in%20Gaudi%20VM%7CE6A83F9A-23EB-471F-90E4-C591A12372F4%2F%29
onenote:https://intel.sharepoint.com/sites/aisepytorchgaudi/SiteAssets/AISE%20PyTorch%20Gaudi%20Notebook/General.one#Mooncake%20RDMA%20Setup%20in%20Gaudi%20VM&section-id={1D4489F0-A2A4-43A9-8704-58AC434C6690}&page-id={E6A83F9A-23EB-471F-90E4-C591A12372F4}&end">Mooncake RDMA Setup in Gaudi VM</a> for details.

3. install etcd
```
wget https://github.com/etcd-io/etcd/releases/download/v3.5.21/etcd-v3.5.21-linux-amd64.tar.gz
tar -xf etcd-v3.5.21-linux-amd64.tar.gz
mv etcd-v3.5.21-linux-amd64/ etcd-v3.5.21
mv etcd-v3.5.21/ ~/.local
export PATH=$HOME/.local/etcd-v3.5.21/:$PATH
```

4. install vLLM
```
git clone git@github.com:habanaai/vllm-fork.git vllm
cd vllm
git checkout dev/pd_dp
pip install -r requirements/hpu.txt
pip install modelscope quart
VLLM_TARGET_DEVICE=hpu python3 setup.py develop
```

## deploy vLLM in gaudi vm in PD distribution way
0. request three Gaudi2 vms (each with 1 g2 card. 8 cards is hard to request)
```
hlctl create vm --flavor g2 --namespace framework --replication 3 -s 'habana.ai/site=idc'

# ssh to all of them and install drivers
# you can use "build_and_insmod_habanalabs -i" if it's already built.
build_and_insmod_habanalabs

# unset proxies
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

```
Check mellanox drivers and make sure you see the below mapping and status ("Up")
```
$ ibdev2netdev
mlx5_0 port 1 ==> eth0 (Up)
mlx5_1 port 1 ==> eth1 (Up)
mlx5_2 port 1 ==> eth2 (Up)
```

1. startup etcd service and mooncake store master in the first vm
```
# 10.111.232.134 is ip of 'eth0'. You can get it by 'ifconfig'.
nohup etcd --listen-client-urls http://0.0.0.0:2379  --advertise-client-urls http://10.111.232.134:2379 --data-dir /tmp/etcd >etcd.log 2>&1 &

nohup mooncake_master --port 50001 > master.log 2>&1 &
```

2. prepare vLLM envs in the second and the third vms
```
export MODEL_PATH=/software/users/kepingyan/project/DeepSeek-R1-G2-static/
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_EP_SIZE=1
export VLLM_SKIP_WARMUP=True
export VLLM_LOGGING_LEVEL=DEBUG
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MAX_MODEL_LEN=8192
```
save above envs in file kv_envs.sh which will be sourced in same terminals where you startup prefill and decode servers

3. prepare mooncake config files for prefill and decode

Note: You need to replace all IPs with your actual IPs.

mooncake_prefill.json
```
{
    "local_hostname": "10.111.232.107", // ip of eth0 in the second vm (for prefill)
    "metadata_server": "etcd://10.111.232.134:2379",
    "protocol": "rdma",
    "device_name": "mlx5_1", // use the second mlx device, the first device is mapped to eth0
    "master_server_address": "10.111.232.134:50001"
}
```
mooncake_decode.json
```
{
    "local_hostname": "10.111.232.106", // ip of eth0 in the third vm (for decode)
    "metadata_server": "etcd://10.111.232.134:2379",
    "protocol": "rdma",
    "device_name": "mlx5_1", // use the second mlx device, the first device is mapped to eth0
    "master_server_address": "10.111.232.134:50001"
}
```

4. startup prefill server in the second vm

Since we only have 1 card in each vm, we cannot load all weights of deepseek R1. Thus, we make below changes to load only 4 layers. After this change, the generation is not accurate. But it doesn't affect our RDMA enabling.
```
diff --git a/vllm/model_executor/models/deepseek_v2.py b/vllm/model_executor/models/deepseek_v2.py
index b06594fac..9b883ece4 100644
--- a/vllm/model_executor/models/deepseek_v2.py
+++ b/vllm/model_executor/models/deepseek_v2.py
@@ -609,7 +609,8 @@ class DeepseekV2Model(nn.Module):
             self.embed_tokens = PPMissingLayer()

         self.start_layer, self.end_layer, self.layers = make_layers(
-            config.num_hidden_layers,
+            #config.num_hidden_layers,
+            4,
             lambda prefix: DeepseekV2DecoderLayer(
                 config,
                 prefix,

```
Then, startup prefill server
```
source kv_envs.sh
# make sure host ip is correct
MOONCAKE_CONFIG_PATH=./mooncake_prefill.json python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --host 10.111.232.107 --port 8100 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 -tp 1 --disable-async-output-proc --max-num-seqs 32 --enforce-eager --trust-remote-code --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
```

5. startup decode server in the third vm
```
source kv_envs.sh
# make sure host ip is correct
MOONCAKE_CONFIG_PATH=./mooncake_decode.json python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --host 10.111.232.106 --port 8200 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 -tp 1 --disable-async-output-proc --max-num-seqs 32 --enforce-eager --trust-remote-code --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
```

6. startup proxy server in the first vm
```
source kv_envs.sh
cd vllm/vllm
python3 ./examples/online_serving/disagg_examples/disagg_proxy_demo.py --model $MODEL_PATH --prefill 10.111.232.107:8100 --decode 10.111.232.106:8200 --port 8123
```

7. request generation in the first vm
```
curl -s http://localhost:8123/v1/completions -H "Content-Type: application/json" -d '{
    "model": "/software/users/kepingyan/project/DeepSeek-R1-G2-static/",
    "prompt": "San Francisco is a",
    "max_tokens": 1000
}'
```
