
1. setup and start prefill server, refer pd_distributed/prefill_setup.sh (etcd server also start in this script)

2. setup and start decode server, refer pd_distributed/decode_setup.sh

3. start proxy_server
```
python3 pd_distributed/proxy_server.py
```

4. run benchmark_serving as usual.


role in this poc:

Proxy server: 10.112.110.50:8123​

Prefill server: 10.112.110.50:8100​

Decode server: 10.112.110.51:8200​

Etcd server: 10.112.110.50:2379​

for faster tcp data transport, please change mooncake.json to leverage high speed network.

Note: you may need change route or network config to force use fastest net interface. Otherwise performance will not be optimal.