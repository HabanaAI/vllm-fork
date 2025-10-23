#!/bin/bash
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
cd /tmp/vllm-fork-skaul/tests/v1/kv_connector/nixl_integration
PREFILL_HOST="172.26.47.37" # sc09super16-nvd
PREFILL_PORT=9700
DECODE_HOST="172.26.47.41" # super17 G2

DECODE_PORT=9800
PROXY_PORT=8888
# Start the proxy server with log redirection and save its PID
python toy_proxy_server.py --port $PROXY_PORT --prefiller-hosts $PREFILL_HOST --prefiller-ports $PREFILL_PORT --decoder-hosts $DECODE_HOST --decoder-ports $DECODE_PORT > /tmp/run_toy_proxy.log 2>&1 &

echo $! > /tmp/run_toy_proxy.pid
echo "Proxy server started with PID: $(cat /tmp/run_toy_proxy.pid)"
echo "Logs are being written to: /tmp/run_toy_proxy.log"

# Wait for the background process
wait
