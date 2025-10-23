#!/bin/bash
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
# cd /tmp/vllm-fork-skaul/tests/v1/kv_connector/nixl_integration
cd /tmp/vllm-gaudi-hsub/examples/nixl
PREFILL_HOST="172.26.46.49" # sc09dell06-nvd
# PREFILL_HOST="172.26.47.37" # sc09super16-nvd
PREFILL_PORT=9711
DECODE_HOST="172.26.47.134" # dell03 G3
# DECODE_HOST="172.26.47.41" # dell01 G3
# DECODE_HOST="172.26.47.138" # super17
DECODE_PORT=1909
PROXY_PORT=1295
# Start the proxy server with log redirection and save its PID
python toy_proxy_server.py --port $PROXY_PORT --prefiller-hosts $PREFILL_HOST --prefiller-ports $PREFILL_PORT --decoder-hosts $DECODE_HOST --decoder-ports $DECODE_PORT > /tmp/run_toy_proxy.log 2>&1 &

echo $! > /tmp/run_toy_proxy.pid
echo "Proxy server started with PID: $(cat /tmp/run_toy_proxy.pid)"
echo "Logs are being written to: /tmp/run_toy_proxy.log"

# Wait for the background process
wait
