#!/bin/bash
export VLLM_TORCH_PROFILER_DIR=/tmp
export VLLM_SKIP_WARMUP=true
rm -f "$VLLM_TORCH_PROFILER_DIR"/*.pt.trace.json.gz


asynch_test() {
  echo "profiling asynch test"
  HABANA_PROFILE=1 python3 -m vllm.entrypoints.openai.api_server --port 8080 --model "facebook/opt-125m" &
  server_pid=$!
  # wait for server to start, timeout after 120 seconds
  timeout 120 bash -c 'until curl localhost:8080/v1/models; do sleep 1; done' || exit 1
  sleep 1
  curl -X POST http://localhost:8080/start_profile
  sleep 1
  curl http://localhost:8080/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
          "model": "facebook/opt-125m",
          "prompt": "San Francisco is a",
          "max_tokens": 7,
          "temperature": 0
      }'
  sleep 1
  curl -X POST http://localhost:8080/stop_profile
  sleep 120
  kill $server_pid
}

script_test() {
  echo "profiling script test"
  python vllm_demo_script.py
}

usage() {
    echo
    echo "Runs simple profiling test"
    echo
    echo "usage: $(basename "$0") <options>"
    echo
    echo "  -a   - run simple asynch profiling test"
    echo "  -s   - run simple profiling from script test"
    echo
}


mode=""
while getopts ":ash" opt; do
  case "$opt" in
    a) mode="asynch"  ;;
    s) mode="script"  ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG"; usage; exit 2 ;;
  esac
done

if [[ -z "$mode" ]]; then
  usage
  exit 2
fi

case "$mode" in
  asynch) asynch_test ;;
  script) script_test ;;
esac


found_file=false
for file in "$VLLM_TORCH_PROFILER_DIR"/*.pt.trace.json.gz; do
  if [[ -s "$file" ]]; then
    found_file=true
    break
  fi
done

if [[ "$found_file" == true ]]; then
  echo "PASS. Trace file is created"
  exit 0
else
  echo "FAIL. Trace file is not created"
  exit 1
fi
