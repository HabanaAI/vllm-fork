#!/bin/bash
# This file demonstrates the example usage of multimodal models for single batch, multi batch, video, images, text-only

##set -xe

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# a function that waits on vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false 

# Launch the server
#vllm serve "Qwen/Qwen2.5-VL-72B-Instruct" --task generate --trust-remote-code --tensor-parallel-size 8 &
vllm serve "Qwen/Qwen2.5-VL-7B-Instruct" --task generate --trust-remote-code &
# wait until instance is ready
wait_for_server 8000

# serve requests
output_text=$(python3 ./openai_chat_completion_client_for_multimodal.py -c text-only)
echo "-------------------------------------"
echo "Output of text request: $output_text"
echo "-------------------------------------"

output_single=$(python3 ./openai_chat_completion_client_for_multimodal.py -c single-image --image-folder ../../images)
#output_single=$(curl http://localhost:8000/v1/chat/completions \
#    -H "Content-Type: application/json" \
#    -d '{
#    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
#    "messages": [
#    {"role": "system", "content": "You are a helpful assistant."},
#    {"role": "user", "content": [
#        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
#        {"type": "text", "text": "What is the text in the illustrate?"}
#    ]}
#    ]
#    }')
echo "-------------------------------------"
echo "Output of single image request: $output_single"
echo "-------------------------------------"
#output_multi=$(curl http://localhost:8000/v1/chat/completions \
#    -H "Content-Type: application/json" \
#    -d '{
#    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
#    "messages": [
#    {"role": "system", "content": "You are a helpful assistant."},
#    {"role": "user", "content": [
#        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
#        {"type": "text", "text": "What is the text in the illustrate?"}
#    ]}
#    ]
#    }')
##output_multi=$(python3 ./openai_chat_completion_client_for_multimodal.py -c multi-image --image-folder ../../images)
##echo "-------------------------------------"
##echo "Output of multi image request:$output_multi"
##echo "-------------------------------------"

# pip install video requirements
##pip install -e "../../[video]"
##output_video=$(python3 ./openai_chat_completion_client_for_multimodal.py -c video)
##echo "-------------------------------------"
##echo "Output of video request:$output_video"
##echo "-------------------------------------"

# Cleanup commands
pgrep python | xargs kill -9
pkill -f python

echo ""

sleep 1

echo "🎉🎉 Successfully finished test requests! 🎉🎉"
echo ""
