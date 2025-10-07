#!/bin/bash

python3 -m vllm.entrypoints.openai.api_server \
--port 8080 \
--model "ibm-granite/granite-3.3-8b-instruct" \
--tensor-parallel-size 1 \
--max-num-seqs 8 \
--dtype bfloat16 \
--gpu-memory-util 0.9 \
--enable-auto-tool-choice \
--tool-call-parser granite \
--chat-template /software/stanley/benchmark-config/vllm-fork-granite3.3-8b/examples/tool_chat_template_granite.jinja \
--max-model-len 8192 \
--block-size 128