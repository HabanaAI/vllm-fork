# Multi Models

## How to Run

```
$ python scripts/launch_multi_models.py --models mistralai/Mistral-7B-Instruct-v0.3 meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
```

* expected output - doesn't match with previous settings
```
python scripts/launch_multi_models.py --models mistralai/Mistral-7B-Instruct-v0.3 meta-llama/Llama-3.1-8B-Instruct
running_models ['mistralai/Mistral-7B-Instruct-v0.3', 'meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen2.5-7B-Instruct']
Running models do not match the requested models. Restarting server.
Killed existing process with PID: 2871900
Launched new server with PID: 2874940
Waiting for server to start. Monitoring mm_vllm_server.log for output...
.....................................Server started successfully! Took 37.04 seconds.
running_models ['mistralai/Mistral-7B-Instruct-v0.3', 'meta-llama/Llama-3.1-8B-Instruct']

```

* expected output - match with previous settings
```
running_models ['mistralai/Mistral-7B-Instruct-v0.3', 'meta-llama/Llama-3.1-8B-Instruct']
Server is already running with the requested models. No action required.
```