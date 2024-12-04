# Multi Models

## How to Run

### method 1 - Use scripts

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

### method 2 - Use server API

step 1. List current models - Mistral-7B-Instruct-v0.3 and Llama3.1-8B-Instruct

```
$ curl http://localhost:8000/v1/models
{"object":"list","data":[{"id":"mistralai/Mistral-7B-Instruct-v0.3","object":"model","created":1736483684,"owned_by":"vllm","root":"mistralai/Mistral-7B-Instruct-v0.3","parent":null,"max_model_len":4096,"permission":[{"id":"modelperm-ccfc4677b19a45428c6be738df76da70","object":"model_permission","created":1736483684,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]},{"id":"meta-llama/Llama-3.1-8B-Instruct","object":"model","created":1736483684,"owned_by":"vllm","root":"meta-llama/Llama-3.1-8B-Instruct","parent":null,"max_model_len":4096,"permission":[{"id":"modelperm-0efec96c46284560b05b856eb70f68a7","object":"model_permission","created":1736483684,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

step 2. Provide new model list to server

```
$ curl http://localhost:8000/v1/update_models -H "Content-Type: application/json" -d '{"models": [{"id":"mistralai/Mistral-7B-Instruct-v0.3"},{"id":"Qwen/Qwen2.5-7B-Instruct"}]}'

"Existing models: ['mistralai/Mistral-7B-Instruct-v0.3', 'meta-llama/Llama-3.1-8B-Instruct'].
Closed Models: ['meta-llama/Llama-3.1-8B-Instruct'].
Starting new model: [('Qwen/Qwen2.5-7B-Instruct', 'launch time: 12.815344989299774 seconds')]."
```

step 3. verify through model list - Mistral-7B-Instruct-v0.3 and Qwen/Qwen2.5-7B-Instruct

```
$ curl http://localhost:8000/v1/models
{"object":"list","data":[{"id":"mistralai/Mistral-7B-Instruct-v0.3","object":"model","created":1736483860,"owned_by":"vllm","root":"mistralai/Mistral-7B-Instruct-v0.3","parent":null,"max_model_len":4096,"permission":[{"id":"modelperm-5c6a05a4831e486abd37a44129b6af9f","object":"model_permission","created":1736483860,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]},{"id":"Qwen/Qwen2.5-7B-Instruct","object":"model","created":1736483860,"owned_by":"vllm","root":"Qwen/Qwen2.5-7B-Instruct","parent":null,"max_model_len":4096,"permission":[{"id":"modelperm-3b5a6791efe54e83a62ef2f70be321b2","object":"model_permission","created":1736483860,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

step 4. verify through chat completion

```
$ curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0 }'
{"id":"cmpl-2790273630b14f289bdb786fbd83736a","object":"text_completion","created":1736483932,"model":"Qwen/Qwen2.5-7B-Instruct","choices":[{"index":0,"text":" city of neighborhoods, each with its","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":4,"total_tokens":11,"completion_tokens":7,"prompt_tokens_details":null}}
```

```
$ curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "mistralai/Mistral-7B-Instruct-v0.3", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0 }'
{"id":"cmpl-ce9c7799a1934afda74faa89c441453d","object":"text_completion","created":1736483978,"model":"mistralai/Mistral-7B-Instruct-v0.3","choices":[{"index":0,"text":" city that is known for its beautiful","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":5,"total_tokens":12,"completion_tokens":7,"prompt_tokens_details":null}}
```

```
$ curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0 }'
{"object":"error","message":"The model `meta-llama/Llama-3.1-8B-Instruct` does not exist.","type":"NotFoundError","param":null,"code":404}
```
