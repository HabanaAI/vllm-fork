# Static Quantization 
The below steps are given for an model Llama-4-Maverick-17B-128E-Instruct as example 

## Configuration

1. Locate the file `maxabs_quant_g3.json` inside the `model quantization` folder.
2. Edit it and set the parameter `dump_stats_path` to the absolute path where the repository is cloned.

Example:
```json
"dump_stats_path": "/root/vllm-fork/.static_quant/1.22.0/Llama-4-Maverick-17B-128E-Instruct/g3/inc_output"
```

## Environment Variable
Export the environment variable QUANT_CONFIG before running the server. It must point to the location of maxabs_quant_g3.json.

Example:
```bash 
export QUANT_CONFIG='/root/vllm-fork/.static_quant/1.22.0/Llama-4-Maverick-17B-128E-Instruct/maxabs_quant_g3.json'
```

## Run vLLM Server

Start the vLLM server with quantization enabled:

```bash 
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --quantization inc \
  --kv-cache-dtype fp8_inc \
  --weights-load-device cpu \
  --tensor-parallel-size 8 \
  --max-model-len 2048
```

## Notes

1. The dump_stats_path in maxabs_quant_g3.json must be an absolute path.
2. QUANT_CONFIG must be exported before running vllm serve.
3. Adjust --tensor-parallel-size and --max-model-len according to your system resources.
