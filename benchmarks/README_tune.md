# Bucket Tuning with Benchmark Scripts on Gaudi

This README guides you through tuning bucket related configurations on Gaudi with online benchmark script and
offline benchmark script with Optuna.

## Set Up

Install Optuna and optionally install Optuna dashboard:

```bash
pip install optuna
```
```bash
pip install optuna-dashboard
```

## Example - Online Benchmark

Example command to tune linear bucket configurations for Llama3.3-70B FP8 precison on 4x:

```bash
QUANT_CONFIG=/root/mjli/quant/llama3.3-70b/maxabs_quant_g3.json RUNTIME_SCALE_PATCHING=1  PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
DISABLE_PROMPT_LOGPROBS=1 VLLM_PROMPT_USE_FUSEDSDPA=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_EXPONENTIAL_BUCKETING=false  PT_HPU_LAZY_MODE=1 PT_HPU_WEIGHT_SHARING=0 VLLM_PROMPT_SEQ_BUCKET_MAX=132096 \
VLLM_PROMPT_SEQ_BUCKET_STEP=2048 python tune_benchmark_online.py --prompt-bs-bucket-min-range 1 2 1 \
--prompt-bs-bucket-step-range 1 2 1 --prompt-bs-bucket-max-range 1 4 1 --prompt-seq-bucket-min-range 49152 98304 2048  \
--decode-bs-bucket-min-range 1 2 1 --decode-bs-bucket-max-range 4 10 2 --decode-bs-bucket-step-range 2 8 2 \
--decode-block-bucket-step-range 1024 2048 1024 --decode-block-bucket-min-range 1024 2048 1024 \
--vllm-server-cmd "python3 -m vllm.entrypoints.openai.api_server --port 9990 --model meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4 --max-num-seqs 1 --max-model-len 132096 --quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu" \
--benchmark-serving-cmd "python3 benchmark_serving.py --backend vllm --model meta-llama/Llama-3.3-70B-Instruct --dataset-name random --request-rate inf --num-prompts 5 --max-concurrency 21 --port 9990 --random-input-len 122880 --random-output-len 8192 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90 --ignore-eos" \
--task-name tuning_llama3.3_70b --time-out 1500
```

You can check the tuning results with optuna dashboard:

```bash
optuna-dashboard sqlite:////tmp/tuning_llama3.3_70b.db
```


## Example - Offline Benchmark

