export VLLM_SKIP_WARMUP=true
export LLM_MODEL_ID=google/gemma-3-27b-it
export HF_TOKEN="xxx"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false
export DATA_PATH=~/data
export MAX_TOTAL_TOKENS=500
export VLLM_USE_V1=0
export PT_HPU_LAZY_MODE=1
export VLLM_FP32_SOFTMAX=1
#python vllm-gemma3-offline.py


python gemma3.py 0
