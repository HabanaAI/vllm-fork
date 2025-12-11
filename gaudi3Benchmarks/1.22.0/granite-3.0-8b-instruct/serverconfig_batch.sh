#granite-3.0-8b-instruct batch mode(offline) with batch size 192 

VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_LAZY_MODE=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_FUSED_BLOCK_SOFTMAX=true \
VLLM_DELAYED_SAMPLING=false \
python3 -m vllm.entrypoints.openai.api_server \
    --model=ibm-granite/granite-3.0-8b-instruct \
    --port 8080 \
    --max-num-seqs=192 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.95 \
    --tensor-parallel-size=1 \
    --max-model-len=131072 \
    --block-size=128 \
    --disable-log-requests \
    --use-v2-block-manager \
    --use-padding-aware-scheduling \
    --disable-log-stats \
    --trust-remote-code 2>&1 | tee granite-3.0-8b-instruct_server.txt

