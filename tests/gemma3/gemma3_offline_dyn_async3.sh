export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=true
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=2
export VLLM_PROMPT_SEQ_BUCKET_MIN=512
export VLLM_PROMPT_SEQ_BUCKET_STEP=512
export VLLM_PROMPT_SEQ_BUCKET_MAX=1536
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_MAX=2
export VLLM_DECODE_BLOCK_BUCKET_MIN=1
export VLLM_DECODE_BLOCK_BUCKET_STEP=16
export VLLM_DECODE_BLOCK_BUCKET_MAX=80 
export VLLM_USE_V1=0


export PT_HPU_LAZY_MODE=1
#export VLLM_FP32_SOFTMAX=1
#export VLLM_PROMPT_USE_FUSEDSDPA=False


export VLLM_SKIP_WARMUP=true


#python gemma3_offline_dyn.py --model google/gemma-3-4b-it --tensor-parallel-size 1 --batchconfig config1.json
python gemma3_offline_dyn_async3.py --model google/gemma-3-4b-it --tensor-parallel-size 1 --batchconfig config1.json