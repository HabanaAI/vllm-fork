export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=2
export VLLM_PROMPT_SEQ_BUCKET_MIN=384
export VLLM_PROMPT_SEQ_BUCKET_MAX=384
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_MAX=1
export VLLM_DECODE_BLOCK_BUCKET_MIN=512
export VLLM_DECODE_BLOCK_BUCKET_MAX=512 
export VLLM_USE_V1=0
export VLLM_EXPONENTIAL_BUCKETING=False

#export VLLM_SKIP_WARMUP=true
export PT_HPU_LAZY_MODE=1
#export VLLM_FP32_SOFTMAX=1
#export VLLM_PROMPT_USE_FUSEDSDPA=False

python gemma3_offline.py --model /root/software/data/pytorch/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/ --tensor-parallel-size 1 --num-images 2 --batch-size 1
