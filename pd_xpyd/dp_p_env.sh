#set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_bucket.sh
source "$BASH_DIR"/pd_env.sh


export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_PROMPT_RATIO=1
# params
model_len=16384
max_num_batched_tokens=16384
max_num_seqs=8
input_min=128
input_max=16384
output_max=16384

# ***************************************  bucketing ******************************************* #
unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

set_bucketing

export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=1
export VLLM_DECODE_BLOCK_BUCKET_MIN=2
export VLLM_DECODE_BLOCK_BUCKET_STEP=1
export VLLM_DECODE_BLOCK_BUCKET_MAX=2

echo " environments are reseted "

env | grep VLLM_PROMPT_BS
env | grep VLLM_PROMPT_SEQ
env | grep VLLM_DECODE_BS
env | grep VLLM_DECODE_BLOCK
# ***************************************  bucketing ends ************************************* #

# prefill specific setting
export VLLM_SKIP_PREFILL_SAMPLING=1
export VLLM_DP_SIZE=1
export VLLM_USE_V1=0
export VLLM_EP_SIZE=8

# warmup settings
export VLLM_SKIP_WARMUP=True
#export PT_HPU_RECIPE_CACHE_CONFIG=/workspace/pd_p_cache,false,131072

# MoE settings
export VLLM_SUPPORT_MOE_CHUNK="false"  # Can be true after following para are tuned.
#export PT_HPU_MOE_CHUNK="64, 128"
#export PT_HPU_MOE_TOKEN_BOUNDARY="2048, 4096"

# INC FP8 settings
if [ "$INC_FP8" -eq 1 ]; then
  export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp8ep8.json
fi
