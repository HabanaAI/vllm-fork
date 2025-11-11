#set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_bucket.sh
source "$BASH_DIR"/pd_env.sh

export VLLM_USE_NUMACTL=1

# This is to avoid logic in torch.distributed.hccl.__init__.py _setup_module_id overwriting
# this env var incorrectly
export HLS_MODULE_ID=-1

export VLLM_GPU_MEMORY_UTILIZATION=0.8
export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_GRAPH_PROMPT_RATIO=0

if [ "$INC_FP8" -eq 1 ]; then
  export VLLM_GPU_MEMORY_UTILIZATION=0.9
  export VLLM_GRAPH_RESERVED_MEM=0.6
  export VLLM_GRAPH_PROMPT_RATIO=0
fi

# enable delayed samping on decode
export VLLM_DELAYED_SAMPLING="true"

# params
model_len=16384
max_num_batched_tokens=16384
max_num_seqs=64
input_min=128
input_max=16384
output_max=16384

# ***************************************  bucketing ******************************************* #
unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX VLLM_PROMPT_BS_BUCKET_LIMIT
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX VLLM_PROMPT_SEQ_BUCKET_LIMIT
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX VLLM_DECODE_BS_BUCKET_LIMIT
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX VLLM_DECODE_BLOCK_BUCKET_LIMIT

set_bucketing

export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_PROMPT_SEQ_BUCKET_MIN=1
export VLLM_PROMPT_SEQ_BUCKET_STEP=128
export VLLM_PROMPT_SEQ_BUCKET_MAX=1

#export VLLM_DECODE_BLOCK_BUCKET_MIN=2048
#export VLLM_DECODE_BS_BUCKET_STEP=2
#export VLLM_DECODE_BLOCK_BUCKET_STEP=128

echo " environments are reseted "

env | grep VLLM_PROMPT_BS
env | grep VLLM_PROMPT_SEQ
env | grep VLLM_DECODE_BS
env | grep VLLM_DECODE_BLOCK
# ***************************************  bucketing ends ************************************* #

SWAP_SPACE=64 # GB, memory per rank for preemption.swap.

# decode specific settings
export VLLM_DP_SIZE=2
export VLLM_USE_V1=0
export VLLM_DP_MASTER_IP=10.239.129.81
export VLLM_DP_MASTER_PORT=25940
export VLLM_EP_SIZE=16

# warmup settings
#export VLLM_SKIP_WARMUP=True
if [ "$INC_FP8" -eq 1 ]; then
  export PT_HPU_RECIPE_CACHE_CONFIG=/workspace/pd_fp8_inc_d_cache,false,262144,false
else
  export PT_HPU_RECIPE_CACHE_CONFIG=/workspace/pd_bf16_d_cache,false,131072,false
fi

# MoE settings
export VLLM_SUPPORT_MOE_CHUNK="true"
export PT_HPU_MOE_CHUNK="64, 128"
export PT_HPU_MOE_TOKEN_BOUNDARY="2048, 4096" # to be fine tuned further

# INC FP8 settings
if [ "$INC_FP8" -eq 1 ]; then
  export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp1ep16.json
fi
