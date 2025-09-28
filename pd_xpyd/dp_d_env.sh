#set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_bucket.sh
source "$BASH_DIR"/pd_env.sh

export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_GRAPH_RESERVED_MEM=0.3

# Enable packed allgather optimization
export ENABLE_PACKED_ALLGATHER=1
export ENABLE_PRE_SHARED_EXPERT=1
export VLLM_USE_NUMACTL=1
export VLLM_SPLIT_CPU_BIND=0
export VLLM_DEBUG_TOPO=1

# This is to avoid logic in torch.distributed.hccl.__init__.py _setup_module_id overwriting
# this env var incorrectly
export HLS_MODULE_ID=-1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export VLLM_GRAPH_PROMPT_RATIO=0

# enable delayed samping on decode
export VLLM_DELAYED_SAMPLING="true"

# params
#<<<<<<< HEAD
#model_len=16384
#max_num_batched_tokens=16384
#max_num_seqs=64
#input_min=128
#input_max=16384
#output_max=16384
#=======
#model_len=16384
#max_num_batched_tokens=16384
#max_num_seqs=512
#input_min=2048
#input_max=3584
#output_max=1024

model_len=40960
max_num_batched_tokens=40960
max_num_seqs=16
input_min=3000
input_max=4000
output_max=1000

# ***************************************  bucketing ******************************************* #
unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

set_bucketing



export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BLOCK_BUCKET_STEP=2

export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_PROMPT_SEQ_BUCKET_MIN=1
export VLLM_PROMPT_SEQ_BUCKET_STEP=128
export VLLM_PROMPT_SEQ_BUCKET_MAX=1

#<<<<<<< HEAD
##export VLLM_DECODE_BLOCK_BUCKET_MIN=2048
##export VLLM_DECODE_BS_BUCKET_STEP=2
##export VLLM_DECODE_BLOCK_BUCKET_STEP=2
#=======
echo VLLM_PROMPT_BS_BUCKET_MIN:$VLLM_PROMPT_BS_BUCKET_MIN, VLLM_PROMPT_BS_BUCKET_STEP:$VLLM_PROMPT_BS_BUCKET_STEP, VLLM_PROMPT_BS_BUCKET_MAX:$VLLM_PROMPT_BS_BUCKET_MAX
echo VLLM_PROMPT_SEQ_BUCKET_MIN:$VLLM_PROMPT_SEQ_BUCKET_MIN, VLLM_PROMPT_SEQ_BUCKET_STEP:$VLLM_PROMPT_SEQ_BUCKET_STEP, VLLM_PROMPT_SEQ_BUCKET_MAX:$VLLM_PROMPT_SEQ_BUCKET_MAX
echo VLLM_DECODE_BS_BUCKET_MIN:$VLLM_DECODE_BS_BUCKET_MIN, VLLM_DECODE_BS_BUCKET_STEP:$VLLM_DECODE_BS_BUCKET_STEP, VLLM_DECODE_BS_BUCKET_MAX:$VLLM_DECODE_BS_BUCKET_MAX
echo VLLM_DECODE_BLOCK_BUCKET_MIN:$VLLM_DECODE_BLOCK_BUCKET_MIN, VLLM_DECODE_BLOCK_BUCKET_STEP:$VLLM_DECODE_BLOCK_BUCKET_STEP, VLLM_DECODE_BLOCK_BUCKET_MAX:$VLLM_DECODE_BLOCK_BUCKET_MAX

#>>>>>>> kf-fork/deepseek_r1_ww33_kf

echo " environments are reseted "

env | grep VLLM_PROMPT_BS
env | grep VLLM_PROMPT_SEQ
env | grep VLLM_DECODE_BS
env | grep VLLM_DECODE_BLOCK
# ***************************************  bucketing ends ************************************* #

#<<<<<<< HEAD
SWAP_SPACE=64 # GB, memory per rank for preemption.swap.
#=======
export VLLM_SKIP_WARMUP=True
#unset VLLM_SKIP_WARMUP
#export PT_HPU_RECIPE_CACHE_CONFIG=/workspace/ww33_inc_fp8_d,false,16384
export PT_HPU_RECIPE_CACHE_CONFIG=/host/mnt/disk002/kf/recipe_cache/ww33_inc_fp8_d,false,16384,false
#>>>>>>> kf-fork/deepseek_r1_ww33_kf

# decode specific settings
export VLLM_DP_SIZE=2
export VLLM_USE_V1=0
export VLLM_DP_MASTER_IP=10.239.129.81
export VLLM_DP_MASTER_PORT=25940
export VLLM_EP_SIZE=16

#<<<<<<< HEAD
## warmup settings
#export VLLM_SKIP_WARMUP=True
##export PT_HPU_RECIPE_CACHE_CONFIG=/workspace/pd_d_cache,false,131072
#
## MoE settings
#export VLLM_SUPPORT_MOE_CHUNK="true"
#export PT_HPU_MOE_CHUNK="64, 128"
#export PT_HPU_MOE_TOKEN_BOUNDARY="2048, 4096" # to be fine tuned further
#
## INC FP8 settings
#if [ "$INC_FP8" -eq 1 ]; then
#  export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp1ep16.json
#fi
#=======
export PT_HPU_MOE_THRESHOLD=64

if [ "$INC_FP8" -eq 1 ]; then
  export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp1ep16.json
fi

#>>>>>>> kf-fork/deepseek_r1_ww33_kf
