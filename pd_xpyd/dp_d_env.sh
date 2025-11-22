#set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_bucket.sh
source "$BASH_DIR"/pd_env.sh


if [ "${PLATFORM_TYPE}" = "SEDV" ]; then
  echo "SEDV platform type detected"
  export HCL_HLS3RACK_NUM_DEVICES=4
  export HCL_HLS3RACK_SCALEUP_GROUP_SIZE=16
  export HLS3_RACK_SCALEOUT_PORT_MASK=0
  export GLOO_SOCKET_IFNAME=ens11f1np1
  export ENABLE_EXPERIMENTAL_FLAGS=true
  export CONGESTION_CONTROL_ENABLE=1
  #export EXP_FLAGS=1
  #export CONGESTION_WINDOW=8 #32 or 16 or 32
elif [ "${PLATFORM_TYPE}" = "SKYRIVERV3" ]; then
  echo "SKYRIVERV3 platform type detected"
  export HCL_HLS3RACK_NUM_DEVICES=16
  export HCL_HLS3RACK_SCALEUP_GROUP_SIZE=16
  export HLS3_RACK_SCALEOUT_PORT_MASK=0
  #export GLOO_SOCKET_IFNAME=ens11f1np1
  export ENABLE_EXPERIMENTAL_FLAGS=true
  export CONGESTION_CONTROL_ENABLE=1
  export HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
fi


export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_GRAPH_RESERVED_MEM=0.3

# Enable packed allgather optimization
export ENABLE_PACKED_ALLGATHER=1
export SHARED_EXPERT_DISPOSITION=0
export VLLM_USE_NUMACTL=0
export VLLM_SPLIT_CPU_BIND=0
export VLLM_DEBUG_TOPO=1
export VLLM_USE_ASYNC_RECV_KV_CACHES_OPT=0

# This is to avoid logic in torch.distributed.hccl.__init__.py _setup_module_id overwriting
# this env var incorrectly
export HLS_MODULE_ID=-1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export VLLM_GRAPH_PROMPT_RATIO=0

# enable delayed samping on decode
export VLLM_DELAYED_SAMPLING="true"

model_len=40960
max_num_batched_tokens=40960
max_num_seqs=16
input_min=3500
input_max=3500
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

echo VLLM_PROMPT_BS_BUCKET_MIN:$VLLM_PROMPT_BS_BUCKET_MIN, VLLM_PROMPT_BS_BUCKET_STEP:$VLLM_PROMPT_BS_BUCKET_STEP, VLLM_PROMPT_BS_BUCKET_MAX:$VLLM_PROMPT_BS_BUCKET_MAX
echo VLLM_PROMPT_SEQ_BUCKET_MIN:$VLLM_PROMPT_SEQ_BUCKET_MIN, VLLM_PROMPT_SEQ_BUCKET_STEP:$VLLM_PROMPT_SEQ_BUCKET_STEP, VLLM_PROMPT_SEQ_BUCKET_MAX:$VLLM_PROMPT_SEQ_BUCKET_MAX
echo VLLM_DECODE_BS_BUCKET_MIN:$VLLM_DECODE_BS_BUCKET_MIN, VLLM_DECODE_BS_BUCKET_STEP:$VLLM_DECODE_BS_BUCKET_STEP, VLLM_DECODE_BS_BUCKET_MAX:$VLLM_DECODE_BS_BUCKET_MAX
echo VLLM_DECODE_BLOCK_BUCKET_MIN:$VLLM_DECODE_BLOCK_BUCKET_MIN, VLLM_DECODE_BLOCK_BUCKET_STEP:$VLLM_DECODE_BLOCK_BUCKET_STEP, VLLM_DECODE_BLOCK_BUCKET_MAX:$VLLM_DECODE_BLOCK_BUCKET_MAX


echo " environments are reseted "

env | grep VLLM_PROMPT_BS
env | grep VLLM_PROMPT_SEQ
env | grep VLLM_DECODE_BS
env | grep VLLM_DECODE_BLOCK
# ***************************************  bucketing ends ************************************* #

SWAP_SPACE=64 # GB, memory per rank for preemption.swap.
export PT_HPU_RECIPE_CACHE_CONFIG=/host/mnt/disk002/kf/recipe_cache/ww33_inc_fp8_d,false,16384,false

# decode specific settings
export VLLM_DP_SIZE=2
export VLLM_USE_V1=0


export VLLM_DP_MASTER_PORT=25940
export VLLM_EP_SIZE=16

## warmup settings
export VLLM_SKIP_WARMUP=True

export VLLM_SUPPORT_MOE_CHUNK="true"
export PT_HPU_MOE_CHUNK="64, 128"
export PT_HPU_MOE_TOKEN_BOUNDARY="2048, 4096" # to be fine tuned further
#export PT_HPU_MOE_THRESHOLD=64

if [ "$INC_FP8" -eq 1 ]; then
  #export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp1ep16.json
  if [ -z "${QUANT_CONFIG_DECODE:-}" ]; then
    echo "[ERROR] QUANT_CONFIG_DECODE is not set. Please specify the quant config filename in your env." >&2
    exit 1
  fi
  export QUANT_CONFIG="$BASH_DIR"/${QUANT_CONFIG_DECODE}
  if [ ! -f "$QUANT_CONFIG" ]; then
    echo "[ERROR] Quant config file '$QUANT_CONFIG' not found." >&2
    exit 1
  fi

fi

