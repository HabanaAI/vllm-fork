#set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_bucket.sh
source "$BASH_DIR"/pd_env.sh

echo "PLATFORM_TYPE=${PLATFORM_TYPE}"
if [ "${PLATFORM_TYPE}" = "SEDV" ]; then
  echo "SEDV platform type detected"
  export HCL_HLS3RACK_NUM_DEVICES=4
  export HCL_HLS3RACK_SCALEUP_GROUP_SIZE=8
  export HLS3_RACK_SCALEOUT_PORT_MASK=0
  export GLOO_SOCKET_IFNAME=ens11f1np1
  export ENABLE_EXPERIMENTAL_FLAGS=true
  export CONGESTION_CONTROL_ENABLE=1
  #export EXP_FLAGS=1
  #export HCL_IMB_SIZE=1M
  #export HCL_SLICE_SIZE=1M
  #export HCL_GDR_SLICE_SIZE=1M
  #export CONGESTION_WINDOW=8 #32 or 16 or 32
elif [ "${PLATFORM_TYPE}" = "SKYRIVERV3" ]; then
  echo "SKYRIVERV3 platform type detected"
  export HCL_HLS3RACK_NUM_DEVICES=8
  export HCL_HLS3RACK_SCALEUP_GROUP_SIZE=8
  export HLS3_RACK_SCALEOUT_PORT_MASK=0
  #export GLOO_SOCKET_IFNAME=ens11f1np1
  export ENABLE_EXPERIMENTAL_FLAGS=true
  export CONGESTION_CONTROL_ENABLE=1
  export HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  #export EXP_FLAGS=1
  #export HCL_IMB_SIZE=1M
fi


export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_PROMPT_RATIO=1
export VLLM_SKIP_PREFILL_SAMPLING=1
export VLLM_FETCH_KV_USE_ASYNC_D2H=1
export SHARED_EXPERT_DISPOSITION=0
# params
model_len=32768
max_num_batched_tokens=32768 #4*4096
max_num_seqs=8
input_min=3500
input_max=3500
output_max=1000

# ***************************************  bucketing ******************************************* #
unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

set_bucketing

export VLLM_PROMPT_SEQ_BUCKET_STEP=512

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

## warmup settings
export VLLM_SKIP_WARMUP=True

export VLLM_DP_SIZE=1
export VLLM_USE_V1=0
export VLLM_EP_SIZE=${PREFILL_NUM_CARDS:-8}
export VLLM_TTFT_TRACE=false
export VLLM_TTFT_TRACE_STACK=false

export PT_HPU_RECIPE_CACHE_CONFIG=/host/mnt/disk002/kf/recipe_cache/ww33_inc_fp8_p,false,16384,false

# MoE settings
export VLLM_SUPPORT_MOE_CHUNK="false"  # Can be true after following para are tuned.

# INC FP8 settings
if [ "$INC_FP8" -eq 1 ]; then
  if [ -z "${QUANT_CONFIG_PREFILL:-}" ]; then
    echo "[ERROR] QUANT_CONFIG_PREFILL is not set. Please specify the quant config filename in your env." >&2
    exit 1
  fi
  export QUANT_CONFIG="$BASH_DIR"/${QUANT_CONFIG_PREFILL}
  if [ ! -f "${QUANT_CONFIG}" ]; then
    echo "[ERROR] Quant config file '${QUANT_CONFIG}' not found." >&2
    exit 1
  fi

fi
