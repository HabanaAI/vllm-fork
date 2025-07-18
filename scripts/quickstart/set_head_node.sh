#! /bin/bash

# set -x
# parameters to be changed
# set IP address of header node
export VLLM_HOST_IP=10.239.128.244
# set NIC interface name of worker IP address
export GLOO_SOCKET_IFNAME=enx6c1ff7012f87

# warmup cache folder
export PT_HPU_RECIPE_CACHE_CONFIG=/data/cache/cache_32k_1k_20k_16k,false,32768

# vllm parameters
max_num_batched_tokens=32768
max_num_seqs=512
input_min=768
input_max=20480
output_max=16896

export HCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export VLLM_DELAYED_SAMPLING="true"
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_MLA_DISABLE_REQUANTIZATION=0


BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

ray stop --force




timestamp=$(date +%Y%m%d_%H%M%S)
torch_prof_out_dir="dyanmic_torch_profiler_${timestamp}"

echo "torch_prof_out_dir: $torch_prof_out_dir"


export HABANA_PROFILE_WRITE_HLTV=1 
export HABANA_PROFILE=1
# hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --trace-analyzer-xlsx on -invoc csv,hltv -merged csv,hltv
hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --trace-analyzer-xlsx on
hl-prof-config --gaudi2
hl_prof_out_dir="dynamic_prof_hlv_${VLLM_PT_PROFILE}_${timestamp}"
hl-prof-config -o $hl_prof_out_dir

export VLLM_TORCH_PROFILER_DIR=$torch_prof_out_dir
export VLLM_ENGINE_PROFILER_ENABLED=1
export VLLM_ENGINE_PROFILER_WARMUP_STEPS=1033
export VLLM_ENGINE_PROFILER_STEPS=4
export VLLM_ENGINE_PROFILER_REPEAT=1


export VLLM_PROFILE_FILE="server_events_${torch_prof_out_dir}.json"
export VLLM_PROFILER_ENABLED=true


# DO NOT change unless you fully undersand its purpose
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export PT_HPU_LAZY_MODE=1

export VLLM_MOE_N_SLICE=8
export VLLM_EP_SIZE=16

block_size=128
# DO NOT change ends...

# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0

#export VLLM_SKIP_WARMUP=true



unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

set_bucketing

echo " environments are reseted "

env | grep VLLM