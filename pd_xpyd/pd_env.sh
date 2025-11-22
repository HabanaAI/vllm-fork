#! /bin/bash


# ***************************** DO NOT CHANGE SETTINGS *************************************** #

# system settings
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/lib/habanalabs/:/usr/local/lib/
export REQUIRED_VERSION=1.22.0

# HPU settings
export PT_HPU_LAZY_MODE=1

#ray stop --force

# DO NOT change unless you fully understand its purpose
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1

# VLLM settings
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export VLLM_MLA_DISABLE_REQUANTIZATION=0
export VLLM_MOE_N_SLICE=8
export VLLM_EP_SIZE=8
export VLLM_DELAYED_SAMPLING="false"
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_USE_ASYNC_TRANSFER_IN_PD=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

block_size=128

# clear envs
unset VLLM_HPU_LOG_STEP_GRAPH_COMPILATION PT_HPU_METRICS_GC_DETAILS GRAPH_VISUALIZATION


#hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --gaudi2 --merged "hltv,csv"
#hl-prof-config --use-template profile_api_with_nics  --fuser on --trace-analyzer on
hl-prof-config --use-template profile_api_with_nics -b 32768 --trace-analyzer on --fuser on --gaudi3 || true


unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX
unset QUANT_CONFIG VLLM_REQUANT_FP8_INC VLLM_ENABLE_RUNTIME_DEQUANT VLLM_HPU_MARK_SCALES_AS_CONST

# ***************************** DO NOT CHANGE SETTINGS END *************************************** #

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONFIG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
DEBUG_LOG=1
DEBUG_GRAPH=1
DEBUG_PROFILE=0
INC_FP8=1
BENCHMARK_MODE=0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONFIG END ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if [ "$BENCHMARK_MODE" -eq 1 ]; then
    export VLLM_USE_ASYNC_TRANSFER_IN_PD=1
fi

if [ "$DEBUG_LOG" == "1" ]; then
    # Set XPYD_LOG based on WORKSPACE_ROOT or default to /workspace
    XPYD_LOG="${WORKSPACE_ROOT:-/workspace}/xpyd_logs"
    echo "XPYD_LOG set to: $XPYD_LOG"
    # Create the log directory if it doesn't exist
    mkdir -p "$XPYD_LOG"
else
    unset XPYD_LOG
fi

if [ "$DEBUG_GRAPH" == "1" ]; then
    export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=true
    export PT_HPU_METRICS_GC_DETAILS=1
    export GRAPH_VISUALIZATION=1
fi

if [ "$DEBUG_PROFILE" == "1" ]; then
    hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --gaudi2 --merged "hltv,csv"

    export HABANA_PROFILE=1
    export VLLM_PROFILER_ENABLED=full
    export VLLM_TORCH_PROFILER_DIR=/workspace/
    export HABANA_PROFILE_WRITE_HLTV=1
fi

if [ "$INC_FP8" -eq 1 ]; then
  model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
else
  model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2-static/
fi

if [ "$INC_FP8" -eq 1 ]; then
  export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp1ep16.json
  export VLLM_REQUANT_FP8_INC=1
  export VLLM_ENABLE_RUNTIME_DEQUANT=1
  export VLLM_MOE_N_SLICE=1
  export VLLM_HPU_MARK_SCALES_AS_CONST=false
fi


if [ "$INC_FP8" -eq 1 ]; then
  #model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
  if [ "$(hostname)" == "3FG3" ]; then
    model_path=/host/mnt/disk6/hf_models/DeepSeek-R1-G2-inc/
  elif [ "$(hostname)" == "3FG15" ] || [ "$(hostname)" == "3FG16" ]; then
    model_path=/host/mnt/disk2/hf_models/DeepSeek-R1-G2-inc
  elif [ "$(hostname)" == "G15" ] || [ "$(hostname)" == "G16" ] || [ "$(hostname)" == "G13" ]; then
    model_path=/host/mnt/ceph1/hf_models/DeepSeek-R1-G2-static
  elif [ "$(hostname)" == "sys11" ] || [ "$(hostname)" == "sys12" ] \
  || [ "$(hostname)" == "lnc13" ] || [ "$(hostname)" == "lnc-sys21" ] \
  || [ "$(hostname)" == "sys22" ] || [ "$(hostname)" == "sys31" ] \
  || [ "$(hostname)" == "sys32" ] || [ "$(hostname)" == "sys33" ] \
  || [ "$(hostname)" == "lnc-sys21" ]; then
    model_path=/host/mnt/kefei/HF_Models/DeepSeek-R1
  elif [ "$(hostname)" == "skyriver01" ] || [ "$(hostname)" == "skyriver02" ]; then
    model_path=/host/mnt/disk001/hf_models/DeepSeek-R1
  else
    model_path=/host/mnt/disk001/HF_Models/DeepSeek-R1
  fi
else
  echo "hostname: $(hostname)"
  if [ "$(hostname)" == "G15" ] || [ "$(hostname)" == "G16" ] || [ "$(hostname)" == "G13" ]; then
    model_path=/host/mnt/ceph1/hf_models/DeepSeek-R1-G2-dynamic
  elif [ "$(hostname)" == "sys11" ] || [ "$(hostname)" == "sys12" ] \
  || [ "$(hostname)" == "lnc13" ] || [ "$(hostname)" == "lnc-sys21" ] \
  || [ "$(hostname)" == "sys22" ] || [ "$(hostname)" == "sys31" ] \
  || [ "$(hostname)" == "sys32" ] || [ "$(hostname)" == "sys33" ] \
  || [ "$(hostname)" == "lnc-sys21" ]; then
    model_path=/host/mnt/kefei/HF_Models/DeepSeek-R1-Gaudi3/
  elif [ "$(hostname)" == "skyriver01" ] || [ "$(hostname)" == "skyriver02" ]; then
    model_path=/host/mnt/disk001/hf_models/DeepSeek-R1-Gaudi3/
  else
    model_path=/host/mnt/disk002/HF_Models/DeepSeek-R1-Gaudi3/
  fi
fi

unset QUANT_CONFIG VLLM_REQUANT_FP8_INC VLLM_ENABLE_RUNTIME_DEQUANT VLLM_HPU_MARK_SCALES_AS_CONST

if [ "$INC_FP8" -eq 1 ]; then
  export QUANT_CONFIG="$BASH_DIR"/inc_fp8_tp1ep16.json
  export VLLM_REQUANT_FP8_INC=1
  export VLLM_ENABLE_RUNTIME_DEQUANT=1
  export VLLM_MOE_N_SLICE=1
  export VLLM_HPU_MARK_SCALES_AS_CONST=false
fi

