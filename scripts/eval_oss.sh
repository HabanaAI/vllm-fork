model_path=/mnt/disk5/lmsys/gpt-oss-20b-bf16
model_path=/software/users/yiliu7/HF_HOME/lmsys/gpt-oss-20b-bf16
model_path=/software/users/yiliu7/HF_HOME/lmsys/gpt-oss-120b-bf16
basename=$(basename $model_path)
taskname=gsm8k
# taskname=gsm8k
#replace the , in taskname with '--'
taskname_str=${taskname//,/--}
model_basename=$(basename $model_path)
output_log_file_name=lm_eval_output_${model_basename}_${taskname_str}


task_name="gsm8k"
# task_name="gsm8k_oss"
limit=1500
batch_size=32

# export QUANT_CONFIG=./quant_configs/inc_unit_scale.json
# export QUANT_CONFIG=./quant_configs/inc_quant.json


export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
# export PT_HPU_WEIGHT_SHARING="0"
export VLLM_DELAYED_SAMPLING=true

basename=$(basename $model_path)
is_120b=false
if [[ $basename == *"120b"* ]]; then
    is_120b=true
fi

# is 120b
if [ "$is_120b" = true ]; then
    echo "Using model 120B, setting tp_size=4"
    tp_size=4
    ep_size=1
    # export QUANT_CONFIG=./quant_configs/inc_quant_120b.json
    # export QUANT_CONFIG=./quant_configs/inc_measure_120b.json

else
    echo "Using model 20B, setting tp_size=1"
    tp_size=1
    ep_size=1
fi

# VLLM_ENABLE_FUSED_MOE_WITH_BIAS=1 \
# VLLM_DISABLE_MARK_SCALES_AS_CONST=1 \
# VLLM_LOGGING_LEVEL=DEBUG \
# PT_HPU_LAZY_MODE=1 \
# VLLM_PROMPT_USE_FUSEDSDPA=0 \

PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128 \
PT_HPU_ENABLE_FUSED_SDPA_SINK=1 \
PT_HPU_SDPA_QKV_SLICE_MODE_FWD=1 \
PT_HPU_LAZY_MODE=1 \
VLLM_DISABLE_MARK_SCALES_AS_CONST=1 \
INC_PT_ONLY=1 \
VLLM_BUILD=1.23.0.248 \
VLLM_SKIP_WARMUP=true   \
lm-eval --model vllm \
    --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=8192,max_num_seqs=32,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=4096,disable_log_stats=True"\
    --tasks ${task_name} \
    --batch_size $batch_size \
    --limit $limit
