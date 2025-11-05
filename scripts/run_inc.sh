############# debug
# ray stop --force
# pkill -9 python

# OFFICIAL_FP8_MODEL=/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207/
OFFICIAL_FP8_MODEL="/mnt/disk2/MiniMaxAI/MiniMax-M2"
# OFFICIAL_FP8_MODEL="/mnt/disk2/MiniMaxAI/MiniMax-M2-G2/"
export VLLM_MOE_N_SLICE=1
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_ENABLE_RUNTIME_DEQUANT=1
export VLLM_HPU_MARK_SCALES_AS_CONST=false
export PT_HPU_LAZY_MODE=1 

export VLLM_REQUANT_FP8_INC=1
export INC_FORCE_NAIVE_SCALING=1

# export QUANT_CONFIG="./scripts/quant_configs/inc_quant_with_fp8kv_config.json" 
export VLLM_HPU_CONVERT_TO_FP8UZ=1
QUANT_CONFIG="/mnt/disk3/yiliu4/vllm-fork/scripts/quant_configs/inc_quant_unit.json" \
python ./scripts/run_example_tp_local.py \
    --model ${OFFICIAL_FP8_MODEL} \
    --tokenizer ${OFFICIAL_FP8_MODEL} \
    --osl 16 \
    --max_model_len 1024 \
    --max_num_seqs 1  \
    --enforce_eager 2>&1 | tee ./DeepSeek-R1-G2-INC-424-Converter207.inc_quant_unit.log
# export VLLM_HPU_FORCE_CHANNEL_FP8=0
# export VLLM_LOGGING_LEVEL=DEBUG
# QUANT_CONFIG="./scripts/quant_configs_local/inc_measure_with_fp8kv_config.json" \
# python ./scripts/run_example_tp_local.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 16 \reppredd
#     --max_num_seqs 1



# INC_MEASURMENT_DUMP_FOLDER="/mnt/disk3/yiliu4/vllm-fork" \
# QUANT_CONFIG="./scripts/quant_configs/inc_quant_per_channel_bf16kv.json" \
# python ./scripts/run_example_tp.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 16 \
#     --max_num_seqs 1

# INC_MEASURMENT_DUMP_FOLDER="/mnt/disk3/yiliu4/vllm-fork" \
# QUANT_CONFIG="./quant_configs/inc_quant_per_channel_bf16kv.json" \
# python run_example_tp.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 16 \
#     --max_num_seqs 1

#############
#####
# Warmup finished in 180 secs
################



# echo "==================== Starting BF16 ===================="
# export OFFICIAL_FP8_MODEL="/mnt/disk2/hf_models/Llama-2-7b-chat-hf"
# export OFFICIAL_FP8_MODEL="meta-llama/Llama-3.1-8B-Instruct"
# export PT_HPU_LAZY_MODE=1 
# # VLLM_SKIP_WARMUP=1 \
# python run_example_llama.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --tp_size 1 \
#     --ep_size 1  | tee ./scalar_warmup/llama2-7b.bf16

# sleep 5


# echo "==================== Starting INC prepare ===================="
# VLLM_SKIP_WARMUP=true \
# QUANT_CONFIG=inc_measure.json \
# python run_example_llama.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --tp_size 1 \
#     --ep_size 1 \
#     --inc 2>&1 | tee ./scalar_warmup/llama3.1.prepare

# sleep 5

# echo "==================== Starting INC with fp8 kv cache ===================="
# PT_HPU_LAZY_MODE=1 \
# QUANT_CONFIG=inc_quant.json \
# python run_example_llama.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --tp_size 1 \
#     --ep_size 1 \
#     --fp8_kv_cache \
#     --inc 2>&1 | tee ./scalar_warmup/llama3.1.convert_2
    
    
# echo "==================== Starting INC with fp8 kv cache RUNTIME_SCALE_PATCHING=1 ===================="


# timestamp=""

# # ENABLE_EXPERIMENTAL_FLAGS=1 PRINT_FILE_AND_LINE=1  LOG_LEVEL_PASS_MANAGER=1  \
# # LOG_LEVEL_ALL=1 HABANA_LOGS=.habana_logs-scalar-warmup  GRAPH_VISUALIZATION=1 \
# PT_HPU_LAZY_MODE=1 \
# RUNTIME_SCALE_PATCHING=1 \
# PT_HPU_ENABLE_H2D_SCALES=1 \
# QUANT_CONFIG=inc_quant.json \
# python run_example_llama.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --tp_size 1 \
#     --ep_size 1 \
#     --fp8_kv_cache \
#     --inc 2>&1 | tee ./scalar_warmup/llama3.1.convert.runtime_scale_patching

# ENABLE_CONSOLE=true LOG_LEVEL_ALL=4 \

# ENABLE_EXPERIMENTAL_FLAGS=1 PRINT_FILE_AND_LINE=1  LOG_LEVEL_PASS_MANAGER=1  \
# LOG_LEVEL_ALL=1 HABANA_LOGS=.habana_logs-scalar-not-skip_b2b  GRAPH_VISUALIZATION=1 \
# PT_HPU_LAZY_MODE=1 \
# RUNTIME_SCALE_PATCHING=1 \
# PT_HPU_ENABLE_H2D_SCALES=1 \
# QUANT_CONFIG=inc_quant.json \
# python run_example_llama.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --tp_size 1 \
#     --ep_size 1 \
#     --fp8_kv_cache \
#     --inc 2>&1 | tee ./scalar_warmup/llama2-7b.convert.runtime_scale_patching_all
    
    
    # skip_block2batch_matmul_only : failed
    # skip_batch2block_matmul_only : failed
    # skip_skip_two_b2b: good




# VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
# GRAPH_VISUALIZATION=1 \
# VLLM_REQUANT_FP8_INC=1 \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# VLLM_MOE_N_SLICE=1 \
# QUANT_CONFIG=./scripts/inc_quant_per_channel_bf16kv.json \
# python ./scripts/run_example_tp.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 16 \
#     --max_num_seqs 1

# ray stop --force
# pkill -9 python
# export OFFICIAL_FP8_MODEL="/software/users/yiliu4/HF_HOME/hub/models--deepseek-ai--DeepSeek-R1/snapshots/a157fa3d494497a54586a333a23df6c2143e7697"
# export OFFICIAL_FP8_MODEL="/software/users/yiliu4/HF_HOME/hub/Meta-Llama-3-8B-Instruct"


# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# VLLM_PROMPT_BS_BUCKET_MIN=1 \
# VLLM_PROMPT_BS_BUCKET_MAX=1 \
# VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
# VLLM_PROMPT_SEQ_BUCKET_STEP=512 \
# VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
# VLLM_DECODE_BS_BUCKET_MIN=1 \
# VLLM_DECODE_BS_BUCKET_MAX=1 \
# VLLM_REQUANT_FP8_INC=1 \
# QUANT_CONFIG=inc_measure_with_fp8kv_config.json \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
#     python run_example_tp.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_num_seqs 1 \
#     --nprompts 512 \
#     --enforce_eager \
#     --dataset pile 2>&1 | tee ./woq_logs/prepare.g2.full.disable_graph

# INC_DYNAMIC_MOE_EXPERTS=32 \
# PT_HPU_LAZY_MODE=1 \
# VLLM_REQUANT_FP8_INC=1 \
# QUANT_CONFIG=inc_measure_with_fp8kv_config.json \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# python run_example_tp.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_num_seqs 1
    


# export OFFICIAL_FP8_MODEL="/mnt/weka/llm/DeepSeek-V3/"

# export PT_HPU_LAZY_MODE=1 
# # HABANA_LOGS=.habana_logs \
# # LOG_LEVEL_ALL=1 \
# export GRAPH_VISUALIZATION=1 
# INC_DYNAMIC_MOE_EXPERTS=32 \
# VLLM_REQUANT_FP8_INC=1 \

# VLLM_ENABLE_RUNTIME_DEQUANT=1 \

# export HABANA_LOGS=.habana_logs_425_1
# export LOG_LEVEL_ALL=1 

# export OFFICIAL_FP8_MODEL="/mnt/weka/data/pytorch/DeepSeek-R1"
# export OFFICIAL_FP8_MODEL="/software/users/yiliu4/HF_HOME/hub/Qwen/Qwen1.5-MoE-A2.7B-Chat"
# export OFFICIAL_FP8_MODEL="/dev/shm/Qwen1.5-MoE-A2.7B/"
# export OFFICIAL_FP8_MODEL="/mnt/disk5/qwen3/Qwen3-30B-A3B-250425"
# export OFFICIAL_FP8_MODEL="/mnt/disk5/Qwen3-30B-A3B-250425"

# export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_DISABLE_MARK_SCALES_AS_CONST=1
# python ./scripts/run_example_tp_local.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_num_seqs 1  \
#     --tp_size 8 \
#     --ep_size 8

# QUANT_CONFIG=./scripts/inc_measure_with_v2.json \

# --tp_size 1 \
# --ep_size 1 \
#############################
# Qwen
#############################
# remove it ?
# export VLLM_DYNAMIC_MOE_MIN_TOKENS=0

# python ./scripts/run_example_tp_local.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1 
      
# QUANT_CONFIG=./scripts/inc_measure_with_v2.json \
# python ./scripts/run_example_tp_local.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1  \
#     --max_model_len 2048 \
#     --inc \
#     --dataset pile \
#     --nprompts 

# QUANT_CONFIG=./scripts/inc_quant_v2.json \
# python ./scripts/run_example_tp_local.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_model_len 2048 \
#     --max_num_seqs 1  \
#     --inc \
#     --fp8_kv_cache

# QUANT_CONFIG=./scripts/inc_quant_v2.json \
# python ./scripts/run_lm_eval_local.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --task gsm8k \
#     --batch_size 16 \
#     --limit 128 \
#     --inc \
#     --fp8_kv_cache

# python ./scripts/run_lm_eval_local.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --task gsm8k \
#     --batch_size 16 \
#     --limit 128

# DEBUG 04-28 07:23:31 [llm_engine.py:1509] Stopping remote worker execution loop.
# Running generate_until requests: 100%|████████████| 128/128 [00:57<00:00,  2.23it/s]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8750|±  |0.0293|
# |     |       |strict-match    |     5|exact_match|↑  |0.9219|±  |0.0238|

# INC pile-128
# WARNING 04-28 07:43:17 [hpu_model_runner.py:1013] Configuration: ('prompt', 16, 896) was not warmed-up!
# DEBUG 04-28 07:45:18 [llm_engine.py:1509] Stopping remote worker execution loop.
# Running generate_until requests: 100%|██████████████████████| 128/128 [15:50<00:00,  7.43s/it]
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8516|±  |0.0315|
# |     |       |strict-match    |     5|exact_match|↑  |0.8906|±  |0.0277|

#############################
# Qwen End
#############################


# QUANT_CONFIG=./scripts/inc_quant_with_fp8kv_config_post_process.json \
# python ./scripts/run_example_tp_local.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_num_seqs 1 \
#     --fp8_kv_cache



# 
# QUANT_CONFIG=./scripts/inc_quant_with_fp8kv_config_post_process.json \
# export OFFICIAL_FP8_MODEL="/mnt/weka/data/pytorch/llama2/Llama-2-7b-hf/"
# export OFFICIAL_FP8_MODEL=/dev/shm/hub/ibm-granite/granite-20b-code-instruct-8k

# QUANT_CONFIG="./scripts/inc_measure_config_ibm.json" \
# python ./scripts/run_example_single_node.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_num_seqs 1 \
#     --inc

# QUANT_CONFIG="./scripts/inc_quant_config_ibm.json" \
# python ./scripts/run_example_single_node.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_num_seqs 1 \
#     --inc \
#     --fp8_kv_cache



# # for i in {20..25}; do
# #     ray stop --force
# #     echo "Running iteration $i"
# #     export OFFICIAL_FP8_MODEL="/mnt/disk2/hf_models/DeepSeek-R1-G2/"
# #     export OFFICIAL_FP8_MODEL="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC/"
# #     export VLLM_NUM_LAYERS=$i
# #     # test normal
# #     # export OFFICIAL_FP8_MODEL="/mnt/disk2/hf_models/DeepSeek-R1"

# #     # model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
# #     # model_path=/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC/
# #     unset PT_HPU_RECIPE_CACHE_CONFIG
# #     export VLLM_MOE_N_SLICE=1
# #     export VLLM_MLA_DISABLE_REQUANTIZATION=1
# #     export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
# #     export VLLM_REQUANT_FP8_INC=1
# #     export VLLM_ENABLE_RUNTIME_DEQUANT=1
# #     export VLLM_DISABLE_MARK_SCALES_AS_CONST=1

# #     VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
# #     GRAPH_VISUALIZATION=1 \
# #     VLLM_REQUANT_FP8_INC=1 \
# #     VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# #     VLLM_MOE_N_SLICE=1 \

# #     QUANT_CONFIG=./scripts/inc_quant_per_channel_bf16kv.json \
# #     python ./scripts/run_example_tp.py \
# #         --model ${OFFICIAL_FP8_MODEL} \
# #         --tokenizer ${OFFICIAL_FP8_MODEL} \
# #         --osl 16 \
# #         --max_num_seqs 1
        
# #     # ==========================================
# #     ray stop --force

# #     export OFFICIAL_FP8_MODEL="/mnt/disk2/hf_models/DeepSeek-R1-G2/"
# #     # export OFFICIAL_FP8_MODEL="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC/"
# #     # export VLLM_NUM_LAYERS=25
# #     # test normal
# #     # export OFFICIAL_FP8_MODEL="/mnt/disk2/hf_models/DeepSeek-R1"

# #     # model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/
# #     # model_path=/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC/
# #     unset PT_HPU_RECIPE_CACHE_CONFIG
# #     export VLLM_MOE_N_SLICE=1
# #     export VLLM_MLA_DISABLE_REQUANTIZATION=1
# #     export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
# #     export VLLM_REQUANT_FP8_INC=1
# #     export VLLM_ENABLE_RUNTIME_DEQUANT=1
# #     export VLLM_DISABLE_MARK_SCALES_AS_CONST=1

# #     VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
# #     GRAPH_VISUALIZATION=1 \
# #     VLLM_REQUANT_FP8_INC=1 \
# #     VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# #     VLLM_MOE_N_SLICE=1 \

# #     QUANT_CONFIG=./scripts/inc_quant_per_channel_bf16kv.json \
# #     python ./scripts/run_example_tp.py \
# #         --model ${OFFICIAL_FP8_MODEL} \
# #         --tokenizer ${OFFICIAL_FP8_MODEL} \
# #         --osl 16 \
# #         --max_num_seqs 1

# # done




# # # Test blocks 
# # # 16k
# # max_model_len=16384
# # max_num_batched_tokens=16384
# # max_num_seqs=64
# # input_min=1
# # input_max=16384
# # output_max=16384

# # VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
# # GRAPH_VISUALIZATION=1 \
# # VLLM_REQUANT_FP8_INC=1 \
# # VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# # VLLM_MOE_N_SLICE=1 \
# # QUANT_CONFIG=inc_quant_per_channel_bf16kv.json \
# # python run_example_tp.py \
# #     --model ${OFFICIAL_FP8_MODEL} \
# #     --tokenizer ${OFFICIAL_FP8_MODEL} \
# #     --isl  1024 \
# #     --osl  1024 \
# #     --max_num_seqs ${max_num_seqs} \
# #     --max_model_len ${max_model_len} \
# #     --enforce_eager \
# #     --dataset pile \
# #     --nprompts ${max_num_seqs}



# export OFFICIAL_FP8_MODEL="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC/"

# VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0 \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# VLLM_PROMPT_BS_BUCKET_MIN=1 \
# VLLM_PROMPT_BS_BUCKET_MAX=1 \
# VLLM_PROMPT_SEQ_BUCKET_MIN=1024 \
# VLLM_PROMPT_SEQ_BUCKET_STEP=512 \
# VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
# VLLM_DECODE_BS_BUCKET_MIN=1 \
# VLLM_DECODE_BS_BUCKET_MAX=1 \
# VLLM_REQUANT_FP8_INC=1 \
# VLLM_MOE_N_SLICE=1 \
# QUANT_CONFIG=inc_measure_with_fp8kv_config.json \
#     python run_example_tp.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_num_seqs 1 \
#     --nprompts 1024 \
#     --enforce_eager \
#     --max_model_len 2048 \
#     --dataset pile 2>&1 | tee ./woq_logs/prepare.g2.full.enable_graph.DeepSeek-R1-G2-INC.5120


