    # --max-gen-toks 2048 \# VLLM_USE_STANDALONE_COMPILE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn lm_eval --model vllm   \
#     --model_args "pretrained=/home/yiliu7/models/deepseek-ai/DeepSeek-R1,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_expert_parallel=True,enforce_eager=True"   \
#         --tasks piqa,mmlu --batch_size 128 \
#             --log_samples --output_path lmeval.ds.piqa_mmlu.out \
#                         --trust_remote_code \
#                     --show_config 2>&1 | tee lmeval.log.ds.piqa_mmlu.2nd.txt
                    

pip install lm-eval[api]
timestamp=$(date +%Y%m%d-%H%M%S)
log_file=server.$timestamp.log
model_path=/home/yiliu7/models/deepseek-ai/DeepSeek-R1
model_path=/home/yiliu7/models/meta-llama/Llama-3.1-405B/
taskname=piqa,mmlu,hellaswag
# taskname=gsm8k
#replace the , in taskname with '--'

# taskname=mmlu_high_school_mathematics_generative
taskname_str=${taskname//,/--}
model_basename=$(basename $model_path)
output_log_file_name=lm_eval_output_${model_basename}_${taskname_str}


# HF_ALLOW_CODE_EVAL=1 lm_eval \
#     --model local-completions \
#     --tasks $taskname \
#     --model_args "model=$model_path,base_url=http://127.0.0.1:8000/v1/completions,max_length=8192,max_gen_toks=2048", \
#     --batch_size 128 \
#     --confirm_run_unsafe_code \
#     --gen_kwargs="max_length=8192,max_gen_toks=2048" \
#     --log_samples \
#     --trust_remote_code \
#     --output_path $output_log_file_name 2>&1 | tee "${output_log_file_name}.out"
    
export model_path="/mnt/disk5/lmsys/gpt-oss-20b-bf16"
model_path="/mnt/disk5/lmsys/gpt-oss-20b-bf16"
# curl -X POST http://127.0.0.1:8000/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "/mnt/disk5/lmsys/gpt-oss-20b-bf16",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'

taskname=gsm8k
# taskname=gsm8k
#replace the , in taskname with '--'

# taskname=mmlu_high_school_mathematics_generative
taskname_str=${taskname//,/--}
model_basename=$(basename $model_path)
output_log_file_name=lm_eval_output_${model_basename}_${taskname_str}


task_name="gsm8k"
task_name="gsm8k_oss"

# lm-eval --model local-chat-completions \
#     --model_args pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/chat/completions,max_length=16384,max_gen_toks=8192,num_concurrent=128 \
#     --tasks ${task_name} \
#     --apply_chat_template \
#     --gen_kwargs="max_length=16384,max_gen_toks=8192" 


lm-eval --model local-completions \
    --model_args pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/completions,max_length=16384,max_gen_toks=8192,num_concurrent=128 \
    --tasks ${task_name} \
    --limit 128 \
    --gen_kwargs="max_length=16384,max_gen_toks=8192" 
    # --num_fewshot 1 


# - TP 8
# Requesting API: 100%|██████████████████████████████████████████████████████████████████████| 128/128 [00:58<00:00,  2.19it/s]
# 2025-08-20:03:04:44,469 INFO     [lm_eval.loggers.evaluation_tracker:272] Output path not provided, skipping saving results aggregated
# local-chat-completions (pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/chat/completions,max_gen_toks=2048,num_concurrent=128), gen_kwargs: (max_length=8192,max_gen_toks=2048), limit: 128.0, num_fewshot: None, batch_size: 1
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8359|±  |0.0329|
# |     |       |strict-match    |     5|exact_match|↑  |0.0312|±  |0.0154|

# - TP 8 - low
# local-chat-completions (pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/chat/completions,max_gen_toks=2048,num_concurrent=128), gen_kwargs: (max_length=8192,max_gen_toks=2048), limit: 128.0, num_fewshot: None, batch_size: 1
# |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k_oss|      3|flexible-extract|     0|exact_match|↑  |0.8438|±  |0.0322|
# |         |       |strict-match    |     0|exact_match|↑  |0.6641|±  |0.0419|

# - TP8 high
# local-chat-completions (pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/chat/completions,max_length=16384,max_gen_toks=8192,num_concurrent=128), gen_kwargs: (max_length=16384,max_gen_toks=8192), limit: None, num_fewshot: None, batch_size: 1
# |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k_oss|      3|flexible-extract|     0|exact_match|↑  |0.7134|±  |0.0125|
# |         |       |strict-match    |     0|exact_match|↑  |0.4814|±  |0.0138|


# - TP8 low 
# local-completions (pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/completions,max_length=16384,max_gen_toks=8192,num_concurrent=128), gen_kwargs: (max_length=16384,max_gen_toks=8192), limit: None, num_fewshot: None, batch_size: 1
# |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k_oss|      3|flexible-extract|     0|exact_match|↑  |0.8658|±  |0.0094|
# |         |       |strict-match    |     0|exact_match|↑  |0.6892|±  |0.0127|


# local-chat-completions (pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/chat/completions,max_gen_toks=1024,num_concurrent=128), gen_kwargs: (None), limit: 128.0, num_fewshot: None, batch_size: 1
# |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k_oss|      3|flexible-extract|     0|exact_match|↑  |0.8438|±  |0.0322|
# |         |       |strict-match    |     0|exact_match|↑  |0.6406|±  |0.0426|

# HF_ALLOW_CODE_EVAL=1 lm_eval \
#     --model local-completions \
#     --tasks $taskname \
#     --model_args "model=$model_path,base_url=http://127.0.0.1:8000/v1/completions,max_length=8192,max_gen_toks=2048", \
#     --batch_size 128 \
#     --confirm_run_unsafe_code \
#     --gen_kwargs="max_length=8192,max_gen_toks=2048" \
#     --log_samples \
#     --trust_remote_code \
#     --output_path $output_log_file_name 2>&1 | tee "${output_log_file_name}.out"