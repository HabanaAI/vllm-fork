model_path=/mnt/disk5/lmsys/gpt-oss-20b-bf16
model_path=/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-20b-bf16
basename=$(basename $model_path)
taskname=gsm8k
# taskname=gsm8k
#replace the , in taskname with '--'
taskname_str=${taskname//,/--}
model_basename=$(basename $model_path)
output_log_file_name=lm_eval_output_${model_basename}_${taskname_str}


task_name="gsm8k"
# task_name="gsm8k_oss"

# lm-eval --model local-chat-completions \
#     --model_args pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/chat/completions,max_length=16384,max_gen_toks=8192,num_concurrent=128 \
#     --tasks ${task_name} \
#     --apply_chat_template \
#     --gen_kwargs="max_length=16384,max_gen_toks=8192" 


lm-eval --model local-completions \
    --model_args pretrained=${model_path},base_url=http://localhost:8688/v1/completions,max_length=8192,max_gen_toks=4096 \
    --tasks ${task_name} \
    --batch_size 32 \
    --limit 32 \
    --gen_kwargs="max_length=8192,max_gen_toks=4096" 