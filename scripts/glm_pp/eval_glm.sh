export no_proxy="localhost, 127.0.0.1, ::1"
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

lm-eval --model local-completions \
    --model_args pretrained=/mnt/disk8/yiliu7/zai-org/GLM-4.5-Air-FP8-G2,base_url=http://localhost:8688/v1/completions,max_length=8192,max_gen_toks=4096,num_concurrent=2 \
    --tasks ${task_name} \
    --batch_size 128 \
    --gen_kwargs="max_length=8192,max_gen_toks=4096"
    


