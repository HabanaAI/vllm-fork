ModelName=Qwen/Qwen2.5-VL-3B-Instruct
# without warmups
bash run_offline.sh -m $ModelName -i snowscat --multiple_prompts --skip_warmup
bash run_offline.sh -m $ModelName -i snowscat --skip_warmup
bash run_offline.sh -m $ModelName -i synthetic --multiple_prompts --skip_warmup
bash run_offline.sh -m $ModelName -i synthetic --skip_warmup
bash run_offline.sh -m $ModelName -v --skip_warmup
bash run_offline.sh -m $ModelName -v --multiple_prompts --skip_warmup
# with warmups
bash run_offline.sh -m $ModelName -i snowscat --multiple_prompts
bash run_offline.sh -m $ModelName -i snowscat
bash run_offline.sh -m $ModelName -i synthetic --multiple_prompts
bash run_offline.sh -m $ModelName -i synthetic
bash run_offline.sh -m $ModelName -v
bash run_offline.sh -m $ModelName -v --multiple_prompts
