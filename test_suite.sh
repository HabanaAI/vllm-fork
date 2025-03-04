ModelName=Qwen/Qwen2.5-VL-3B-Instruct
RunBasic=true
RunPytests=true
RunLargeImages=false

#Basic tests
if $RunBasic; then
	# without warmups
	bash run_offline.sh -m $ModelName -i snowscat --multiple_prompts --skip_warmup
	bash run_offline.sh -m $ModelName -i snowscat --skip_warmup
	bash run_offline.sh -m $ModelName -i synthetic --multiple_prompts --skip_warmup
	bash run_offline.sh -m $ModelName -i synthetic --skip_warmup
	bash run_offline.sh -m $ModelName -v --skip_warmup
	bash run_offline.sh -m $ModelName -v --multiple_prompts --skip_warmup
	bash run_offline.sh -m $ModelName -t --skip_warmup
	bash run_offline.sh -m $ModelName -t --multiple_prompts --skip_warmup
	# with warmups
	bash run_offline.sh -m $ModelName -i snowscat --multiple_prompts
	bash run_offline.sh -m $ModelName -i snowscat
	bash run_offline.sh -m $ModelName -i synthetic --multiple_prompts
	bash run_offline.sh -m $ModelName -i synthetic
	bash run_offline.sh -m $ModelName -v
	bash run_offline.sh -m $ModelName -v --multiple_prompts
	bash run_offline.sh -m $ModelName -t
	bash run_offline.sh -m $ModelName -t --multiple_prompts
fi

#pytests
if $RunPytests; then pytest tests/models/decoder_only/vision_language/test_models.py -s -v -k "[qwen2_5"; fi

# larger image sizes (currently failing)
if $RunLargeImages; then
	bash run_offline.sh -m $ModelName -i snowscat --multiple_prompts --skip_warmup --image_width 1800 --image_height 1200
	bash run_offline.sh -m $ModelName -i snowscat --multiple_prompts --skip_warmup --image_width 3600 --image_height 2400
	bash run_offline.sh -m $ModelName -i synthetic --skip_warmup --image_width 1800 --image_height 1200
fi
