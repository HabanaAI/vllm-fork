ModelName=Qwen/Qwen2.5-VL-3B-Instruct
#ModelName=meta-llama/Llama-3.2-11B-Vision-Instruct
RunBasic=true
RunPytests=true
RunLargeImages=true
RunOnlineTests=true
RunMemBenchAnalysis=false
RandomizedImage=true
RandomShuffle=true
RunProcessorUT=true

#Basic tests
if $RunBasic; then
	# without warmups
	bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup
	bash run_offline.sh -m $ModelName -i --skip_warmup
	bash run_offline.sh -m $ModelName -v --skip_warmup
	bash run_offline.sh -m $ModelName -v --multiple_prompts --skip_warmup
	bash run_offline.sh -m $ModelName -t --skip_warmup
	bash run_offline.sh -m $ModelName -t --multiple_prompts --skip_warmup
	# with warmups
	bash run_offline.sh -m $ModelName -i --multiple_prompts
	bash run_offline.sh -m $ModelName -i
	bash run_offline.sh -m $ModelName -v
	bash run_offline.sh -m $ModelName -v --multiple_prompts
	bash run_offline.sh -m $ModelName -t
	bash run_offline.sh -m $ModelName -t --multiple_prompts
fi

# Run a case with randomized shuffle of input values and len and compare to a reference
if $RandomShuffle; then
	VLLM_GRAPH_RESERVED_MEM=0.4 bash run_offline.sh -m $ModelName -i --random_image_size --iter 5 --image_width 1120 --image_height 1120 --mix_prompt_lenght
	VLLM_GRAPH_RESERVED_MEM=0.4 bash run_offline.sh -m $ModelName -i --random_image_size --iter 5 --image_width 1120 --image_height 1120 --multiple_prompts
fi

#Unit Test for Processor
if $RunProcessorUT; then
  pytest /workspace/vllm/tests/models/multimodal/processing/test_qwen2_5_vl.py -s -v;
fi

#pytests
if $RunPytests && [[ "$ModelName" == *"Qwen"* ]]; then pytest tests/models/decoder_only/vision_language/test_models.py -s -v -k "[qwen2_5"; fi

# larger image sizes
if $RunLargeImages; then
	if [[ "$ModelName" == *"Qwen"* ]]; then
		VLLM_GRAPH_RESERVED_MEM=0.2 bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 1800 --image_height 1200
		VLLM_GRAPH_RESERVED_MEM=0.4 bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 2400 --image_height 1800
		VLLM_GRAPH_RESERVED_MEM=0.6 bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 2400 --image_height 1800 --iter 5
		VLLM_LIMIT_HPU_GRAPH=true bash run_offline.sh -m $ModelName -i --skip_warmup --image_width 3600 --image_height 2400
		VLLM_GRAPH_RESERVED_MEM=0.2 bash run_offline.sh -m $ModelName -i --skip_warmup --image_width 1800 --image_height 1200
	elif [[ "$ModelName" == *"Llama"* ]]; then
		bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 1800 --image_height 1200
		bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 2400 --image_height 1800
		bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 3600 --image_height 2400
		bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 3600 --image_height 3600
	fi
fi

# Online tests
if $RunOnlineTests; then bash test_multimodal.sh -m $ModelName; fi

# Bechmark for different VLLM_GRAPH_RESERVED_MEM and gpu_mem_usage values
if $RunMemBenchAnalysis; then
	filename=$(echo $ModelName | rev | cut -d"/" -f1 | rev).MemAnalysis.$(date -u +%Y%m%d%H%M).$(hostname).log
	for i in {.2,.3,.4,.5,.6}; do
		cmd="VLLM_GRAPH_RESERVED_MEM=$i bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 1800 --image_height 1200 --iter 5"
		echo $cmd >>$filename && eval $cmd 2>&1 | tee -a $filename
	done
	cmd="VLLM_LIMIT_HPU_GRAPH=true bash run_offline.sh -m Qwen/Qwen2.5-VL-3B-Instruct -i --multiple_prompts --skip_warmup --image_width 1800 --image_height 1200 --iter 5"
	echo $cmd >>$filename && eval $cmd 2>&1 | tee -a $filename
	for i in {.9,.8,.7,.6,.5}; do
		cmd="VLLM_GRAPH_RESERVED_MEM=.2 bash run_offline.sh -m $ModelName -i --multiple_prompts --skip_warmup --image_width 1800 --image_height 1200 --iter 5 -gmu $i"
		echo $cmd >>$filename && eval $cmd 2>&1 | tee -a $filename
	done
fi

# Tests with randomized image sizes
if $RandomizedImage; then
	VLLM_GRAPH_RESERVED_MEM=0.4 bash run_offline.sh -m $ModelName -i --skip_warmup --multiple_prompts --random_image_size --iter 5
	VLLM_GRAPH_RESERVED_MEM=0.4 bash run_offline.sh -m $ModelName -i --skip_warmup --random_image_size --iter 5
fi

if $RunOnlineDatasets; then
	bash run_online.sh -m $ModelName --skip_warmup --hpu -ds lmarena-ai/vision-arena-bench-v0.1,LIME-DATA/infovqa,echo840/OCRBench --num-prompts 500
	bash run_online.sh -m $ModelName --skip_warmup --hpu --run_sonet --num-prompts 1000
fi
