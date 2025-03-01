# install

```
git clone https://github.com/HabanaAI/vllm-fork.git; git checkout deepseek_r1
cd vllm;  pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;
```

# prepare model

```
huggingface-cli download --local-dir ${YOUR_PATH}/DeepSeek-R1 deepseek-ai/DeepSeek-R1
python scripts/convert_block_fp8_to_channel_fp8.py --model_path ${YOUR_PATH}/DeepSeek-R1 --qmodel_path ${YOUR_PATH}/DeepSeek-R1-static --input_scales_path scripts/DeepSeek-R1-BF16-w8afp8-static-no-ste_input_scale_inv.pkl.gz
```

# run example
```
python scripts/run_example_tp.py --model ${YOUR_PATH}/DeepSeek-R1-static
```

# run benchmark
```
bash scripts/run_static-online-fp8-i1k-o1k-ep8-bestperf.sh
```

# run with multi nodes
```
# head node
HABANA_VISIBLE_MODULES='0,1,2,3,4,5,6,7'  \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
ray start --head --resources='{"HPU": 8, "TPU": 0}'
```

```
# worker node
HABANA_VISIBLE_MODULES='0,1,2,3,4,5,6,7'  \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
ray start --address='${head_ip}:6379' --resources='{"HPU": 8, "TPU": 0}'
```

```
python scripts/run_example_tp_2nodes.py --model ${YOUR_PATH}/DeepSeek-R1-static
```

