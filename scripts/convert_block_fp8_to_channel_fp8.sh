#! /bin/bash

pip install -q compress_pickle

original_model_path=/data/models/DeepSeek-R1
static_model_path=/data/models/DeepSeek-R1-static
input_scales_path=DeepSeek-R1-BF16-w8afp8-static-no-ste_input_scale_inv.pkl.gz
inc_config_path=config_static.json

# convert weights and scales in *.safetensors and update model.safetensors.index.json
python convert_block_fp8_to_channel_fp8.py \
    --model_path $original_model_path \
    --qmodel_path $static_model_path \
    --input_scales_path $input_scales_path \
    --inc_config $inc_config_path