#! /bin/bash

pip install compress_pickle

original_model_path=/hf_models/DeepSeek-R1
output_model_root=/models
dynamic_model_path=${output_model_root}/DeepSeek-R1-G2-dynamic
static_model_path=${output_model_root}/DeepSeek-R1-G2-static
input_scales_path=DeepSeek-R1-BF16-w8afp8-static-no-ste_input_scale_inv.pkl.gz

workdir=`pwd`

# convert weights and scales in *.safetensors and update model.safetensors.index.json
python convert_block_fp8_to_channel_fp8.py \
    --model_path $original_model_path \
    --qmodel_path $dynamic_model_path \
    --input_scales_path $input_scales_path

# copy all except for the *.safetensors and model.safetensors.index.json
cp -r -n $original_model_path/* $dynamic_model_path
# use the config.json for dynamic activation quant
cp $workdir/config_dynamic.json $dynamic_model_path/config.json

mkdir -p $static_model_path
cd $static_model_path
# re-use all the dynamic activation quant weights by soft link
ln -s ../DeepSeek-R1-G2-dynamic/* .
# use the config.json for static activation quant
rm config.json
cp $workdir/config_static.json config.json