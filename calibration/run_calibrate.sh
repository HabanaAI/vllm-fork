ModelName=Qwen/Qwen2.5-3B-Instruct
#ModelName=Qwen/Qwen2.5-VL-3B-Instruct

CalDir=$PWD

# pre-step 1: Download the datasets
wget https://rclone.org/install.sh
chmod a+x ./install.sh
bash install.sh
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca /root/open_orca -P
gzip -c -d /root/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz >/root/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl

# pre-step 2: install oh 
cd $CalDir
git clone git clone https://github.com/huggingface/optimum-habana.git -b v1.17.0
cd optimum-habana
pip install . -q

# step 1: install the vllm
cd $CalDir
git clone https://github.com/HabanaAI/vllm-fork.git -b v0.6.6.post1+Gaudi-1.20.0
cd vllm-fork
pip install -r requirements-hpu.txt
python3 setup.py develop
pip install datasets

## step 2: install the vllm-hpu-extension
cd $CalDir
git clone https://github.com/HabanaAI/vllm-hpu-extension.git -b v1.20.0
cd vllm-hpu-extension
pip install -e .
cd calibration

### qwen2.5-vl crashes... and vision doesn't run in calibration. WA: to run regular qwen2.5, update the weight as needed for langugae model. 
if [[ "$ModelName" == *"Qwen2.5"* ]]; then
    VLLM_SKIP_WARMUP=true VLLM_FP32_SOFTMAX=false bash calibrate_model.sh -m Qwen/Qwen2.5-3B-Instruct -d /root/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl -o g2 -b 128 -t 1 -l 1024
else
    echo "INFO this script is for qwen model ...continue with your own risk"
    sleep 5
    VLLM_SKIP_WARMUP=true VLLM_FP32_SOFTMAX=false bash calibrate_model.sh -m $ModelName -d /root/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl -o g2 -b 128 -t 1 -l 1024
fi

# step3: finish if  qwen2.5, otherwise for qwen2.5-vl continue to
if [[ "$ModelName" == *"Qwen2.5-VL"* ]]; then
    cp -r $CalDir/vllm-hpu-extension/calibration/g2/qwen2.5-3b-instruct/ $CalDir/vllm-hpu-extension/calibration/g2/qwen2.5-vl-3b-instruct/
    cp -r $CalDir/hqt/qwen2.5-vl-3b-instruct/maxabs_*json $CalDir/vllm-hpu-extension/calibration/g2/qwen2.5-vl-3b-instruct/
    # update the npz file for -VL mdoel and overwrite it.
    python $CalDir/update_npz.py $CalDir/vllm-hpu-extension/calibration/g2/qwen2.5-3b-instruct/g2/inc_output_hooks_maxabs_MAXABS_HW_0_1.npz
    mv tmp_new.npz $CalDir/vllm-hpu-extension/calibration/g2/qwen2.5-vl-3b-instruct/g2/inc_output_hooks_maxabs_MAXABS_HW_0_1.npz
fi
