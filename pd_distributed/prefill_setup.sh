#CT4 prefill node

cd ~/kunshang
git clone  https://github.com/yangulei/vllm-fork
cd vllm-fork
git checkout deepseek_r1_g2
docker run -it -d --runtime=habana --name deepseek-pd -v `pwd`:/workspace/vllm/ -v /software/data/disk10:/software/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/software/data/ artifactory-kfs.habana-labs.com/docker-local/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.20.0-521 /bin/bash



#inside docker
docker exec -it deepseek-pd /bin/bash

apt update
apt install sudo etcd -y

#proxy:

export http_proxy=http://child-prc.intel.com:913
export https_proxy=http://child-prc.intel.com:913
export no_proxy=10.112.*,localhost,127.0.0.1


#install vllm
cd /workspace/vllm/kunshang/vllm-fork
pip install -r requirements-hpu.txt
pip install -r  .jenkins/requirements-test-hpu.txt
pip install modelscope quart

VLLM_TARGET_DEVICE=hpu python3 setup.py develop

#Optional: verify you can run deepseek r1 example.


# install mooncake, refer https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md
cd ..
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh

mkdir build
cd build
cmake ..
make -j
make install

cd ../../vllm-fork


# mooncake setup
#start etcd on prefill node
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379 >etcd.log 2>&1 &

# create 

#start server
export VLLM_EP_SIZE=8
export VLLM_SKIP_WARMUP=True
MOONCAKE_CONFIG_PATH=./mooncake.json VLLM_USE_MODELSCOPE=True python3 -m vllm.entrypoints.openai.api_server --model /software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/ --port 8100 --max-model-len 4096 --gpu-memory-utilization 0.9 -tp 8 --max-num-seqs 32 --trust-remote-code  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":2e9}'

