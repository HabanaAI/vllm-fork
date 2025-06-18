#!/bin/bash

export https_proxy=http://child-igk.intel.com:912
pip install -v -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu python3 setup.py install
apt update
apt install etcd -y
pip3 install mooncake-transfer-engine==0.3.0b3

unset https_proxy