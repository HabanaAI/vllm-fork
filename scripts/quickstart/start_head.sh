#!/bin/bash
# lsof -t -i :8850 | xargs -r kill -9
source ./set_head_node.sh
ray start --head --node-ip-address=10.239.129.40 --port=8850  --num-cpus 160
