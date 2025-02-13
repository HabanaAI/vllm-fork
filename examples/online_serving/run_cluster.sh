#!/bin/bash

# Check for minimum number of required arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_address --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

# Assign the first three arguments and shift them away
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker
PATH_TO_HF_HOME="$4"
shift 4

# Additional arguments are passed directly to the Docker command
ADDITIONAL_ARGS=("$@")

# Validate node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Define a function to cleanup on EXIT signal
cleanup() {
    docker stop node
    docker rm node
}
trap cleanup EXIT

# Command setup for head or worker node
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Run the docker command with the user specified parameters and additional arguments
docker run \
    -td \
    --entrypoint /bin/bash \
    --network host \
    --ipc=host \
    --name node \
    --shm-size 10.24g \
    --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e GLOO_SOCKET_IFNAME="ens9f1np1" \
    -e HCCL_SOCKET_IFNAME="ens9f1np1" \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    -v "/tmp/logs:/workspace/logs" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}"

# This is likely not neccesary. Just need the last line. This is done here so hccl_demo can be ran.
#docker exec -it node bash -c "apt install make"
#docker exec -it node bash -c "export no_proxy=localhost; export http_proxy=http://proxy-dmz.intel.com:911; export https_proxy=http://proxy-dmz.intel.com:912; git clone https://github.com/HabanaAI/hccl_demo.git"
#docker exec -it node bash -c "export no_proxy=localhost; export http_proxy=http://proxy-dmz.intel.com:911; export https_proxy=http://proxy-dmz.intel.com:912; export REQUIRED_VERSION=1.20.0; wget https://github.com/ofiwg/libfabric/releases/download/v1.20.0/libfabric-1.20.0.tar.bz2 -P /tmp/libfabric; pushd /tmp/libfabric; tar -xf libfabric-1.20.0.tar.bz2; export LIBFABRIC_ROOT=/workspace/libfabric_root/; mkdir -p /workspace/libfabric_root/; chmod 777 /workspace/libfabric_root/; cd libfabric-1.20.0/; ./configure --prefix=/workspace/libfabric_root/ --with-synapseai=/usr; make -j 32; make install; popd; rm -rf /tmp/libfabric; export LD_LIBRARY_PATH=/workspace/libfabric_root/lib:$LD_LIBRARY_PATH; fi_info --version"
#docker exec -it node bash -c "export no_proxy=localhost; export http_proxy=http://proxy-dmz.intel.com:911; export https_proxy=http://proxy-dmz.intel.com:912; git clone https://github.com/HabanaAI/hccl_ofi_wrapper.git; export LIBFABRIC_ROOT=/workspace/libfabric_root/; cd hccl_ofi_wrapper; make; cp libhccl_ofi_wrapper.so /usr/lib/habanalabs/libhccl_ofi_wrapper.so; ldconfig; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/habanalabs/"

# If running ray
#docker exec -it node bash -c "${RAY_START_CMD}"

# If running MP
docker exec -it node bash -c "hl-smi -l 10"
