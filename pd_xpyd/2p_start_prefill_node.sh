#set -x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

export HCCL_OVER_OFI=1
export HCCL_GAUDI_DIRECT=1
export HCCL_SOCKET_IFNAME=enp24s0f0np0
export LD_LIBRARY_PATH=/opt/libfabric/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MOONCAKE_CONFIG_PATH="$BASH_DIR"/mooncake_${1:-g12}.json

source "$BASH_DIR"/dp_2p_env.sh
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY

ray start --address="${2:-10.239.129.9:6886}"

