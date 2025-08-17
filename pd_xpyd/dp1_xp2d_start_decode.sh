BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=10.112.242.154,localhost,127.0.0.1
export no_proxy=10.112.242.154,localhost,127.0.0.1


if [ -z "$1" ]; then
    echo "please input the tp size"
    echo "run with default mode n=1"
    TP_SIZE=1
else
    TP_SIZE=$1
fi

export HCCL_OVER_OFI=1
export HCCL_GAUDI_DIRECT=1
export HCCL_SOCKET_IFNAME=enp24s0f0np0
export LD_LIBRARY_PATH=/opt/libfabric/lib

source "$BASH_DIR"/dp_start_decode.sh G3D-sys04 16 $TP_SIZE 1 "10.112.242.153"
