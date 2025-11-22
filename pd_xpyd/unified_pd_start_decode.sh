set -euo pipefail

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=10.112.242.154,localhost,127.0.0.1
export no_proxy=10.112.242.154,localhost,127.0.0.1

#if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
#    echo "Usage: $0 <TP_SIZE> <HOSTNAME> <INSTANCE_IDX>"
#    exit 1
#fi

#TP_SIZE="$1"
#HOSTNAME="$2"
INSTANCE_IDX="$1"

if [ "${DECODE_NEED_SCALEOUT:-1}" == "1" ]; then
    export HCCL_OVER_OFI=1
    export HCCL_GAUDI_DIRECT=1
    if [ "$(hostname)" == "G15" ] || [ "$(hostname)" == "G16" ] || [ "$(hostname)" == "G13" ]; then
        export HCCL_SOCKET_IFNAME=ens20f1
    else
        export HCCL_SOCKET_IFNAME=enp24s0f0np0
    fi
    export LD_LIBRARY_PATH=/opt/libfabric/lib:${LD_LIBRARY_PATH:-}
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="xpyd_logs"
mkdir -p "$log_dir"
log_file="$log_dir/decode${INSTANCE_IDX}_${timestamp}.log"


#DP_MASTER_IP=${USR_DP_MASTER_IP:-10.239.129.21}
echo DP_MASTER_IP=$DP_MASTER_IP
source "$BASH_DIR"/dp_start_decode.sh $INSTANCE_IDX

