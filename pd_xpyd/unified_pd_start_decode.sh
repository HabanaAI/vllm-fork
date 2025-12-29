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
# Multi-instance support: optional instance indices
D_INSTANCE_IDX="${2:-}"
D_INTRA_INSTANCE_IDX="${3:-}"

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
# Include instance indices in log file name if available
if [[ -n "$D_INSTANCE_IDX" && -n "$D_INTRA_INSTANCE_IDX" ]]; then
    log_file="$log_dir/decode_${D_INSTANCE_IDX}_${D_INTRA_INSTANCE_IDX}_${timestamp}.log"
else
    log_file="$log_dir/decode${INSTANCE_IDX}_${timestamp}.log"
fi


#DP_MASTER_IP=${USR_DP_MASTER_IP:-10.239.129.21}
echo DP_MASTER_IP=$DP_MASTER_IP

# Function to get local IP address filtered by MOONCAKE_LOCAL_ADDR_PREFIX
get_local_ip() {
    local addr_prefix="${MOONCAKE_LOCAL_ADDR_PREFIX:-}"
    local local_ip=""
    if [ -z "$addr_prefix" ]; then
        echo "Error: MOONCAKE_LOCAL_ADDR_PREFIX is not set" >&2
        return 1
    fi
    # Get all IPs from hostname -I and filter by prefix, print first match
    local_ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 "^$addr_prefix")
    if [ -z "$local_ip" ]; then
        # If no match, fallback to empty string
        local_ip=""
    fi
    echo "$local_ip"
}

# Get mooncake config path and create/overwrite JSON file
export MOONCAKE_CONFIG_PATH="$BASH_DIR/mooncake_`hostname`.json"

# Only run mooncake config generation if both ETCD_META_SERVER and MOONCAKE_SERVER are defined
if [[ -z "${ETCD_META_SERVER:-}" || -z "${MOONCAKE_SERVER:-}" ]]; then
    echo "################################################################################"
    echo "Warning: ETCD_META_SERVER or MOONCAKE_SERVER is not defined. Use mooncake config file as is."
    echo "################################################################################"
else
    ETCD_META_SERVER_VALUE=${ETCD_META_SERVER}
    MOONCAKE_SERVER_VALUE=${MOONCAKE_SERVER}
    # Get local IP and server addresses
    local_hostname_ip=$(get_local_ip)
    # Ensure metadata_server has etcd:// prefix if not present
    if [[ ! "$ETCD_META_SERVER_VALUE" =~ ^etcd:// ]]; then
        ETCD_META_SERVER_VALUE="etcd://${ETCD_META_SERVER_VALUE}"
    fi

    # Determine device_name array size (default to 16, or use CARDS_PER_NODE if available)
    DEVICE_COUNT=${CARDS_PER_NODE}

    # Load hostname to device mapping from external file
    hostname=$(hostname)
    HOST_DEVICE_MAP_FILE="${HOST_DEVICE_MAP_FILE:-$BASH_DIR/host_cx7_map.sh}"

    if [ ! -f "$HOST_DEVICE_MAP_FILE" ]; then
        echo "ERROR: Host device map file not found: $HOST_DEVICE_MAP_FILE" >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "$HOST_DEVICE_MAP_FILE"

    # Look up device array for current hostname
    if [[ -z "${HOST_DEVICE_MAP[$hostname]:-}" ]]; then
        echo "ERROR: Unknown hostname $hostname (not found in $HOST_DEVICE_MAP_FILE)" >&2
        exit 1
    fi

    # Convert space-separated string to array
    read -r -a DEVICE_NAME_ARRAY <<< "${HOST_DEVICE_MAP[$hostname]}"

    # Check that array size matches DEVICE_COUNT
    if [ "${#DEVICE_NAME_ARRAY[@]}" -ne "$DEVICE_COUNT" ]; then
        echo "Error: DEVICE_NAME_ARRAY size (${#DEVICE_NAME_ARRAY[@]}) does not match DEVICE_COUNT ($DEVICE_COUNT)" >&2
        exit 1
    fi

    # Join device names with double quotes for JSON
    DEVICE_NAME_ARRAY_JSON=$(printf '"%s",' "${DEVICE_NAME_ARRAY[@]}")
    DEVICE_NAME_ARRAY_JSON="[${DEVICE_NAME_ARRAY_JSON%,}]"

    # Create/overwrite mooncake JSON file
    cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
        "local_hostname": "$LOCAL_HOSTNAME_IP",
        "metadata_server": "$ETCD_META_SERVER_VALUE",
        "protocol": "rdma",
        "device_name": $DEVICE_NAME_ARRAY_JSON,
        "master_server_address": "$MOONCAKE_SERVER_VALUE"
}
EOF

    echo "Generated mooncake config at $MOONCAKE_CONFIG_PATH:"
    cat "$MOONCAKE_CONFIG_PATH"
fi

source "$BASH_DIR"/dp_start_decode.sh $INSTANCE_IDX

