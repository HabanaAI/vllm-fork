#!/bin/bash
set -euo pipefail

# Default values
LOCAL_HOSTNAME=""
METADATA_SERVER=""
PROTOCOL=""
DEVICE_NAME=""
MASTER_SERVER_ADDRESS=""
HOSTNAME_ARG=""
OUTPUT_FILE=""

usage() {
    cat <<EOF
Usage: $0 -l <local_hostname> -m <metadata_server> -p <protocol> -d <device_name_list> -s <master_server_address> -h <hostname> [-o <output_file>]

Arguments:
    -l    Local hostname (e.g., 10.239.44.122)
    -m    Metadata server address (e.g., etcd://10.239.44.122:2379 or 10.239.44.122:2379)
    -p    Protocol (e.g., rdma)
    -d    Device name list, comma-separated (e.g., mlx5_0,mlx5_0,mlx5_0,mlx5_0)
          Or use pattern expansion: [mlx5_0]*16 or [mlx5_0,mlx5_1]*8
    -s    Master server address (e.g., 10.239.44.122:50001)
    -h    Hostname for config file naming (e.g., OAM_1)
    -o    Output file path (optional, defaults to mooncake_<hostname>.json)

Examples:
    $0 -l 10.239.44.122 -m 10.239.44.122:2379 -p rdma -d mlx5_0,mlx5_0,mlx5_0,mlx5_0 -s 10.239.44.122:50001 -h OAM_1
    $0 -l 10.239.44.122 -m 10.239.44.122:2379 -p rdma -d [mlx5_0]*16 -s 10.239.44.122:50001 -h OAM_1
    $0 -l 10.239.44.122 -m 10.239.44.122:2379 -p rdma -d [mlx5_0,mlx5_1]*8 -s 10.239.44.122:50001 -h OAM_1
EOF
    exit 1
}

# Expand pattern syntax: [pattern]*N
# Examples:
#   [mlx5_0]*16 -> mlx5_0,mlx5_0,... (16 times)
#   [mlx5_0,mlx5_1]*8 -> mlx5_0,mlx5_1,mlx5_0,mlx5_1,... (8 repetitions of the pattern)
expand_device_pattern() {
    local input="$1"
    
    # Check if input matches pattern [something]*N
    if [[ "$input" =~ ^\[(.+)\]\*([0-9]+)$ ]]; then
        local pattern="${BASH_REMATCH[1]}"
        local count="${BASH_REMATCH[2]}"
        
        # Build the expanded list
        local result=""
        for ((i=0; i<count; i++)); do
            if [ $i -gt 0 ]; then
                result+=","
            fi
            result+="$pattern"
        done
        
        echo "$result"
    else
        # No pattern expansion needed, return as-is
        echo "$input"
    fi
}

# Parse arguments
while getopts "l:m:p:d:s:h:o:" opt; do
    case $opt in
        l)
            LOCAL_HOSTNAME="$OPTARG"
            ;;
        m)
            METADATA_SERVER="$OPTARG"
            ;;
        p)
            PROTOCOL="$OPTARG"
            ;;
        d)
            DEVICE_NAME="$OPTARG"
            ;;
        s)
            MASTER_SERVER_ADDRESS="$OPTARG"
            ;;
        h)
            HOSTNAME_ARG="$OPTARG"
            ;;
        o)
            OUTPUT_FILE="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$LOCAL_HOSTNAME" ] || [ -z "$METADATA_SERVER" ] || [ -z "$PROTOCOL" ] || [ -z "$DEVICE_NAME" ] || [ -z "$MASTER_SERVER_ADDRESS" ] || [ -z "$HOSTNAME_ARG" ]; then
    echo "Error: Missing required arguments" >&2
    usage
fi

# Set default output file if not provided
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="mooncake_${HOSTNAME_ARG}.json"
fi

# Expand device name pattern if needed (e.g., [mlx5_0]*16 -> mlx5_0,mlx5_0,...)
DEVICE_NAME=$(expand_device_pattern "$DEVICE_NAME")

# Convert device name list to JSON array
# Split by comma and create JSON array format
IFS=',' read -ra DEVICES <<< "$DEVICE_NAME"
DEVICE_JSON="["
for i in "${!DEVICES[@]}"; do
    if [ $i -gt 0 ]; then
        DEVICE_JSON+=","
    fi
    # Trim whitespace and add quotes
    device=$(echo "${DEVICES[$i]}" | xargs)
    DEVICE_JSON+="\"$device\""
done
DEVICE_JSON+="]"

# Ensure metadata_server has etcd:// prefix if it doesn't already have a protocol
if [[ ! "$METADATA_SERVER" =~ ^[a-zA-Z]+:// ]]; then
    METADATA_SERVER="etcd://$METADATA_SERVER"
fi

# Generate JSON config file
cat > "$OUTPUT_FILE" <<EOF
{
    "local_hostname": "$LOCAL_HOSTNAME",
    "metadata_server": "$METADATA_SERVER",
    "protocol": "$PROTOCOL",
    "device_name": $DEVICE_JSON,
    "master_server_address": "$MASTER_SERVER_ADDRESS"
}
EOF

echo "Mooncake config file generated: $OUTPUT_FILE"
cat "$OUTPUT_FILE"


