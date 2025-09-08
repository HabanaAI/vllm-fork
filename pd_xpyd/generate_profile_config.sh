#!/bin/bash

# Default values
INFLIGHT_REQUESTS=""
BLOCK_SIZE=""
STEPS=""
RANKS=""
TARGET=""
CONFIG_DIR=""
WARMUP_STEPS=""

# Function to display usage
usage() {
    echo "Usage: $0 -i inflight_requests -b block_size -s steps -r ranks -t target -p config_directory_path [-w warmup_steps]" >&2
    echo "" >&2
    echo "Arguments:" >&2
    echo "  -i: Number of inflight requests (required)" >&2
    echo "  -b: Block size (required)" >&2
    echo "  -s: Number of steps (required)" >&2
    echo "  -r: Ranks (comma-separated list, e.g., '0,1,2' or 'all') (required)" >&2
    echo "  -t: Target - must be one of: P, D, BOTH (required)" >&2
    echo "  -p: Directory path where profile_config.json will be created (required)" >&2
    echo "  -w: Warmup steps - defer profiling start by this many steps (optional, default: 0)" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 -i 10 -b 1024 -s 100 -r '0,1,2' -t 'P' -p ./configs" >&2
    echo "  $0 -i 5 -b 512 -s 50 -r 'all' -t 'BOTH' -p /tmp -w 3" >&2
    echo "  $0 -i 8 -b 2048 -s 200 -r '0,1' -t 'D' -p /path/to/configs -w 1" >&2
    exit 1
}

# Parse command line arguments
while getopts "i:b:s:r:t:p:w:h" opt; do
    case $opt in
        i) INFLIGHT_REQUESTS="$OPTARG" ;;
        b) BLOCK_SIZE="$OPTARG" ;;
        s) STEPS="$OPTARG" ;;
        r) RANKS="$OPTARG" ;;
        t) TARGET="$OPTARG" ;;
        p) CONFIG_DIR="$OPTARG" ;;
        w) WARMUP_STEPS="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check if all required arguments are provided
if [[ -z "$INFLIGHT_REQUESTS" || -z "$BLOCK_SIZE" || -z "$STEPS" || -z "$RANKS" || -z "$TARGET" || -z "$CONFIG_DIR" ]]; then
    echo "Error: All arguments are required." >&2
    echo "" >&2
    usage
fi

# Set default warmup steps if not provided
if [[ -z "$WARMUP_STEPS" ]]; then
    WARMUP_STEPS=0
fi

# Validate numeric inputs
if ! [[ "$INFLIGHT_REQUESTS" =~ ^[0-9]+$ ]]; then
    echo "Error: Inflight requests (-i) must be a positive integer." >&2
    exit 1
fi

if ! [[ "$BLOCK_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: Block size (-b) must be a positive integer." >&2
    exit 1
fi

if ! [[ "$STEPS" =~ ^[0-9]+$ ]]; then
    echo "Error: Steps (-s) must be a positive integer." >&2
    exit 1
fi

if ! [[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
    echo "Error: Warmup steps (-w) must be a non-negative integer." >&2
    exit 1
fi

# Validate target
if [[ "$TARGET" != "P" && "$TARGET" != "D" && "$TARGET" != "BOTH" ]]; then
    echo "Error: Target (-t) must be one of: P, D, BOTH" >&2
    exit 1
fi

# Function to convert comma-separated string to JSON array
csv_to_json_array() {
    local input="$1"
    if [[ "$input" == "all" ]]; then
        echo '"all"'
    else
        # Split by comma and create JSON array
        echo "$input" | sed 's/,/", "/g' | sed 's/^/["/' | sed 's/$/"]/'
    fi
}

# Convert ranks to JSON array
RANKS_JSON=$(csv_to_json_array "$RANKS")

# Create the directory for the config file if it doesn't exist
if [[ ! -d "$CONFIG_DIR" ]]; then
    mkdir -p "$CONFIG_DIR"
    echo "Created directory: $CONFIG_DIR"
fi

# Set the full path to the config file
CONFIG_PATH="$CONFIG_DIR/profile_config.json"

# Generate the profile_config.json file
cat > "$CONFIG_PATH" << EOF
{
    "inflight": $INFLIGHT_REQUESTS,
    "block_size": $BLOCK_SIZE,
    "steps": $STEPS,
    "profile_ranks": $RANKS_JSON,
    "profile_targets": "$TARGET",
    "warmup": $WARMUP_STEPS
}
EOF

# Check if file was created successfully
if [[ $? -eq 0 && -f "$CONFIG_PATH" ]]; then
    echo "✅ Profile config generated successfully!"
    echo "📁 Output file: $CONFIG_PATH"
    echo ""
    echo "📋 Configuration:"
    echo "   • Inflight requests: $INFLIGHT_REQUESTS"
    echo "   • Block size: $BLOCK_SIZE"
    echo "   • Steps: $STEPS"
    echo "   • Ranks: $RANKS"
    echo "   • Target: $TARGET"
    echo "   • Warmup steps: $WARMUP_STEPS"
    echo ""
    echo "📄 Generated JSON content:"
    cat "$CONFIG_PATH" | sed 's/^/   /'
else
    echo "❌ Error: Failed to create profile config file at $CONFIG_PATH" >&2
    exit 1
fi
