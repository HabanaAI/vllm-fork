#!/bin/bash
# Parse command line arguments
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
unset http_porxy https_proxy HTTP_PROXY HTTPS_PROXY

BENCHMARK_MODE=false
FIRST_TOKEN_FROM_D=false
KILL_MODE=false
RESTART_MODE=false
REPEAT_D_TIMES=127
ENV_FILE=""

while getopts "bdkrt:e:" opt; do
    case $opt in
        b) BENCHMARK_MODE=true ;;
        d) FIRST_TOKEN_FROM_D=true ;;
        k) KILL_MODE=true ;;
        r) RESTART_MODE=true ;;
        t) REPEAT_D_TIMES=$OPTARG ;;
        e) ENV_FILE=$OPTARG ;;
        *) echo "Usage: $0 [-b] [-d] [-k] [-r] [-t repeat_d_times] [-e env_file]" >&2
           echo "  -b: Enable benchmark mode" >&2
           echo "  -d: First token from decode node" >&2
           echo "  -k: Kill proxy server processes" >&2
           echo "  -r: Restart proxy server" >&2
           echo "  -t: Set repeat_d_times for benchmark (default: 127)" >&2
           echo "  -e: Path to environment file defining ROLE_IP entries" >&2
           echo "" >&2
           echo "Example: $0 -b -t 50 -e env_G16_G15_G13.sh" >&2
           exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [ -n "$ENV_FILE" ]; then
    if [[ "$ENV_FILE" != /* ]]; then
        ENV_FILE="$BASH_DIR/$ENV_FILE"
    fi
    if [ ! -f "$ENV_FILE" ]; then
        echo "Environment file $ENV_FILE not found"
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$ENV_FILE"

    PREFILL_LIST=()
    DECODE_LIST=()

    if declare -p ROLE_IP >/dev/null 2>&1; then
        idx=0
        while true; do
            key="P${idx}"
            ip="${ROLE_IP[$key]}"
            [[ -z "$ip" ]] && break
            PREFILL_LIST+=("$ip")
            ((idx++))
        done

        idx=0
        while true; do
            key="D${idx}"
            ip="${ROLE_IP[$key]}"
            [[ -z "$ip" ]] && break
            DECODE_LIST+=("$ip")
            ((idx++))
        done
    else
        echo "Warning: ROLE_IP array not defined in $ENV_FILE; defaults will be used by xpyd_start_proxy.sh"
    fi

    if [ ${#PREFILL_LIST[@]} -gt 0 ]; then
        export XPYD_PREFILL_IPS="${PREFILL_LIST[*]}"
    fi
    if [ ${#DECODE_LIST[@]} -gt 0 ]; then
        export XPYD_DECODE_IPS="${DECODE_LIST[*]}"
    fi
fi
echo "XPYD_PREFILL_IPS: ${XPYD_PREFILL_IPS:-"(none)"}"
echo "XPYD_DECODE_IPS: ${XPYD_DECODE_IPS:-"(none)"}"

# Handle kill mode - kill proxy server and exit
if [ "$KILL_MODE" = true ]; then
    echo "-------------------------------------------------------------------"
    echo "Killing proxy server processes..."
    echo "-------------------------------------------------------------------"
    pkill -KILL -f disagg_proxy
    pkill -KILL -f xpyd_start_proxy.sh
    echo "Proxy server processes killed."
    exit 0
fi

# Handle restart mode - kill and restart proxy server
if [ "$RESTART_MODE" = true ]; then
    echo "-------------------------------------------------------------------"
    echo "Restarting proxy server - killing existing processes..."
    echo "-------------------------------------------------------------------"
    pkill -KILL -f disagg_proxy
    pkill -KILL -f xpyd_start_proxy.sh
    echo "Existing proxy server processes killed."
fi

# Build the command arguments

P_INSTANCE_NUMBER=${USR_PREFILL_INSTANCE_NUM:-0}
if [ "$P_INSTANCE_NUMBER" -le 0 ]; then
    if [ -n "$XPYD_PREFILL_IPS" ]; then
        read -r -a __prefill_arr <<< "$XPYD_PREFILL_IPS"
        P_INSTANCE_NUMBER=${#__prefill_arr[@]}
        unset __prefill_arr
    elif declare -p ROLE_IP >/dev/null 2>&1; then
        P_INSTANCE_NUMBER=0
        while true; do
            key="P${P_INSTANCE_NUMBER}"
            ip="${ROLE_IP[$key]}"
            [[ -z "$ip" ]] && break
            ((P_INSTANCE_NUMBER++))
        done
    else
        P_INSTANCE_NUMBER=1
    fi
fi

D_INSTANCE_NUMBER=${USR_DECODE_INSTANCE_NUM:-0}
if [ "$D_INSTANCE_NUMBER" -le 0 ]; then
    if [ -n "$XPYD_DECODE_IPS" ]; then
        read -r -a __decode_arr <<< "$XPYD_DECODE_IPS"
        D_INSTANCE_NUMBER=${#__decode_arr[@]}
        unset __decode_arr
    elif declare -p ROLE_IP >/dev/null 2>&1; then
        D_INSTANCE_NUMBER=0
        while true; do
            key="D${D_INSTANCE_NUMBER}"
            ip="${ROLE_IP[$key]}"
            [[ -z "$ip" ]] && break
            ((D_INSTANCE_NUMBER++))
        done
    else
        D_INSTANCE_NUMBER=2
    fi
fi

TP_SIZE=${USR_PROXY_TP_SIZE:-1}


#CMD_ARGS="$P_INSTANCE_NUMBER $D_INSTANCE_NUMBER $TP_SIZE"
CMD_ARGS="1 $D_INSTANCE_NUMBER $TP_SIZE"

echo CMD_ARGS:$CMD_ARGS

# Set first token source argument (arg 4)
if [ "$FIRST_TOKEN_FROM_D" = true ]; then
    CMD_ARGS="$CMD_ARGS false"
else
    CMD_ARGS="$CMD_ARGS true"
fi

# Set repeat_d_times argument (arg 5)
CMD_ARGS="$CMD_ARGS $REPEAT_D_TIMES"

# Add benchmark argument if benchmark mode is enabled (arg 6)
if [ "$BENCHMARK_MODE" = true ]; then
    CMD_ARGS="$CMD_ARGS benchmark"
fi

mkdir -p pd_test_log

log_file="pd_test_log/proxy.log"

MAX_SIZE=$((50*1024*1024))
if [ -f "$log_file" ]; then
  cur_size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
  if [ "$cur_size" -gt "$MAX_SIZE" ]; then
    ts=$(TZ="Asia/Shanghai" date +"%Y%m%d_%H%M%S")
    mv "$log_file" "${log_file}.${ts}"
    : > "$log_file"
  fi
fi

export CARDS_PER_NODE=${USR_CARDS_PER_NODE:-8}

echo "-------------------------------------------------------------------"
echo "Being brought to background"
echo "Log will be redirect to $log_file"
echo "Benchmark mode: $BENCHMARK_MODE"
echo "First token from d: $FIRST_TOKEN_FROM_D"
echo "Repeat D times: $REPEAT_D_TIMES"
echo "Command: bash xpyd_start_proxy.sh $CMD_ARGS"
echo "..."
echo "-------------------------------------------------------------------"
bash xpyd_start_proxy.sh $CMD_ARGS >> $log_file 2>&1 &
