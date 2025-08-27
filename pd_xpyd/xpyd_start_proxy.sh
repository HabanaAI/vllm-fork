#set +x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_env.sh

if [ -z "$1" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, advanced/basic/benchmark proxy mode"
    echo "run with default mode P=1, D=2, TP size=1, advanced"
    P_INSTANCE_NUMBER=1
    D_INSTANCE_NUMBER=2
    NUM_DECODE=8
else
    P_INSTANCE_NUMBER=$1
fi

if [ -z "$2" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, advanced/basic/benchmark proxy mode"
    echo "run with P=$P_INSTANCE_NUMBER, D=2, TP size=1, advanced"
    D_INSTANCE_NUMBER=2
    TP_SIZE=1
    NUM_DECODE=$((8 / TP_SIZE))
else
    D_INSTANCE_NUMBER=$2
fi

if [ -z "$3" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, advanced/basic/benchmark proxy mode"
    echo "run with P=$P_INSTANCE_NUMBER, D=$D_INSTANCE_NUMBER, TP size=1, advanced"
    TP_SIZE=1
    NUM_DECODE=$((8 / TP_SIZE))
else
    TP_SIZE=$3
    NUM_DECODE=$((8 / TP_SIZE))
fi

PROXY_MODE=0

if [ "$4" == "benchmark" ]; then
    PROXY_MODE=2
    echo " Benchmark mode enabled"
fi

if [ "$4" == "basic" ]; then
    PROXY_MODE=1
    echo " Basic mode enabled"
fi

# For backward compatibility.....

if [ "$5" == "benchmark" ]; then
    PROXY_MODE=2
    echo " Benchmark mode enabled"
fi

if [ "$5" == "basic" ]; then
    PROXY_MODE=1
    echo " Basic mode enabled"
fi

#For OAM
DECODE_IPS=("10.239.129.81" "10.239.129.165" "10.239.129.67" "10.239.129.21")
#For PCIE
# DECODE_IPS=("10.112.110.161" "10.112.110.148")

DBASE_PORT=8200
DECODE_ARGS=""

for ((i=0; i<$NUM_DECODE; i++)); do
    PORT=$((DBASE_PORT + i))
    for ((j=0; j<D_INSTANCE_NUMBER; j++)); do
	IP=${DECODE_IPS[$j]}
	DECODE_ARGS="$DECODE_ARGS ${IP}:${PORT}"
    done
done

#For OAM
PREFILL_IPS=("10.239.129.9" "10.239.129.67" "10.239.129.21" "10.239.128.165" "10.239.128.244" "10.239.128.153")
#For PCIE
# PREFILL_IPS=("10.112.110.157")

PBASE_PORT=8100
PREFILL_ARGS=""

PORT=$PBASE_PORT
for ((i=0; i<P_INSTANCE_NUMBER; i++)); do
    IP=${PREFILL_IPS[$i]}
    PREFILL_ARGS="$PREFILL_ARGS ${IP}:${PORT}"
done

if [ "$PROXY_MODE" == 2 ]; then
    CMD="python3 ./examples/online_serving/disagg_examples/disagg_proxy_benchmark.py \
        --model $model_path \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port 8868 \
        --repeat_p_request 1 \
        --repeat_d_times 639 \
        --benchmark_mode"

elif [ "$PROXY_MODE" == 0 ]; then
    CMD="python3 ./examples/online_serving/disagg_examples/disagg_proxy_advanced.py \
        --model $model_path \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port 8868"
else
    CMD="python3 ./examples/online_serving/disagg_examples/disagg_proxy_basic.py \
        --model $model_path \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port 8868"
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="xpyd_logs"
mkdir -p "$log_dir"
log_file="$log_dir/ProxyServer_${timestamp}.log"

CMD="$CMD 2>&1 | tee $log_file"

echo "Running: $CMD"
eval $CMD

