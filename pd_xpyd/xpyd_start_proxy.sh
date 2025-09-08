#set +x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_env.sh

if [ -z "$1" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D), repeat_d_times"
    echo "run with default mode P=1, D=2, TP size=1, false, 127"
    P_INSTANCE_NUMBER=1
    D_INSTANCE_NUMBER=2
    NUM_DECODE=8
    FIRST_TOKEN_FROM_P=false
    REPEAT_D_TIMES=127
else
    P_INSTANCE_NUMBER=$1
fi

if [ -z "$2" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D), repeat_d_times"
    echo "run with P=$P_INSTANCE_NUMBER, D=2, TP size=1, false, 127"
    D_INSTANCE_NUMBER=2
    TP_SIZE=1
    NUM_DECODE=$((8 / TP_SIZE))
    FIRST_TOKEN_FROM_P=false
    REPEAT_D_TIMES=127
else
    D_INSTANCE_NUMBER=$2
fi

if [ -z "$3" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D), repeat_d_times"
    echo "run with P=$P_INSTANCE_NUMBER, D=$D_INSTANCE_NUMBER, TP size=1, false, 127"
    TP_SIZE=1
    NUM_DECODE=$((8 / TP_SIZE))
    FIRST_TOKEN_FROM_P=false
    REPEAT_D_TIMES=127
else
    TP_SIZE=$3
    NUM_DECODE=$((8 / TP_SIZE))
fi

if [ -z "$4" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D), repeat_d_times"
    echo "run with P=$P_INSTANCE_NUMBER, D=$D_INSTANCE_NUMBER, TP size=$TP_SIZE, false, 127"
    FIRST_TOKEN_FROM_P=false
    REPEAT_D_TIMES=127
else
    FIRST_TOKEN_FROM_P=$4
fi

if [ -z "$5" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D), repeat_d_times"
    echo "run with P=$P_INSTANCE_NUMBER, D=$D_INSTANCE_NUMBER, TP size=$TP_SIZE, $FIRST_TOKEN_FROM_P, 127"
    REPEAT_D_TIMES=127
else
    REPEAT_D_TIMES=$5
fi

BENCHMARK_MODE=0

if [ "$6" == "benchmark" ]; then
    BENCHMARK_MODE=1
    echo " Benchmark mode enabled"
fi

#For OAM
#DECODE_IPS=("10.112.242.154" "10.112.242.24" "10.239.129.67" "10.239.129.21")
DECODE_IPS=("10.112.242.153" "10.112.242.24" "10.239.129.67" "10.239.129.21")
#For PCIE
# DECODE_IPS=("10.112.110.161" "10.112.110.148")

DBASE_PORT=8200
DECODE_ARGS=""
echo $NUM_DECODE
for ((i=0; i<$NUM_DECODE; i++)); do
    PORT=$((DBASE_PORT + i))
    for ((j=0; j<D_INSTANCE_NUMBER; j++)); do
	IP=${DECODE_IPS[$j]}
	DECODE_ARGS="$DECODE_ARGS ${IP}:${PORT}"
    done
done

#For OAM
#PREFILL_IPS=("10.112.242.153" "10.239.129.67" "10.239.129.21" "10.239.128.165" "10.239.128.244" "10.239.128.153")
PREFILL_IPS=("10.112.242.154" "10.239.129.67" "10.239.129.21" "10.239.128.165" "10.239.128.244" "10.239.128.153")
#For PCIE
# PREFILL_IPS=("10.112.110.157")

PBASE_PORT=8100
PREFILL_ARGS=""

PORT=$PBASE_PORT
for ((i=0; i<P_INSTANCE_NUMBER; i++)); do
    IP=${PREFILL_IPS[$i]}
    PREFILL_ARGS="$PREFILL_ARGS ${IP}:${PORT}"
done

if [ "$BENCHMARK_MODE" == "1" ]; then
    CMD="python3 ../examples/online_serving/disagg_examples/disagg_proxy_demo_benchmark.py \
        --model $model_path \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port 8868 \
        --repeat_p_request 1 \
        --repeat_d_times $REPEAT_D_TIMES \
        --benchmark_mode"

else

    CMD="python3 ../examples/online_serving/disagg_examples/disagg_proxy_demo.py \
        --model $model_path \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port 8868"
fi

if [ "$FIRST_TOKEN_FROM_P" = "true" ]; then
    CMD="$CMD --generator_on_p_node"
fi

echo "Running: $CMD"
eval $CMD

