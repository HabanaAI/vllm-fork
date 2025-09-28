#set +x
BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_env.sh
#<<<<<<< HEAD

#if [ -z "$1" ]; then
#    echo "please input P instance number, D instance number, TP size of D instance, advanced/basic/benchmark proxy mode"
#    echo "run with default mode P=1, D=2, TP size=1, advanced"
#    P_INSTANCE_NUMBER=1
#    D_INSTANCE_NUMBER=2
#    NUM_DECODE=8
#=======

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo $1,$2,$3,$4,$5,$6

if [ -z "$1" ]; then
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D), repeat_d_times"
    echo "run with default mode P=1, D=2, TP size=1, false, 127"
    P_INSTANCE_NUMBER=1
    D_INSTANCE_NUMBER=2
    NUM_DECODE=8
    FIRST_TOKEN_FROM_P=false
    REPEAT_D_TIMES=127
#>>>>>>> kf-fork/deepseek_r1_ww33_kf
else
    P_INSTANCE_NUMBER=$1
fi

if [ -z "$2" ]; then
#<<<<<<< HEAD
#    echo "please input P instance number, D instance number, TP size of D instance, advanced/basic/benchmark proxy mode"
#    echo "run with P=$P_INSTANCE_NUMBER, D=2, TP size=1, advanced"
#    D_INSTANCE_NUMBER=2
#    TP_SIZE=1
#    NUM_DECODE=$((8 / TP_SIZE))
#=======
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D), repeat_d_times"
    echo "run with P=$P_INSTANCE_NUMBER, D=2, TP size=1, false, 127"
    D_INSTANCE_NUMBER=2
    TP_SIZE=1
    NUM_DECODE=$((8 / TP_SIZE))
    FIRST_TOKEN_FROM_P=false
    REPEAT_D_TIMES=127
#>>>>>>> kf-fork/deepseek_r1_ww33_kf
else
    D_INSTANCE_NUMBER=$2
fi

if [ -z "$3" ]; then
#<<<<<<< HEAD
#    echo "please input P instance number, D instance number, TP size of D instance, advanced/basic/benchmark proxy mode"
#    echo "run with P=$P_INSTANCE_NUMBER, D=$D_INSTANCE_NUMBER, TP size=1, advanced"
#    TP_SIZE=1
#    NUM_DECODE=$((8 / TP_SIZE))
#=======
    echo "please input P instance number, D instance number, TP size of D instance, true or false (true for first token from P, default from D), repeat_d_times"
    echo "run with P=$P_INSTANCE_NUMBER, D=$D_INSTANCE_NUMBER, TP size=1, false, 127"
    TP_SIZE=1
    NUM_DECODE=$((8 / TP_SIZE))
    FIRST_TOKEN_FROM_P=false
    REPEAT_D_TIMES=127
#>>>>>>> kf-fork/deepseek_r1_ww33_kf
else
    TP_SIZE=$3
    NUM_DECODE=$((8 / TP_SIZE))
fi

#<<<<<<< HEAD
#PROXY_MODE=0
#
#if [ "$4" == "benchmark" ]; then
#    PROXY_MODE=2
#    echo " Benchmark mode enabled"
#fi
#
#if [ "$4" == "basic" ]; then
#    PROXY_MODE=1
#    echo " Basic mode enabled"
#fi
#
## For backward compatibility.....
#
#if [ "$5" == "benchmark" ]; then
#    PROXY_MODE=2
#=======
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


if [ "$6" == "benchmark" ]; then
    PROXY_MODE=2
#>>>>>>> kf-fork/deepseek_r1_ww33_kf
    echo " Benchmark mode enabled"
fi

if [ "$5" == "basic" ]; then
    PROXY_MODE=1
    echo " Basic mode enabled"
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

if [ "$PROXY_MODE" == 2 ]; then
    CMD="python3 ../examples/online_serving/disagg_examples/disagg_proxy_benchmark.py \
        --model $model_path \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port 8868 \
        --repeat_p_request 1 \
        --repeat_d_times $REPEAT_D_TIMES \
        --benchmark_mode"

elif [ "$PROXY_MODE" == 0 ]; then
    CMD="python3 ../examples/online_serving/disagg_examples/disagg_proxy_advanced.py \
        --model $model_path \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port 8868"
else
    CMD="python3 ../examples/online_serving/disagg_examples/disagg_proxy_basic.py \
        --model $model_path \
        --prefill $PREFILL_ARGS \
        --decode $DECODE_ARGS \
        --port 8868"
fi

# Check if XPYD_LOG is defined and non-empty
if [ -n "$XPYD_LOG" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="$XPYD_LOG/ProxyServer_${timestamp}.log"

    CMD="$CMD 2>&1 | tee $log_file"
fi

echo "Running: $CMD"
eval $CMD

