
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

code_path="../pd_xpyd"
delay_minutes=0 # Set your delay in minutes here

# Default values
input_len=3500
concurrency=16

while getopts "m:i:e:r:l:c:d:" opt; do
  case "$opt" in
    m ) model_path="$OPTARG" ;;
    i ) ip="$OPTARG" ;;
    e ) env_file="$OPTARG" ;;
    r ) repeat_times="$OPTARG" ;;
    l ) input_len="$OPTARG" ;;
    c ) concurrency="$OPTARG" ;;
    d ) delay_minutes="$OPTARG" ;;
    * ) echo "Usage: $0 [-m model_path] [-i ip] [-e env_file] [-r repeat_times] [-l input_len] [-c concurrency] [-d delay_minutes]"; exit 1 ;;
  esac
done

shift $((OPTIND-1)) || true

delay_seconds=$((delay_minutes * 60))

start_time=$(date +%s)
end_time=$((start_time + delay_seconds))

while true; do
  now=$(date +%s)
  remaining=$((end_time - now))
  if [ $remaining -le 0 ]; then
    break
  fi
  min=$((remaining / 60))
  sec=$((remaining % 60))
  # Print remaining time in mm:ss format, overwriting the previous line
  printf "\rDelay in progress: %02d:%02d left..." "$min" "$sec"
  sleep 1
done
# Print a newline after the scroll-back timer
echo

model_path="${model_path:-/host/mnt/kefei/HF_Models/DeepSeek-R1-Gaudi3/}"
ip="${ip:-10.112.242.93}"
env_file="${env_file:-./env_2p4d_sedv+.sh}"
repeat_times="${repeat_times:-3}"

echo "model_path:$model_path"
echo "ip:$ip"
echo "env_file:$env_file"
echo "repeat_times:$repeat_times"
echo "input_len:$input_len"
echo "concurrency:$concurrency"

echo "========================================================"
echo "========  INPUT LEN: $input_len | CONCURRENCY: $concurrency  ========"
echo "========================================================"

for ((round_idx=1; round_idx<=repeat_times; round_idx++)); do
  suffix="th"
  mod10=$((round_idx % 10))
  mod100=$((round_idx % 100))
  if [ "$mod10" -eq 1 ] && [ "$mod100" -ne 11 ]; then
    suffix="st"
  elif [ "$mod10" -eq 2 ] && [ "$mod100" -ne 12 ]; then
    suffix="nd"
  elif [ "$mod10" -eq 3 ] && [ "$mod100" -ne 13 ]; then
    suffix="rd"
  fi
  pushd $code_path
  bash ./PXY.sh -b -d -r -t $((concurrency - 1)) -e "$env_file"
  popd
  echo "########################################################"
  printf "####################   %2d%s ROUND   #####################\n" "$round_idx" "$suffix"
  echo "########################################################"
  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len "$input_len" --sonnet-output-len 1000 --sonnet-prefix-len 100 --host $ip --port 8868 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --model $model_path --num-prompt 1
done

