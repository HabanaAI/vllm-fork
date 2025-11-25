delay_minutes=0 # Set your delay in minutes here

# Default values
input_len=3500
concurrency=64
request_rate="inf"
output_len=1000

while getopts "m:i:e:r:l:c:R:o:d:" opt; do
  case "$opt" in
    m ) model_path="$OPTARG" ;;
    i ) host_ip="$OPTARG" ;;
    e ) env_file="$OPTARG" ;;
    r ) repeat_times="$OPTARG" ;;
    l ) input_len="$OPTARG" ;;
    c ) concurrency="$OPTARG" ;;
    R ) request_rate="$OPTARG" ;;
    o ) output_len="$OPTARG" ;;
    d ) delay_minutes="$OPTARG" ;;
    * ) echo "Usage: $0 [-m model_path] [-i host_ip] [-e env_file] [-r repeat_times] [-l input_len] [-c concurrency] [-R request_rate] [-o output_len] [-d delay_minutes]"; exit 1 ;;
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

host_ip="${host_ip:-localhost}"
model_path="${model_path:-/host/mnt/kefei/HF_Models/DeepSeek-R1-Gaudi3/}"
env_file="${env_file:-./env_2p4d_sedv+.sh}"
repeat_times="${repeat_times:-1}"

if [ -z "$request_rate" ]; then
  request_rate="inf"
fi
if [ -z "$output_len" ]; then
  output_len=1000
fi

num_prompt=$((concurrency * 10))

echo "model_path:$model_path"
echo "host_ip:$host_ip"
echo "env_file:$env_file"
echo "repeat_times:$repeat_times"
echo "input_len:$input_len"
echo "concurrency:$concurrency"
echo "request_rate:$request_rate"
echo "output_len:$output_len"
echo "num_prompt:$num_prompt"

pushd ../pd_xpyd
bash ./PXY.sh -r -d -e $env_file
popd

echo "=========================================================================================="
echo "========  INPUT LEN: $input_len | CONCURRENCY: $concurrency | REQ_RATE: $request_rate | NUM_PROMPT: $num_prompt  ========"
echo "=========================================================================================="

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
  echo "########################################################"
  printf "####################   %2d%s ROUND   #####################\n" "$round_idx" "$suffix"
  echo "########################################################"
  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len "$input_len" --sonnet-output-len "$output_len" --sonnet-prefix-len 100 --host "$host_ip" --port 8868 --max-concurrency "$concurrency" --request-rate "$request_rate" --ignore-eos --model "$model_path" --num-prompt "$num_prompt"
done


