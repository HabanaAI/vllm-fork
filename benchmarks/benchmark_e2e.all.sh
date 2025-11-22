## Define 2D cases as tuples: "<input_len> <concurrency> <request_rate> <output_len>"
## Adjust or extend as needed.
#cases=(
#  "3500 64 3 1"
#  "3500 64 2.7 1"
#  "3500 64 2.5 1"
#  "3500 64 2.3 1"
#  "3500 64 inf 1"
#)
cases=(
  "3500 1 1.2 1000"
  "3500 16 1.2 1000"
  "3500 32 1.2 1000"
  "3500 64 1.2 1000"
  "3500 96 1.6 1000"
  "3500 112 1.8 1000"
  "3500 128 2.5 1000"
  "3500 144 2.5 1000"
  "3500 160 2.5 1000"
  "3500 176 2.3 1000"
  "3500 192 4.5 1000"
  "3500 208 4.5 1000"
  "3500 224 4.5 1000"
  "3500 240 4.5 1000"
  "3500 256 4.5 1000"
  "3500 1 inf 1000"
  "3500 16 inf 1000"
  "3500 32 inf 1000"
  "3500 64 inf 1000"
  "3500 96 inf 1000"
  "3500 112 inf 1000"
  "3500 128 inf 1000"
  "3500 144 inf 1000"
  "3500 160 inf 1000"
  "3500 176 inf 1000"
  "3500 192 inf 1000"
  "3500 208 inf 1000"
  "3500 224 inf 1000"
  "3500 240 inf 1000"
  "3500 256 inf 1000"
  "3500 272 inf 1000"
  "3500 288 inf 1000"
  "3500 512 inf 1000"
  "3500 544 inf 1000"
  "3500 576 inf 1000"
  "3500 624 inf 1000"
  "3500 1024 inf 1000"
  "2000 96 inf 1000"
  "2000 224 inf 1000"
  "2000 384 inf 1000"
  "2000 640 inf 1000"
  "2000 672 inf 1000"
  "2000 768 inf 1000"
  "2000 800 inf 1000"
  "2000 1280 inf 1000"
)
#model_path="/host/mnt/disk001/HF_Models/DeepSeek-R1"
delay_minutes=0 # Set your delay in minutes here

while getopts "m:i:e:r:" opt; do
  case "$opt" in
    m ) model_path="$OPTARG" ;;
    i ) host_ip="$OPTARG" ;;
    e ) env_file="$OPTARG" ;;
    r ) repeat_times="$OPTARG" ;;
    * ) echo "Usage: $0 [-m model_path] [-i ip] [-e env_file] [-r repeat_times]"; exit 1 ;;
  esac
done
shift $((OPTIND-1)) || true

delay_seconds=$((delay_minutes * 60))
interval=600  # 10 minutes in seconds

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
#model_path="/host/mnt/disk002/HF_Models/DeepSeek-R1-Gaudi3/"
model_path="${model_path:-/host/mnt/kefei/HF_Models/DeepSeek-R1-Gaudi3/}"
env_file="${env_file:-./env_2p4d_sedv+.sh}"
repeat_times="${repeat_times:-1}"
echo "model_path:$model_path"
echo "host_ip:$host_ip"
echo "env_file:$env_file"
echo "repeat_times:$repeat_times"

pushd ../pd_xpyd
bash ./PXY.sh -r -d -e $env_file
popd
for case in "${cases[@]}"; do
  read -r input_len concurrency request_rate output_len <<< "$case"
  if [ -z "$request_rate" ]; then
    request_rate="inf"
  fi
  if [ -z "$output_len" ]; then
    output_len=1000
  fi
  num_prompt=$((concurrency * 10))
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
done

