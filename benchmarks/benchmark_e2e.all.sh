## Define 2D cases as pairs: "<input_len> <concurrency>"
## Adjust or extend as needed.
cases=(
  "3500 64 inf"
  "3500 96 inf"
  "3500 112 inf"
  "3500 128 inf"
  "3500 144 inf"
  "3500 160 5"
  "3500 176 5"
  "3500 192 5"
  "3500 208 5"
  "3500 224 5"
  "3500 240 5"
  "3500 256 5"
  "3500 64 inf"
  "3500 96 inf"
  "3500 112 inf"
  "3500 128 inf"
  "3500 144 inf"
  "3500 160 inf"
  "3500 176 inf"
  "3500 192 inf"
  "3500 208 inf"
  "3500 224 inf"
  "3500 240 inf"
  "3500 256 inf"
  #"3500 272 inf"
  #"3500 288 inf"
  #"3500 512 inf"
  #"3500 544 inf"
  #"3500 576 inf"
  #"3500 624 inf"
  #"3500 1024 inf"
  #"2000 96 inf"
  #"2000 224 inf"
  #"2000 384 inf"
  #"2000 640 inf"
  #"2000 672 inf"
  #"2000 768 inf"
  #"2000 800 inf"
  #"2000 1280 inf"
)
#model_path="/host/mnt/disk001/HF_Models/DeepSeek-R1"
#delay_minutes=180  # Set your delay in minutes here
delay_minutes=0 # Set your delay in minutes here

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

model_path="/host/mnt/disk002/HF_Models/DeepSeek-R1-Gaudi3/"
for case in "${cases[@]}"; do
  read -r input_len concurrency request_rate <<< "$case"
  if [ -z "$request_rate" ]; then
    request_rate="inf"
  fi
  num_prompt=$((concurrency * 10))
  echo "=========================================================================================="
  echo "========  INPUT LEN: $input_len | CONCURRENCY: $concurrency | REQ_RATE: $request_rate | NUM_PROMPT: $num_prompt  ========"
  echo "=========================================================================================="
  
  echo "########################################################"
  echo "####################   1st ROUND   #####################"
  echo "########################################################"
  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len "$input_len" --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency "$concurrency" --request-rate "$request_rate" --ignore-eos --model $model_path --num-prompt "$num_prompt"
  
#  echo "########################################################"
#  echo "####################   2nd ROUND   #####################"
#  echo "########################################################"
#  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len "$input_len" --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --model $model_path --num-prompt "$num_prompt"
  
  #echo "########################################################"
  #echo "####################   3rd ROUND   #####################"
  #echo "########################################################"
  #python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len "$input_len" --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --model $model_path --num-prompt "$num_prompt"
done

