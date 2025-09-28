
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

## Define 2D cases as pairs: "<input_len> <concurrency>"
## Adjust or extend as needed.
cases=(
  #"3500 64"
  #"3500 96"
  #"3500 112"
  #"3500 128"
  #"3500 144"
  #"3500 160"
  #"3500 176"
  #"3500 192"
  #"3500 208"
  #"3500 224"
  #"3500 240"
  "3500 256"
  #"3500 272"
  #"3500 288"
  #"3500 512"
  #"3500 544"
  #"3500 576"
  #"3500 624"
  #"3500 1024"
  #"2000 96"
  #"2000 224"
  #"2000 384"
  #"2000 640"
  #"2000 672"
  #"2000 768"
  #"2000 800"
  #"2000 1280"
)
model_path="/host/mnt/disk001/HF_Models/DeepSeek-R1"
#model_path="/host/mnt/disk002/HF_Models/DeepSeek-R1-Gaudi3/"
for case in "${cases[@]}"; do
  read -r input_len concurrency <<< "$case"
  echo "========================================================"
  echo "========  INPUT LEN: $input_len | CONCURRENCY: $concurrency  ========"
  echo "========================================================"
  pushd /host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd/
  bash ./PXY.sh -b -d -r -t $((concurrency - 1))
  popd
  echo "########################################################"
  echo "####################   1st ROUND   #####################"
  echo "########################################################"
  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len "$input_len" --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --model $model_path --num-prompt 1
  pushd /host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd/
  bash ./PXY.sh -b -d -r -t $((concurrency - 1))
  popd
  echo "########################################################"
  echo "####################   2nd ROUND   #####################"
  echo "########################################################"
  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len "$input_len" --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --model $model_path --num-prompt 1
  pushd /host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd/
  bash ./PXY.sh -b -d -r -t $((concurrency - 1))
  popd
  echo "########################################################"
  echo "####################   3rd ROUND   #####################"
  echo "########################################################"
  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len "$input_len" --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --model $model_path --num-prompt 1
done


