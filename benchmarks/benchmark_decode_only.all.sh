concurrency_list=(64 96 144 288 512 544 576 624 1024)

for concurrency in "${concurrency_list[@]}"; do
  echo "========================================================"
  echo "==============  CURRENT CONCURRENCY: $concurrency  =============="
  echo "========================================================"
  pushd /host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd/
  bash ./PXY.sh -b -d -r -t $((concurrency - 1))
  popd
  echo "########################################################"
  echo "####################   1st ROUND   #####################"
  echo "########################################################"
  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 1
  pushd /host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd/
  bash ./PXY.sh -b -d -r -t $((concurrency - 1))
  popd
  echo "########################################################"
  echo "####################   2nd ROUND   #####################"
  echo "########################################################"
  python benchmark_serving.py --backend vllm --dataset-name sonnet --dataset-path sonnet.txt --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --host 10.112.242.154 --port 8868 --max-concurrency "$concurrency" --request-rate inf --ignore-eos --model /host/mnt/disk001/HF_Models/DeepSeek-R1 --num-prompt 1
done


