#workdir=/host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd
workdir=/host/mnt/ctrl/disk1/kf/vllm-fork-deepseek_r1/pd_xpyd

# Parse options for P benchmark mode (default: off). Usage: ./1P2D.sh [-b]
P_ARGS=""
while getopts ":b" opt; do
  case $opt in
    b)
      P_ARGS="-b"
      ;;
    \?)
      echo "Unknown option: -$OPTARG" >&2
      ;;
  esac
done

if [ -n "$P_ARGS" ]; then
  echo "Start 1P on current node (benchmark mode)"
else
  echo "Start 1P on current node"
fi
bash ./1P.sh $P_ARGS
sleep 1
echo "[SSH] Start 2D0 on 10.112.242.153 node"
ssh root@10.112.242.153 "cd $workdir; bash ./2D0.sh"
sleep 1
echo "[SSH] Start 2D1 on 10.112.242.24 node"
ssh root@10.112.242.24 "cd $workdir; bash ./2D1.sh"
