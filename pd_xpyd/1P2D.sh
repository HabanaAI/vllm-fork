workdir=/host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd
echo "Start 1P on current node"
bash ./1P.sh
sleep 1
echo "[SSH] Start 2D0 on 10.112.242.153 node"
ssh root@10.112.242.153 "cd $workdir; bash ./2D0.sh"
sleep 1
echo "[SSH] Start 2D1 on 10.112.242.24 node"
ssh root@10.112.242.24 "cd $workdir; bash ./2D1.sh"
