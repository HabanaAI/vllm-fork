echo "Kill on 1P"
bash ./killall.sh
sleep 1
echo "[SSH] kill on D1 10.112.242.153"
ssh root@10.112.242.153 "cd /host/mnt/disk002/kf/vllm-fork_pd/pd_xpyd; bash ./killall.sh"
sleep 1
echo "[SSH] Kill on D2 10.112.242.24"
ssh root@10.112.242.24 "cd /host/mnt/disk002/kf/vllm-fork_pd/pd_xpyd; bash ./killall.sh"

