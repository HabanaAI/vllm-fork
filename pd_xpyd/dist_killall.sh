workdir=/host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd

source env.sh

echo "Kill on 1P"
bash ./killall.sh
sleep 1
echo ${ROLE_IP[D0]},${ROLE_IP[D1]}
ssh root@${ROLE_IP[D0]} "cd $workdir; bash ./killall.sh"


sleep 1
echo "[SSH] Kill on D2 ${ROLE_IP[D1]}"
ssh root@${ROLE_IP[D1]} "cd $workdir; bash ./killall.sh"

