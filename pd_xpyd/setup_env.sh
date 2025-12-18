echo "1. setting up general requirement......."
. /etc/os-release
if [[ "$ID" != "ubuntu" ]]; then
    echo "This script is only supported on Ubuntu. Exiting."
    exit 1
fi
apt update
apt install git wget curl net-tools sudo iputils-ping -y
if [[ "$VERSION_ID" == "22.04" ]]; then
    apt install etcd -y
elif [[ "$VERSION_ID" == "24.04" ]]; then
    pushd /tmp
    wget https://github.com/etcd-io/etcd/releases/download/v3.6.1/etcd-v3.6.1-linux-amd64.tar.gz
    tar xzf etcd-v3.6.1-linux-amd64.tar.gz
    sudo mv etcd-v3.6.1-linux-amd64/etcd* /usr/local/bin/
    popd
fi

pip install colorlog

echo "2. setting up mooncake mooncake-transfer-engine build............."
#Mooncake
#if [[ "$VERSION_ID" == "22.04" ]]; then
#wget https://github.com/hlin99/Mooncake/releases/download/private_buildv3/mooncake_transfer_engine-0.3.5-cp310-cp310-manylinux_2_17_x86_64.whl
#pip install mooncake_transfer_engine-0.3.5-cp310-cp310-manylinux_2_17_x86_64.whl --force-reinstall
#elif [[ "$VERSION_ID" == "24.04" ]]; then
#pip install mooncake_transfer_engine==0.3.6
#fi
pip uninstall -y mooncake-transfer-engine && pip install mooncake-transfer-engine-non-cuda==0.3.7.post2

echo "3. setting up RDMA for mooncake ..................."
#RDMA
apt remove ibutils libpmix-aws

if [[ "$ID" == "ubuntu" && "$VERSION_ID" == "22.04" ]]; then
    wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.10.0/host/doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
    dpkg -i doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
    apt-get update
    apt-get -y install doca-ofed
elif [[ "$ID" == "ubuntu" && "$VERSION_ID" == "24.04" ]]; then
    export DOCA_URL="https://linux.mellanox.com/public/repo/doca/3.2.0/ubuntu24.04/x86_64/"
    BASE_URL=$([ "${DOCA_PREPUBLISH:-false}" = "true" ] && echo https://doca-repo-prod.nvidia.com/public/repo/doca || echo https://linux.mellanox.com/public/repo/doca)
    DOCA_SUFFIX=${DOCA_URL#*public/repo/doca/}; DOCA_URL="$BASE_URL/$DOCA_SUFFIX"
    curl $BASE_URL/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list
    apt-get update
    apt-get -y install doca-roce
fi

ibdev2netdev
#mlx5_0 port 1 ==> ens108np0 (Up)
#mlx5_1 port 1 ==> ens9f0np0 (Up)
#mlx5_2 port 1 ==> ens9f1np1 (Up)
#mlx5_3 port 1 ==> ens109np0 (Up)
#mlx5_4 port 1 ==> ens110np0 (Up)
#mlx5_5 port 1 ==> ens111np0 (Up)
#mlx5_6 port 1 ==> ens112np0 (Up)
#mlx5_7 port 1 ==> ens113np0 (Up)
#mlx5_8 port 1 ==> ens114np0 (Up)
#mlx5_9 port 1 ==> ens115np0 (Up)

