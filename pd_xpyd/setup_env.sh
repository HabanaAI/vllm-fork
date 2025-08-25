echo "1. setting up general requirement......."
apt update
apt install git wget curl net-tools sudo iputils-ping etcd  -y
pip install colorlog

echo "2. setting up mooncake mooncake-transfer-engine private build............."
#Mooncake
wget https://github.com/hlin99/Mooncake/releases/download/private_buildv1/mooncake_transfer_engine-0.1.0-cp310-cp310-manylinux2014_x86_64.whl
pip install mooncake_transfer_engine-0.1.0-cp310-cp310-manylinux2014_x86_64.whl

echo "3. setting up RDMA for mooncake ..................."
#RDMA
apt remove ibutils libpmix-aws
wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.10.0/host/doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
dpkg -i doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
apt-get update
apt-get -y install doca-ofed

echo "4. setup neural-compressor ..........................."
pip install git+https://github.com/hlin99/neural-compressor.git@r1_woq_moe

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

