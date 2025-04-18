1. you need install doca-ofed library on container.
```
# remove conflict libraries
apt remove ibutils libpmix-aws

# install doca, please refer
# 1. https://forums.developer.nvidia.com/t/installing-mellanox-ofed-drivers-for-my-ubuntu-22-04-5-lts-with-kernel-version-5-15-0-131-generic/322431 
# 2. https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
# 3. https://developer.nvidia.com/networking/mlnx-ofed-eula?mtag=linux_sw_drivers&mrequest=downloads&mtype=ofed&mver=MLNX_OFED-24.10-2.1.8.0&mname=MLNX_OFED_LINUX-24.10-2.1.8.0-ubuntu22.04-x86_64.tgz

wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.10.0/host/doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
sudo dpkg -i doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
sudo apt-get update
sudo apt-get -y install doca-ofed
```

2. try this command and see whether rdma is up. You need check whether mlx5_x is mapping to correct net interface and change mooncake_store.py if necessary.

```
$ ibdev2netdev
mlx5_0 port 1 ==> ens108np0 (Up)
mlx5_1 port 1 ==> ens9f0np0 (Up)
mlx5_2 port 1 ==> ens9f1np1 (Up)
mlx5_3 port 1 ==> ens109np0 (Up)
mlx5_4 port 1 ==> ens110np0 (Up)
mlx5_5 port 1 ==> ens111np0 (Up)
mlx5_6 port 1 ==> ens112np0 (Up)
mlx5_7 port 1 ==> ens113np0 (Up)
mlx5_8 port 1 ==> ens114np0 (Up)
mlx5_9 port 1 ==> ens115np0 (Up)

```