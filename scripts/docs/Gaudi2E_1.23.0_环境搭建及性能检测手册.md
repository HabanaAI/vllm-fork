# Gaudi2E 环境构建及验证手册 – v1.23.0 版本

本手册旨在为开发人员和系统管理员提供一份详尽的指南，指导如何在 Intel Gaudi2E 平台上从零开始构建、配置和验证 v1.23.0 版本的运行环境。本文档以 Ubuntu 22.04.3 LTS (Kernel 5.15.0) 为基础，全面覆盖了从底层硬件设置到上层应用测试的全过程。

**主要内容包括**：

- **环境准备**：涵盖服务器 BIOS 优化、操作系统（Linux）配置、Gaudi2E 驱动及相关软件包的安装与验证。
- **性能测试验证**：详细介绍如何使用 Intel Gaudi Qualification Tool (`hl_qual`) 在用户系统上对Gaudi2E硬件进行快速的健康检测。
- **集合通信测试**：指导用户编译和运行 `hccl_demo` 工具，以评估单机（Scale-up）及多机（Scale-out）环境下的集合通信性能。

通过遵循本手册的步骤，用户可以确保其 Gaudi2E 系统配置正确、硬件运行稳定，并为后续的深度学习模型训练和推理任务奠定坚实的基础。

## 目录

- [1.0 环境准备](#10-环境准备)
    - [1.1 BIOS 设置以及操作系统设置](#11-bios-设置以及操作系统设置)
        - [1.1.1 BIOS 设置](#111-bios-设置)
        - [1.1.2 Linux OS 设置](#112-linux-os-设置)
        - [1.1.3 Gaudi2E 系统检查](#113-gaudi2e-系统检查)
    - [1.2 安装驱动及相关组件](#12-安装驱动及相关组件)
        - [1.2.1 安装驱动及软件](#121-安装驱动及软件)
        - [1.2.2 Docker 环境配置](#122-docker-环境配置)
        - [1.2.3 驱动及软件安装验证](#123-驱动及软件安装验证)
        - [1.2.4 （可选项）环境安装](#124-可选项环境安装)
- [2.0 性能测试验证](#20-性能测试验证)
    - [2.1 hl_qual 工具介绍](#21-hl_qual-工具介绍)
    - [2.2 基础测试](#22-基础测试)
    - [2.3 报告结构说明](#23-报告结构说明)
- [3.0 集合通讯测试](#30-集合通讯测试)
    - [3.1 hccl_demo 工具介绍](#31-hccldemo-工具介绍)
    - [3.2 hccl_demo 编译](#32-hccldemo-编译)
    - [3.3 Host NIC Scale-Out 配置](#33-host-nic-scale-out-配置)
        - [3.3.1 libfabric 安装步骤](#331-libfabric-安装步骤)
        - [3.3.2 hccl_ofi_wrapper 安装步骤](#332-hccl_ofi_wrapper-安装步骤)
        - [3.3.3 libfabric 和 hccl_ofi_wrapper 库加载](#333-libfabric-和-hccl_ofi_wrapper-库加载)
    - [3.4 基础测试](#34-基础测试)
- [参考连接](#参考连接)

## 1.0 环境准备

### 1.1 BIOS 设置以及操作系统设置

#### 1.1.1 BIOS 设置

请在 BIOS 里按照服务器或者主板说明书进行如下的设置：

- 设置 CPU 为性能模式（performance mode）
- 开启 CPU P-state
- 关闭 CPU C6 状态
- 关闭 SNC

参考步骤设置步骤:

```text
# 重启服务器进入BIOS配置界面
Socket Configuration -> Advanced Power Management Configuration
- CPU P state control
    - SpeedStep    [Enable]         
    - Turbo Mode   [Enable]
- Hardware PM State Control
    - Hardware P-States [Native Mode with No Legacy Support]
- CPU C State Control
    - Enable Monitor MWAIT      [Disabled]
    - CPU C6 report             [Disable]
    - Enhanced Halt State(CIE)  [Disabled]
Advanced-> Uncore General Configuration
    - SNC [Disable]
```

#### 1.1.2 Linux OS 设置

进入 Linux OS 后，在主机上设置，在 GRUB 里执行如下设置
- CPU 为性能模式

    参考配置操作如下：

    ```bash
    # 打开文件 `/etc/default/grub`  
    # 给变量 `GRUB_CMDLINE_LINUX_DEFAULT` 增加参数如下
    GRUB_CMDLINE_LINUX_DEFAULT="cpufreq.default_governor=performance intel_idle.max_cstate=0"
    ```

    [<span style="color:blue">**可选项**</span>] 对于 OS: Ubuntu 24.04.2/22.04.5 和 Linux kernel 6.8 需要开启 inte_iommu 并设置为 passthrough 模式

    ```bash
    # 对于 Ubuntu 24.04.2/22.04.5 和 Linux kernel 6.8 版本，增加如下参数
    GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on iommu=pt cpufreq.default_governor=performance intel_idle.max_cstate=0"
    ```

    执行命令 `update-grub` 使命令生效，然后重启 OS。

    查看 CPU 是否是 performance 模式。

    ```text
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    ```

    如果输出为 performance，则说明 CPU 已经设置为性能模式。

- 关闭 NUMA balancing

    ```bash
    echo 0 > /proc/sys/kernel/numa_balancing
    ```

- 设置 hugepages

    ```bash
    sudo sysctl -w vm.nr_hugepages=15000
    echo "vm.nr_hugepages=15000" | sudo tee -a /etc/sysctl.conf
    ```

#### 1.1.3 Gaudi2E 系统检查

检查 Gaudi2E 设备是否能在操作系统中被识别，通过lspci命令查看

```bash
lspci -d 1da3: -nn
--- 输出示例 ---
29:00.0 Processing accelerators [1200]: Habana Labs Ltd. Device [1da3:1021] (rev 05)
2a:00.0 Processing accelerators [1200]: Habana Labs Ltd. Device [1da3:1021] (rev 05)
3a:00.0 Processing accelerators [1200]: Habana Labs Ltd. Device [1da3:1021] (rev 05)
3b:00.0 Processing accelerators [1200]: Habana Labs Ltd. Device [1da3:1021] (rev 05)
aa:00.0 Processing accelerators [1200]: Habana Labs Ltd. Device [1da3:1021] (rev 05)
ab:00.0 Processing accelerators [1200]: Habana Labs Ltd. Device [1da3:1021] (rev 05)
bb:00.0 Processing accelerators [1200]: Habana Labs Ltd. Device [1da3:1021] (rev 05)
bc:00.0 Processing accelerators [1200]: Habana Labs Ltd. Device [1da3:1021] (rev 05)
```

### 1.2 安装驱动及相关组件

在服务器 Bare Metal 上安装驱动及相关依赖组件。

<span style="color:yellow">**提示**</span>  如果使用 Kubernetes 和 OpenShift 等云基础架构环境，可以跳过[安装驱动及软件](#121-安装驱动及软件)步骤，直接参考[（可选项）环境安装](#124-可选项环境安装)进行安装。

#### 1.2.1 安装驱动及软件

1. 连接互联网，下载 Gaudi2E 驱动及相关软件包安装的执行脚本`habana_install.sh`。

    ```bash
    wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.23.0/habanalabs-installer.sh
    chmod +x habanalabs-installer.sh
    ./habanalabs-installer.sh install --type base -y

    --- 输出示例 ---
    ...
    [  +0.000005] habanalabs 0000:27:00.0: Successfully added device 0000:27:00.0 to habanalabs driver
    [  +0.039769] habanalabs_cn: loading driver, version: 1.23.0-2eae87a
    [  +0.007822] habanalabs_en: loading driver, version: 1.23.0-2eae87a
    [  +0.034138] habanalabs_ib: loading driver, version: 1.23.0-2eae87a
    ================================================================================
    Habanalabs software was installed successfully
    ================================================================================
    ================================================================================
    Full install log: /root/habanalabs-installer-log/install-2026-01-06-19-42-55.log
    ================================================================================
    ```

    安装日志存放在默认路径`/root/habanalabs_installer_logs/`, 如果安装过程中出现问题，可以查看日志文件`install-2026-01-06-19-42-55.log`以获取详细信息。

2. 安装 habanalabs-container-runtime 依赖包以支持运行容器化应用

    ``` bash
    sudo apt install -y habanalabs-container-runtime
    ```

3. 安装 libfabric 和 hccl_ofi_wrapper 以支持4卡及以上的scale-out集合通信测试, 参考[Host NIC Scale-Out 配置](#33-host-nic-scale-out-配置)

[<span style="color:blue">**可选项**</span>] 用户可依据自身的需求安装如下的依赖包：

1. 安装 habanalabs-qual-workload 依赖包以支持运行 hl_qual 性能测试中的 ResNet-50 训练测试

    ``` bash
    sudo apt install -y habanalabs-qual-workload
    ```

2. 安装 Python 和 MPI 相关依赖包以支持运行 hl_qual 性能测试 power 和 EDP 测试

    ```bash
    ./habanalabs-installer.sh install -t deps -y -v
    ```

3. 安装 ethtool 以支持网络接口的诊断和配置

    ```bash
    sudo apt install -y ethtool
    ```

#### 1.2.2 Docker 环境配置

配置 Docker 环境以支持运行 Intel Gaudi 容器化应用。详细说明请参考官方文档[Docker 环境安装手册](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/Docker_Installation.html#docker-installation)。

1. 下载安装docker，详细步骤参考[Docker 官方安装文档](https://docs.docker.com/engine/install/ubuntu/)

    ```bash
    # 移除系统上冲突和残留的安装包
    sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc | cut -f1)

    # 添加 docker 官方的 GPG key
    sudo apt update
    sudo apt install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # 添加 docker 软件源
    sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
    Types: deb
    URIs: https://download.docker.com/linux/ubuntu
    Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
    Components: stable
    Signed-By: /etc/apt/keyrings/docker.asc
    EOF

    # 安装 docker 引擎和 docker-compose
    sudo apt update
    sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```

2. 配置 docker 以支持 habanalabs-container-runtime

    ```bash
    sudo tee /etc/docker/daemon.json <<EOF
    {
        "runtimes": {
            "habana": {
                    "path": "/usr/bin/habana-container-runtime",
                    "runtimeArgs": []
            }
        }
    }
    EOF
    ```

3. 重启 docker 服务

    更新 systemd 配置并重启 docker 服务， 确保配置生效且 docker 正常运行。

    ```bash
    sudo systemctl daemon-reload; sudo systemctl restart docker

    # 检查 docker 服务状态，确保其处于 active (running) 状态
    sudo systemctl status docker
    --- 输出示例 ---
    docker.service - Docker Application Container Engine
     Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
     Active: active (running) since Sun 2026-01-18 12:09:59 UTC; 1 week 0 days ago
    ```

4. 拉取 Intel 官方提供的 Docker 镜像

    ```bash
    docker pull vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest
    ```

5. 运行 Intel Gaudi Docker 容器

    ```bash
    docker run -itd --name gaudi2e_1.23.0 --runtime=habana \
        -e HABANA_VISIBLE_DEVICES=all \
        --device=/dev:/dev -v /dev:/dev \
        -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
        --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK \
        --ulimit memlock=-1:-1 --net=host --ipc=host \
        vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest
    ```

    启动容器后， 执行`docker exec -it gaudi2e_1.23.0 bash` 进入容器，运行 `hl-smi` 命令查看设备和驱动状态。

    [<span style="color:red">**注意**</span>]：
    1. 容器内的环境变量和驱动配置已经预先设置好，无需额外配置。
    2. 物理机上可以通过运行 `hl-smi` 查看系统上的程序对Gaudi设备的占用， 但是容器内运行 `hl-smi` 仅能查看该容器内的程序对Gaudi设备的占用情况。
    3. 每张Gaudi设备仅能运行单一程序无法共享， 用户请合理分配物理机和容器内的Gaudi设备使用，避免资源冲突造成程序运行失败。
    4. 用户可以通过设置如下的环境变量来指定程序运行在特定的Gaudi设备上：

        ```bash
        # 检查当前系统上的Gaudi设备编号（index）和模块号 (module_id)的映射关系
        # 注： index 是驱动加载的设备编号， module_id 是设备的物理模块号
        hl-smi -Q index,module_id -f csv
        --- 输出示例 ---
        index, module_id
        0, 2
        1, 6
        2, 3
        3, 7
        4, 0
        5, 4
        6, 5
        7, 1
        # 如输出所示，当前系统中设备编号0对应模块号2， 设备编号1对应模块号6， 以此类推。

        # 指定单卡，程序运行在模块号3的Gaudi设备上
        HLS_MODULE_ID=3 <程序>
        # 指定多卡，程序运行在模块号1和2的Gaudi设备上
        HABANA_VISIBLE_MODULES=1,2 <程序> 
        ```

#### 1.2.3 驱动及软件安装验证

1. 使用 `lsmod` 命令查看 habanalabs 驱动模块是否加载和运行成功

    ```bash
    lsmod | grep habanalabs

    --- 输出示例 ---
    habanalabs_ib          98304  0
    habanalabs_en          69632  0
    habanalabs_cn         864256  1 habanalabs_en
    habanalabs           2248704  0
    habanalabs_compat      16384  1 habanalabs
    ib_uverbs             139264  3 habanalabs_ib,rdma_ucm,mlx5_ib
    ib_core               430080  9 rdma_cm,ib_ipoib,iw_cm,ib_umad,habanalabs_ib,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm
    ```

2. 使用 `dmesg` 命令查看驱动加载日志，确认驱动版本和设备识别情况

    ```bash
    dmesg | grep habanalabs

    --- 输出示例 ---
    [    1.234567] habanalabs 0000:29:00.0: Habana Labs Gaudi2E device detected
    [    1.234890] habanalabs 0000:29:00.0: Driver version: 1.23.0-2eae87a
    [    1.235123] habanalabs 0000:2a:00.0: Habana Labs Gaudi2E device detected
    [    1.235456] habanalabs 0000:2a:00.0: Driver version: 1.23.0-2eae87a
    ```

    <span style="color:yellow">**提示**</span> 如果驱动未正确加载，可以尝试手动加载驱动模块：

    ```bash
    # 先手动卸载已加载的模块
    rmmod habanalabs_ib && rmmod habanalabs_en && rmmod habanalabs_cn && rmmod habanalabs && rmmod habanalabs_compat

    # 然后重新加载模块
    modprobe habanalabs_compat && modprobe habanalabs timeout_locked=0 && modprobe habanalabs_cn && modprobe habanalabs_en && modprobe habanalabs_ib
    ```

3. 使用 Gaudi 的系统管理工具 `hl-smi` 验证和查看驱动版本及设备信息。详细的工具使用说明可参考官方连接 [Intel Gaudi 系统管理工具指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Embedded_System_Tools_Guide/System_Management_Interface_Tool.html#system-management-tools)

    ```bash
    hl-smi

    --- 输出示例 ---
    +-----------------------------------------------------------------------------+
    | HL-SMI Version:                              hl-1.23.0-fw-62.2.1.1          |
    | Driver Version:                                     1.23.0-2eae87a          |
    | Nic Driver Version:                                 1.23.0-2eae87a          |
    |-------------------------------+----------------------+----------------------+
    | AIP  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncor-Events|
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | AIP-Util  Compute M. |
    |===============================+======================+======================|
    |   0  HL-288E             N/A  | 0000:2a:00.0     N/A |                   0  |
    | N/A   28C   P0   80W /  450W  |   768MiB /  98304MiB |     0%            0% |
    |-------------------------------+----------------------+----------------------+
    |   1  HL-288E             N/A  | 0000:aa:00.0     N/A |                   0  |
    | N/A   29C   P0   84W /  450W  |   768MiB /  98304MiB |     0%            0% |
    |-------------------------------+----------------------+----------------------+
    |   2  HL-288E             N/A  | 0000:29:00.0     N/A |                   0  |
    | N/A   30C   P0   96W /  450W  |   768MiB /  98304MiB |     0%            0% |
    |-------------------------------+----------------------+----------------------+
    |   3  HL-288E             N/A  | 0000:ab:00.0     N/A |                   0  |
    | N/A   29C   P0   78W /  450W  |   768MiB /  98304MiB |     0%            0% |
    |-------------------------------+----------------------+----------------------+
    |   4  HL-288E             N/A  | 0000:3a:00.0     N/A |                   0  |
    | N/A   29C   P0   78W /  450W  |   768MiB /  98304MiB |     0%            0% |
    |-------------------------------+----------------------+----------------------+
    |   5  HL-288E             N/A  | 0000:bb:00.0     N/A |                   0  |
    | N/A   28C   P0   77W /  450W  |   768MiB /  98304MiB |     0%            0% |
    |-------------------------------+----------------------+----------------------+
    |   6  HL-288E             N/A  | 0000:3b:00.0     N/A |                   0  |
    | N/A   29C   P0   77W /  450W  |   768MiB /  98304MiB |     0%            0% |
    |-------------------------------+----------------------+----------------------+
    |   7  HL-288E             N/A  | 0000:bc:00.0     N/A |                   0  |
    | N/A   29C   P0   67W /  450W  |   768MiB /  98304MiB |     0%            0% |
    |-------------------------------+----------------------+----------------------+
    | Compute Processes:                                               AIP Memory |
    |  AIP       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |   0        N/A   N/A    N/A                                      N/A        |
    |   1        N/A   N/A    N/A                                      N/A        |
    |   2        N/A   N/A    N/A                                      N/A        |
    |   3        N/A   N/A    N/A                                      N/A        |
    |   4        N/A   N/A    N/A                                      N/A        |
    |   5        N/A   N/A    N/A                                      N/A        |
    |   6        N/A   N/A    N/A                                      N/A        |
    |   7        N/A   N/A    N/A                                      N/A        |
    +=============================================================================+
    ```

4. 利用软件管理工具`apt`查看已安装的软件包版本。

    ```bash
    apt list --installed | grep habana

    --- 输出示例 ---
    habanalabs-dkms/jammy,now 1.23.0-695 all [installed]
    habanalabs-firmware-odm/jammy,now 1.23.0-695 amd64 [installed]
    habanalabs-firmware-tools/jammy,now 1.23.0-695 amd64 [installed]
    habanalabs-firmware/jammy,now 1.23.0-695 amd64 [installed]
    habanalabs-graph/jammy,now 1.23.0-695 amd64 [installed]
    habanalabs-qual/jammy,now 1.23.0-695 amd64 [installed]
    habanalabs-rdma-core/jammy,now 1.23.0-695 all [installed]
    habanalabs-thunk/jammy,now 1.23.0-695 all [installed]
    ```

5. 确保系统正常运行，检查如下的系统的环境变量是否设定正确

    ```bash
    export HABANALABS_HLTHUNK_TESTS_BIN_PATH=/opt/habanalabs/src/hl-thunk/tests/
    export HABANA_LOGS=/var/log/habana_logs/
    export RDMA_CORE_ROOT=/opt/habanalabs/rdma-core/src
    export HABANA_PLUGINS_LIB_PATH=/usr/lib/habanatools/habana_plugins
    export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
    export RDMA_CORE_LIB=/opt/habanalabs/rdma-core/src/build/lib
    export HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw
    export DATA_LOADER_AEON_LIB_PATH=/usr/lib/habanalabs/libaeon.so
    export __python_cmd=python3
    ```

    这些环境变量已经在驱动安装过程中自动配置到系统文件 `/etc/profile.d/habanalabs*.sh` 中，如果存在环境变量缺失，请手动添加。

    ```bash
    source /etc/profile.d/habanalabs*.sh
    ```

6. 检查 Gaudi 设备 internal ports 的启用状态，确保网口处于 `UP` 状态。 如果出现 port 状态为 `DOWN`，需要查看 `dmesg` 中的日志排查问题，尝试重新加载驱动模块以及检查顶板的连接情况。

    <span style="color:yellow">**提示**</span> internal ports 是 Gaudi2E 设备内部用于多张Gaudi2E设备互联的高速通信端口。如果需要多台服务器互联，需要通过服务器上安插RDMA网卡进行跨机器互联, 实现方式可参照[Host NIC Scale-Out 配置](#33-host-nic-scale-out-配置)。

    ```bash
    for b in $(hl-smi -Q bus_id -f csv,noheader); do
    echo "=== $b ==="
    hl-smi -n link -i "$b"
    done

    --- 输出示例 ---
    === 0000:bb:00.0 ===
    port  0:        UP
    port  1:        UP
    port  2:        UP
    port  3:        UP
    port  4:        UP
    port  5:        UP
    port 10:        UP
    port 11:        UP
    port 12:        UP
    port 13:        UP
    port 14:        UP
    port 15:        UP
    port 16:        UP
    port 17:        UP
    port 18:        UP
    port 19:        UP
    port 20:        UP
    port 21:        UP
    ... (省略其他设备输出) ...
    ```

7. 使用 Intel Gaudi Qualification Tool `hl_qual` 进行硬件健康检测, 参照[性能测试验证](#20-性能测试验证)。

#### 1.2.4 （可选项）环境安装

在安装好驱动及相关组件后，依据自身的使用场景，选择合适的环境进行安装。
- Bare Metal 环境安装 - 在裸机上安装Intel Gaudi Pytorch 环境。 参考官方手册 [Bare Metal 环境安装手册](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/Bare_Metal_Installation.html#bare-metal-pytorch)

如过选择 Kubernetes 或 OpenShift 等云基础架构环境
- Kubernetes 环境安装 - 使用 Intel Gaudi Base Operator 在 Kubernetes 环境中安装并自动化管理所有 Intel Gaudi 的驱动和软件。 参考官方手册 [Kubernetes 环境安装手册](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/Kubernetes_Installation/index.html#kubernetes-install)
- OpenShift 环境安装 - 使用 Intel Gaudi Base Operator 在 OpenShit 环境中安装并自动化管理所有 Intel Gaudi 的驱动和软件。 参考官方手册 [OpenShift 环境安装手册](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/OpenShift_Installation/index.html#intel-gaudi-base-operator-openshift)

## 2.0 性能测试验证

Intel 提供了 Gaudi Qualification Tool Package 用于在用户的服务器上对 Gaudi2E 硬件进行全面的健康检查和性能基准测试。通过运行一系列预定义的测试，用户可以确保其 Gaudi2E 硬件和软件环境配置正确，并达到预期的性能标准。

### 2.1 hl_qual 工具介绍

`hl_qual` 是 Intel Gaudi Qualification Tool Package 中执行每个单元测试的统一接口应用程序，通过添加同用配置参数和测试插件专用参数来运行不同的测试。

运行的测试集包括如下：
- **内存压力测试**: 验证 HBM/内部内存的读写稳定性与纠错机制，在长时间压力下观察容量利用与错误统计，常用于排查间歇性 ECC 与超时问题。测试内容详细说明请参考[Memory Stress Test Plugin](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/Memory_Stress_Tests_Plugin.html)
- **功耗与 EDP 测试**: 在不同负载下采样功率于温度与能效点（EDP）压力测试下，评估供电与散热裕度以及能效表现。测试内容详细说明请参考[Power and EDP Stress Test Plugin](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/Power_and_EDP_Stress_Tests_Plugin.html)
- **SerDes 测试**: 验证 SerDes 内/外部端口连通性、数据完整性与带宽稳定性，辅助定位链路训练、降速与误码相关问题。测试内容详细说明请参考[Connectivity SerDes Tests Plugin](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/Connectivity_Serdes_Tests_Plugin.html)
- **功能性测试**: 在真实/合成训练场景下同时驱动多单元（HBM、DMA、MME、TPC、SerDes 等），校验计算正确性与性能（FPS/吞吐），并在长时运行中暴露热、功耗、链路和计算性能问题。测试内容详细说明请参考[Functional Tests Plugin](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/Functional_Tests_Plugin.html)
- **带宽测试**: 测量 DMA/PCI 带宽测量，覆盖 HBM/SRAM 内存通路和主机-设备 PCIe 通路，校验链路是否达标。测试内容详细说明请参考[Bandwidth Tests Plugin](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/Bandwidth_Tests_Plugin.html)

驱动安装好后，测试工具的默认安装路径在 `/opt/habanalabs/qual/gaudi2/bin/hl_qual`，运行前请确保已配置好相关的环境变量，具体可参考[驱动及软件安装验证](#122-驱动及软件安装验证)。

工具使用说明可通过 `./hl_qual -gaudi2 -h` 查看。详细的测试内容描述可参考官方连接 [Inetl Gaudi Qualification Tool 使用指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/index.html#gaudi-qualification-library)

Intel Gaudi Qualification Tool Package 也提供了一键式自动化诊断，测试和报告分析工具, 位于目录 `/opt/habanalabs/qual/diag_tool`，详细的使用说明请参考官方文档 [Intel Gaudi Diagnostic Tool 指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/Diagnostic_Tool/index.html)。

### 2.2 基础测试

以下提供了一些常用的 `hl_qual` 测试示例，用于快速校验Gaudi2E硬件在当前系统中的健康状况，用户可根据自身需求进行调整和运行其他测试。

**前置配置**：
- Gaudi2E 需先加载驱动参数：

    ```bash
    sudo modprobe habanalabs timeout_locked=0
    ```

- 切换到 `hl_qual` 工具目录：

    ```bash
    cd /opt/habanalabs/qual/gaudi2/bin/
    ```

- 设置 Python 解释器环境变量：

    ```bash
    export __python_cmd=python3
    ```

**测试示例**：

1. 测试全部Gaudi2E设备的三种传输模式下的PCI带宽 （Download: Host ==> Device; Upload: Host <== Device; Bidirectional: Host <==> Device）并且检查在PCIe传输链路中的可能存在的通讯带宽瓶颈。

    ```bash
    ./hl_qual -gaudi2 -c all -rmod parallel -t 20 -p -b -gen gen4 -dis_mon
    ```

    测试预期： <span style="color:green">**PASSED**</span>

2. 分别测试每个模组（模组0：卡0-3， 模组1：卡4-7）在模拟训练/推理场景下并发驱动加速卡的全部资源（HBM、DMA、MME、TPC、SerDes），检查系统的稳定性，评估吞吐并揭示长期运行的带宽/计算/功耗/链路问题。

    ```bash
    # 模组0测试命令：
    ./hl_qual  -gaudi2 -c quad_0 -dis_mon -rmod parallel -f2 -l extreme -serdes int -t 300 
    
    # 模组1测试命令：
    ./hl_qual  -gaudi2 -c quad_1 -dis_mon -rmod parallel -f2 -l extreme -serdes int -t 300 
    ```

    测试预期： <span style="color:green">**PASSED**</span>

如果测试结果显示为 <span style="color:red">**FAILED**</span>, 请参考测试失败调试方法官方文档 [hl_qual Expected Output and Failure Debug](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/hl_qual_Expected_Output_and_Failure_Debug.html)

### 2.3 报告结构说明

在`hl_qual`测试完成后会生成包含多个子报告的完整测试日志。

**报告存储与命名**：
- **默认路径**：测试日志通常保存在 `$HABANA_LOGS/qual`（若未定义环境变量，默认路径为 `/var/log/habana_logs/qual`）。
- **文件命名**：文件名包含服务器名、`hl_qual_report` 字样以及详细的时间戳（例如 `server_hl_qual_report_Sat_Dec_4_09-15-16_2021.log`）。

**报告组成部分**：
1. **设备识别报告 (Device Identification Report)**：展示 PCI 总线 ID 及设备的运行状态。
2. **hl-smi 简报 (hl-smi Short Report)**：列出设备的 bus_id、序列号、索引值、模组 ID 及设备类型。
3. **运行状态报告 (Operational Status Report)**：验证设备是否满足运行标准（如内存使用率、驱动状态等）。
4. **NUMA 节点报告 (NUMA Node Report)**：记录 NUMA 节点、CPU 集合以及 Gaudi 设备的分配情况。
5. **版本与命令行报告 (hl_qual Version and Command Line Report)**：记录 `hl_qual` 软件包版本及执行时使用的完整命令行。
6. **受测设备报告 (Tested Device Report)**：包含设备硬件详情（序列号、PCB 版本）、测试起止时间及内部插件运行数据。
7. **总结报告 (Closing Report)**：汇总全过程的统计指标（功率、时钟、温度），并给出单卡及系统的最终 <span style="color:green">**PASSED**</span>/<span style="color:red">**FAILED**</span> 判定。

详细的报告结构说明请参考官方文档 [hl_qual Report Structure](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/hl_qual_Report_Structure.html)

## 3.0 集合通讯测试

### 3.1 hccl_demo 工具介绍

`hccl_demo` 是 Intel Habana 提供的一款 HCCL (Habana Collective Communication Library) 性能测试工具， 用于验证测试 Intel Gaudi 加速卡间 scale-up 以及 scale-out 的集合通信的带宽和时延。

**主要功能**：
- **测试类型**：支持如下的7种通讯原语的测试：`all_reduce`, `all_gather`, `broadcast`, `reduce`, `reduce_scatter`, `all2all`, `send_recv` 。
- **测试指标**：测量不同数据大小下集合通信操作的带宽和时延，帮助评估和优化多卡及多节点间的通信效率。
- **灵活配置**：允许用户通过命令行参数自定义测试的数据类型(float/bfloat16)、数据大小范围、迭代次数以及使用的集合通信算法等。

### 3.2 hccl_demo 编译

1. **获取源码**：从 GitHub 克隆 `hccl_demo` 仓库。

   ```bash
   git clone https://github.com/HabanaAI/hccl_demo.git
   cd hccl_demo
   ```

2. **编译**：使用 CMake 进行编译。

   ```bash
   # 不使用MPI时
   make -j $(nproc)
   
   # 使用MPI时
   MPI=1 make -j $(nproc)
    ```

    <span style="color:yellow">**提示**</span> 当切换MPI模式与非MPI模式时，请确保清理之前的编译文件，执行 `make clean` 后重新编译。

### 3.3 Host NIC Scale-Out 配置

[<span style="color:yellow">**提示**</span>] 若服务器配置了高速互联网卡（如 Mellanox CX6 / CX7）并连接至交换机，需要安装好相应的网卡驱动以及启用网卡端口并配置 IP Address。然后在测试环境中安装 libfabric 及 hccl_ofi_wrapper 库来使能 4 卡以上的通信互联。  
如果没有配置高速互联网卡，则单机内跨模组或者4卡以上的互联则通过 **UPI** 通讯。

#### 3.3.1 libfabric 安装步骤

1. 预定义libfabric软件版本，需要 v1.20.0 及以上的版本。

    ```bash
    export REQUIRED_VERSION=1.20.0
    ```

2. 下载并安装 [libfabric](https://github.com/ofiwg/libfabric/releases)

    ```bash
    wget  https://github.com/ofiwg/libfabric/releases/download/v$REQUIRED_VERSION/libfabric-$REQUIRED_VERSION.tar.bz2 -P /tmp/libfabric
    pushd /tmp/libfabric
    tar -xf libfabric-$REQUIRED_VERSION.tar.bz2
    export LIBFABRIC_ROOT=/opt/libfabric
    mkdir -p ${LIBFABRIC_ROOT}
    chmod 777 ${LIBFABRIC_ROOT}
    cd libfabric-$REQUIRED_VERSION/
    ./configure --prefix=$LIBFABRIC_ROOT --with-synapseai=/usr
    make -j $(nproc) && make install
    popd
    rm -rf /tmp/libfabric
    ```

#### 3.3.2 hccl_ofi_wrapper 安装步骤

1. 克隆 hccl_ofi_wrapper 仓库：

    ```bash
    git clone https://github.com/HabanaAI/hccl_ofi_wrapper.git
    ```

2. 定义`LIBFABRIC_ROOT` 环境变量：

    ```bash
    export LIBFABRIC_ROOT=/opt/libfabric
    ```

3. 编译 hccl_ofi_wrapper：

    ```bash
    cd hccl_ofi_wrapper
    make -j $(nproc)
    ```

4. 拷贝 `libhccl_ofi_wrapper.so` 文件到 `/usr/lib/habanalabs/` 目录下并加载：

    ```bash
    cp libhccl_ofi_wrapper.so /usr/lib/habanalabs/libhccl_ofi_wrapper.so
    ldconfig
    ```

#### 3.3.3 libfabric 和 hccl_ofi_wrapper 库加载

将如下环境变量写入 `~/.bashrc` 文件中，确保每次登录时自动加载：

```bash
export LIBFABRIC_ROOT=/opt/libfabric
export LD_LIBRARY_PATH=$LIBFABRIC_ROOT/lib:$LD_LIBRARY_PATH
```

### 3.4 基础测试

`hccl_demo` 提供了 Python Wrapper 脚本 `run_hccl_demo.py` 用于简化测试的执行，执行参数说明请参照：[Python Wrapper Arguments](https://github.com/HabanaAI/hccl_demo?tab=readme-ov-file#python-wrapper-arguments)。

以下提供了一些常用的 `hccl_demo` 测试示例，用于快速校验Gaudi2E硬件在当前系统中的网络通讯的健康情况，用户可根据自身需求进行调整和运行其他测试。

1. 单机 8 卡测试 all_reduce, 数据集大小 32 MB ，循环10000次，测试网络和算法带宽（单服务器配备8张 Gaudi2E 卡）：

    ```bash
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 10000 --ranks_per_node 8

    --- 输出示例 ---
    ...
    ###############################################################################
    [BENCHMARK] hcclAllReduce(src!=dst, count=8388608, dtype=float, iterations=10000)
    [BENCHMARK]     NW Bandwidth   : <Test results> GB/s
    [BENCHMARK]     Algo Bandwidth : <Test results> GB/s
    ###############################################################################
    ```

    测试预期：打印出 NW Bandwidth 和 Algo Bandwidth 数值，如果有连接高性能网卡则网络带宽性能表现会有较大的提升。

2. （如果多机测试，以双机为例）双机 16 卡测试 all_reduce, 数据集大小 32 MB，循环10000次，测试网络和算法带宽（单服务器配备8张 Gaudi2E 卡）：

    ```bash
    python3 run_hccl_demo.py --test all_reduce --loop 10000 --size 32m -mpi --host 10.111.12.234:8,10.111.12.235:8

    or

    python3 run_hccl_demo.py --test all_reduce --loop 10000 --size 32m -mpi --host <path/to/hostfile.txt>
    ```

    测试预期：打印出 NW Bandwidth 和 Algo Bandwidth 数值。

 <span style="color:yellow">**提示**</span>： 如果测试中出现问题， 可以在执行命令前增加调试参数 `ENABLE_CONSOLE=true LOG_LEVEL_ALL=0 <hccl demo command>` 来输出更多的调试信息， 例如：

```bash
ENABLE_CONSOLE=true LOG_LEVEL=DEBUG HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 10000 --ranks_per_node 8 | tee hccl_demo_debug.log
```

<span style="color:green">**SUCCESS**</span>: 如果上述流程都顺利通过， 那恭喜你可以开始执行接下来的vllm等工作负载的操作。

## 参考连接

- [Intel Gaudi v1.23.0 官方指南](https://docs.habana.ai/en/v1.23.0/index.html)
- [Intel Gaudi 驱动安装指南](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Driver_Installation.html)
- [Intel Gaudi 系统管理工具指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Embedded_System_Tools_Guide/System_Management_Interface_Tool.html#system-management-tools)
- [Intel Gaudi 环境安装指南](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/index.html)
- [Inetl Gaudi Qualification Tool 使用指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/index.html#gaudi-qualification-library)
- [Intel Habana Communications Library GitHub](https://github.com/HabanaAI/HCL)
- [Intel HCCL Demo GIthub](https://github.com/HabanaAI/hccl_demo)
