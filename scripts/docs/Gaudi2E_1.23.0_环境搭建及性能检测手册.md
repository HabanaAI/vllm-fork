# Gaudi2E 环境构建及验证手册 – v1.23.0 版本

本手册旨在为开发人员和系统管理员提供一份详尽的指南，指导如何在 Intel Gaudi2E 平台上从零开始构建、配置和验证 v1.23.0 版本的运行环境。本文档以 Ubuntu 22.04.3 LTS (Kernel 5.15.0) 为基础，全面覆盖了从底层硬件设置到上层应用测试的全过程。

主要内容包括：
- **环境准备**：涵盖服务器 BIOS 优化、操作系统（Linux）配置、Gaudi2E 驱动及相关软件包的安装与验证。
- **性能测试验证**：详细介绍如何使用 Intel Gaudi Qualification Tool (`hl_qual`) 对硬件进行全面的健康检查和性能基准测试，包括内存、功耗、带宽和功能性测试。
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
        - [1.2.2 驱动及软件安装验证](#122-驱动及软件安装验证)
        - [1.2.3 环境安装](#123-环境安装)
- [2.0 性能测试验证](#20-性能测试验证)
    - [2.1 内存压力测试](#21-内存压力测试)
    - [2.2 功耗与 EDP 压力测试](#22-功耗与-edp-压力测试)
    - [2.3 连接/SerDes 测试](#23-连接serdes-测试)
    - [2.4 功能性测试（Functional Test 2）](#24-功能性测试functional-test-2)
    - [2.5 带宽测试](#25-带宽测试)
    - [2.6 hl_qual 报告结构说明](#26-hl_qual-报告结构说明)
    - [2.7 诊断工具](#27-诊断工具)
- [3.0 集合通讯测试](#30-集合通讯测试)
    - [3.1 hccl_demo 工具介绍](#31-hccldemo-工具介绍)
    - [3.2 hccl_demo 编译](#32-hccldemo-编译)
    - [3.3 Host NIC Scale-Out 配置](#33-host-nic-scale-out-配置)
        - [3.3.1 libfabric 安装步骤](#331-libfabric-安装步骤)
        - [3.3.2 hccl_ofi_wrapper 安装步骤](#332-hccl_ofi_wrapper-安装步骤)
    - [3.4 hccl_demo 测试示例](#34-hccldemo-测试示例)
- [参考连接](#参考连接)

## 1.0 环境准备

### 1.1 BIOS 设置以及操作系统设置

#### 1.1.1 BIOS 设置

请在 BIOS 里按照服务器或者主板说明书进行如下的设置：

- 设置 CPU 为性能模式（performance mode）
- 开启 CPU P-state
- 关闭 CPU C6 状态
- 关闭 SNC

参考步骤设置步骤

```
重启服务器进入BIOS配置界面
Socket Configuration -> Advanced Power Management Configuration
- CPU P state control
    - SpeedStep    [Enable]         
    - Turbo Mode [Enable]
- Hardware PM State Control
    - Hardware P-States [Native Mode with No Legacy Support]
- CPU C State Control
    - Enable Monitor MWAIT [Disable]
    - CPU C6 report [Disable]
    - Enhanced Halt State(CIE) [Dsiabled]
Advanced-> Uncore General Configuration
    - SNC [Disable]
```

#### 1.1.2 Linux OS 设置

进入 Linux OS 后，在主机上设置，在 GRUB 里执行如下设置 
- CPU 为性能模式
- 开启 inte_iommu 并设置为 passthrough 模式

参考配置操作如下：

```
打开文件 `/etc/default/grub`  
给变量 `GRUB_CMDLINE_LINUX_DEFAULT` 增加参数如下
GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on iommu=pt cpufreq.default_governor=performance intel_idle.max_cstate=0"
```

执行命令 `update-grub` 使命令生效，然后重启 OS。

查看 CPU 是否是 performance 模式。

```
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

<span style="color:yellow">**提示**</span>  如果使用 Kubernates 和 OpenShift 等云基础架构环境，可以跳过[安装驱动及软件](#121-安装驱动及软件)步骤，直接参考[环境安装](#123-环境安装)进行安装。

#### 1.2.1 安装驱动及软件

连接互联网，下载 Gaudi2E 驱动及相关软件包安装的执行脚本`habana_install.sh`。

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

[<span style="color:blue">**可选项**</span>] 用户可依据自身的需求安装如下的依赖包：

```bash
安装 habanalabs-container-runtime 依赖包以支持运行容器化应用
sudo apt install -y habanalabs-container-runtime

安装 habanalabs-qual-workload 依赖包以支持运行 hl_qual 性能测试中的 ResNet-50 训练测试
sudo apt install -y habanalabs-qual-workload

安装 Python 和 MPI 相关依赖包以支持运行 hl_qual 性能测试 power 和 EDP 测试
./habanalabs-installer.sh install -t deps -y -v

安装 ethtool 以支持网络接口的诊断和配置
sudo apt install -y ethtool
```

#### 1.2.2 驱动及软件安装验证

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
    先手动卸载已加载的模块
    rmmod habanalabs_ib; rmmod habanalabs_en; rmmod habanalabs_cn; rmmod habanalabs; rmmod habanalabs_compat

    然后重新加载模块
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

    ```
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

#### 1.2.3 环境安装

在安装好驱动及相关组件后，依据自身的使用场景，选择合适的环境进行安装。
- Bare Metal 环境安装 - 在裸金属上安装Intel Gaudi Pytorch 环境。 参考官方手册 [Bare Metal 环境安装手册](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/Bare_Metal_Installation.html#bare-metal-pytorch)
- Docker 环境安装 - 拉取和运行 Intel Gaudi Dockers。 参考官方手册 [Docker 环境安装手册](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/Docker_Installation.html#docker-installation)

如过选择 Kubernates 或 OpenShift 等云基础架构环境 
- Kubernetes 环境安装 - 使用 Intel Gaudi Base Operator 在 Kubernetes 环境中安装并自动化管理所有 Intel Gaudi 的驱动和软件。 参考官方手册 [Kubernetes 环境安装手册](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/Kubernetes_Installation/index.html#kubernetes-install)
- OpenShift 环境安装 - 使用 Intel Gaudi Base Operator 在 OpenShit 环境中安装并自动化管理所有 Intel Gaudi 的驱动和软件。 参考官方手册 [OpenShift 环境安装手册](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/OpenShift_Installation/index.html#intel-gaudi-base-operator-openshift)


## 2.0 性能测试验证

Intel Gaudi Qualification Tool `hl_qual` 是包含在驱动安装包内用于验证 Intel Gaudi 加速器硬件质量和性能的工具。通过运行一系列预定义的测试，用户可以确保其 Gaudi2E 硬件和软件环境配置正确，并达到预期的性能标准。

运行的测试集包括如下：
- 内存压力测试: 验证 HBM/内部内存的读写稳定性与纠错机制，在长时间压力下观察容量利用与错误统计，常用于排查间歇性 ECC 与超时问题。
- 功耗与 EDP 测试: 在不同负载下采样功率于温度与能效点（EDP）压力测试下，评估供电与散热裕度以及能效表现。
- 连接/SerDes 测试: 验证 SerDes 内/外部端口连通性、数据完整性与带宽稳定性，辅助定位链路训练、降速与误码相关问题。
- 功能性测试: 在真实/合成训练场景下同时驱动多单元（HBM、DMA、MME、TPC、SerDes 等），校验计算正确性与性能（FPS/吞吐），并在长时运行中暴露热、功耗、链路和计算性能问题。
- 带宽测试: 测量 DMA/PCI 带宽测量，覆盖 HBM/SRAM 内存通路和主机-设备 PCIe 通路，校验链路是否达标。

驱动安装好后，测试工具的默认安装路径在`/opt/habanalabs/qual/gaudi2/bin/hl_qual`，运行前请确保已配置好相关的环境变量，具体可参考[驱动及软件安装验证](#122-驱动及软件安装验证)。

工具使用说明可通过 `./hl_qual -gaudi2 -h` 查看。详细的工具使用说明和测试内容描述可参考官方连接 [Inetl Gaudi Qualification Tool 使用指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/index.html#gaudi-qualification-library)

### 2.1 内存压力测试

**目标**：在高压读写下验证 HBM 内存完整性，捕获 ECC/重置相关异常。

**前置**：Gaudi2 需先加载驱动参数：`sudo modprobe habanalabs timeout_locked=0`。

**子测试及参数**：（以 `--help` 为准）：

- `-hbm_dma_stress`：用 DMA 引擎读写。
- `-hbm_tpc_stress <read|write|read_write|full_rw>`：TPC load/store 压力，默认 `read_write`， 支持：`read`、`write`、`read_write`、`full_rw`。
- `-full_hbm_data_check_test`：遍历 HBM 做全量读写校验。
- `-i`：控制轮次 (1-1024)；单轮约 80s（DMA）、30s（TPC）、~7 分钟（全量校验）。每轮包含测试、ECC 读取、设备复位。
- `-skip_val`：可跳过 ECC 读取。
- `-skip_rst`：可跳过复位。

**判定**： 所选子测必须通过；无 ECC 错误；（如未跳过）复位成功。任一轮失败则整次失败。

**示例**（按需调整设备列表与迭代次数/时长）：

```bash
# DMA 压力，2 轮
./hl_qual -gaudi2 -c all -rmod parallel -i 2 -hbm_dma_stress -dis_mon

# TPC 压力，full_rw 子模式，2 轮
./hl_qual -gaudi2 -c all -rmod parallel -i 2 -hbm_tpc_stress full_rw -dis_mon

# 全量数据校验，跳过复位
./hl_qual -gaudi2 -c all -rmod parallel -i 1 -full_hbm_data_check_test -skip_rst -dis_mon
```

### 2.2 功耗与 EDP 压力测试

**目标**：验证在高功耗与动态功耗周期下，设备的散热、供电与功率管理（PID）是否稳定，确保长时极限负载不触发过温/供电故障，计算结果保持 bit‑exact。

**前置**：Gaudi2 运行前需加载驱动参数：`sudo modprobe habanalabs timeout_locked=0`。

**子测试及参数**：（以 `--help` 为准）：

- `-s`：功耗压力（Power Stress）将设备维持在恒定、较高的功率水平，适合长时间（>2h）稳定性验证。
    - `-l <extreme | high>`：功率档位（默认 high）。
    - `-enable_ports_check <all|int>`： 启用 Gaudi 网卡端口检查。
    - `-toggle`： 记录端口切换次数。
- `-e`：EDP Stress，按周期在高功率与空闲（低功率）间快速切换，验证电源响应与保护机制。
    - `-l <extreme | high | inc_power>`：功率档位（默认 high）。
    - `-Tw <1–20>`： 周期内高功率持续秒数。
    - `-Ts <1–20>`： 周期内空闲持续秒数。
    - `-sync`： 同步多设备的功率上升沿。
    - `-enable_ports_check <all|int>`： 启用 Gaudi 网卡端口检查。
    - `-toggle`： 记录端口切换次数。

**判定与注意事项**：

- Power Stress：不发生过温关机；MME 引擎计算结果需 bit‑exact。
- EDP Stress：测试需在设定时间内完成，期间无过热或电源故障。

**示例**（按需调整设备列表与迭代次数/时长）：

```bash
# Power Stress：并行运行 120 秒
./hl_qual -gaudi2 -c all -rmod parallel -s -t 120 -dis_mon

# EDP Stress：60 秒，周期高功率 3s + 空闲 1s，共约 10 个周期
./hl_qual -gaudi2 -c all -rmod parallel -t 60 -e -Tw 3 -Ts 1 -dis_mon

# EDP Stress：极限功率档位，端口完整性检查（内部端口）
./hl_qual -gaudi2 -c all -rmod parallel -t 120 -e -l extreme -enable_ports_check int -dis_mon
```

### 2.3 连接/SerDes 测试

**目标**：验证内部/外部端口连通性与数据完整性，评估端口/设备对带宽稳定性，定位链路训练、降速与误码相关问题。

**前置**：Gaudi2 运行前需加载驱动参数：`sudo modprobe habanalabs timeout_locked=0`。

**子测试及参数**：（以 `--help` 为准）：

- Serdes Base (`-nic_base`): 基础连通/带宽测试。
    - `-test_type <pairs|allreduce|allgather|loopback|bandwidth>`:
        - `pairs`: 内部端口两两校验与数据一致性。
        - `allgather`: 计算集体通信带宽。
        - `loopback`: 外部端口需环回器，仅计算带宽。
        - `bandwidth`: 唯一设备对间的带宽。
        - `allreduce`: 计算集体通信带宽。
    - `-i <iters>`: 迭代次数 (50–10000)。
    - `-ep <epochs>`: Epochs数 (10–500, 仅 allgather/allreduce)。
    - `-sz <bytes>`: 传输大小。
    - `-enable_ports_check <all|int>`: 启用端口检查。
    - `-toggle`: 记录端口切换次数。
    - 说明: `allgather`/`allreduce`/`loopback`/`bandwidth` 仅统计带宽，不做端口完整性校验。Serdes Base 仅支持并行运行 (`-rmod parallel`)。
- E2E Concurrency (`-e2e_concurrency`): 按“端口”并发测量带宽并校验传输完整性，便于定位具体端口问题。带宽阈值参考：Gaudi2≈97Gbps。
    - `-t <sec>`: 测试时长 (1–3600)。
    - `-enable_ports_check <int|all>`: 启用端口检查。
    - `-toggle`: 记录端口切换次数。
- SER (`-ser`): 符号错误率测试；Gaudi2 支持 pre-/post‑FEC SER。
    - `-ber_enable`: 启用BER检查。
    - `-check_all`: 检查pre和post的SER FEC和BER。
    - 说明: 测试过程中设备可能执行硬复位，耗时较长。

**判定与注意事项**：

- Serdes Base：目标 rank 必须响应；接收数据需与参考数据一致，否则失败。
- E2E Concurrency：带宽与正确性达标（Gaudi2≈97Gbps）。
- SER：pre‑FEC 平均 ≤ 1e‑6，pre‑FEC 最大 ≤ 1e‑4；超限即失败。

**示例**（按需调整设备列表与迭代次数/时长）：

```bash
# Serdes Base：内部端口成对校验（并统计带宽）
./hl_qual -gaudi2 -c all -rmod parallel -i 100 -nic_base -test_type pairs -dis_mon

# Serdes Base：AllReduce 带宽统计（多 epoch/iteration）
./hl_qual -gaudi2 -c all -rmod parallel -i 100 -ep 200 -nic_base -test_type allreduce -dis_mon

# E2E Concurrency：并发按端口测带宽与完整性
./hl_qual -gaudi2 -c all -rmod parallel -t 30 -dis_mon -e2e_concurrency -enable_ports_check int

# SER：符号错误率测试
./hl_qual -gaudi2 -c all -rmod parallel -dis_mon -ser -check_all
```

### 2.4 功能性测试（Functional Test 2）

**目标**：模拟训练/推理场景下并发驱动加速卡的全部资源（HBM、DMA、MME、TPC、SerDes），检查系统的稳定性，评估吞吐并揭示长期运行的带宽/计算/功耗/链路问题。

**前置**：

- 驱动需已加载：`sudo modprobe habanalabs timeout_locked=0`。
- 建议设置环境变量：`export __python_cmd=python3`。

**子测试及参数**：（以 `--help` 为准）：

- `-f2`：选择 Functional Test 2。
- `-t <sec>`：测试时长（建议 240–172800 秒）。
- `-l <extreme|high>`：功率档位，默认 high。
- `-serdes <int|ext>`：开启内/外端口校验（外部需环回）。
- `-enable_ports_check <all|int>` / `-toggle`：端口状态检查与切换计数。
- `-rmod <serial|parallel>`：运行模式, 并行或串行。
- `-c <pci|all>`：设备选择，Gaudi2E建议分开模组进行测试 <quad_0 | quad_1>。
- `-dis_mon` 可关闭监控输出。

**判定与注意事项**：

- 输出需与参考张量 bit‑exact；任何比对失败即判失败。
- 吞吐须满足设备型号参考阈值；如显著下降需排查温控/功耗/链路。
- 长时运行时观察 ECC/重置/链路降级/设备重启等硬件异常；出现不可恢复错误视为失败。

**示例**（按需调整设备列表与迭代次数/时长）：

```bash
# 测试模组0 (卡0-3)，执行 450 秒，开启内部网络端口传输
./hl_qual -gaudi2 -c quad_0 -rmod parallel -f2 -t 450 -serdes int -dis_mon
# 测试模组1 (卡4-7)，执行 450 秒，开启内部网络端口传输
./hl_qual -gaudi2 -c quad_1 -rmod parallel -f2 -t 450 -serdes int -dis_mon
```

### 2.5 带宽测试

**目标**：评估 Gaudi2 设备的 PCI 和设备内存（HBM/SRAM）带宽，包括主机↔设备（Host↔HBM/SRAM）与设备内存间（HBM↔HBM、HBM↔SRAM、SRAM↔HBM）传输性能；验证串行/并行运行模式下的吞吐与系统瓶颈。

**前置**：

- 建议设置：`export __python_cmd=python3`。
- 建议在 Bare Metal 环境运行 PCI 并行测试（VM 可能不正确反映 PCIe 拓扑）。

**子测试及参数**：（以 `--help` 为准）：

- `-mb`：Memory bandwidth 测试选择器（包含 HBM/SRAM 与 PCI 子测）。
    - `-memOnly`：仅运行设备内存（HBM ==> HBM; SRAM ==> HBM; HBM ==> SRAM）子测。
    - `-pciOnly`：仅运行 PCI-HBM 子测(HOST ==> HBM; HBM ==> HOST)，`-b` 执行双向测试 (HOST <==> HBM)。
    - `-sramOnly`：仅运行 PCI-SRAM 子测 (HOST ==> SRAM; SRAM ==> HOST)，`-b` 执行双向测试 (HOST <==> SRAM)。
    - `-pciall`：激活所有 PCI 测试（`-pciOnly` 与 `-sramOnly` 的合集）。
    - `-b`：激活双向（simultaneous upload+download）PCI 测试。
    - `-gen <gen3|gen4>`：指定 PCIe 代以调整阈值, Gaudi2 默认 Gen‑4。
    - `-lanes <8|16>`：指定 PCIe lane 数, 默认 16。
- `-p`：激活 PCI test plugin（上传/下载，`-b` 可启用双向）。
    - `-t <sec>`：PCI 子测试单次时间（10–3600s，注意多子测会乘以该时间）。
    - `-size <bytes>`：上传/下载 buffer 大小（最小 1GB）。
    - `-rmod serial|parallel`：串行侧重单卡准确性，并行用于评估系统瓶颈。

**判定与注意事项**：

- PCI 判定阈值基于默认假设（例如 16 lanes、Gen‑4）；阈值为理论值并允许 10% 降级（低于阈值判 FAIL）。
- PCI 测试建议在串行模式下运行以获得单卡最准确带宽；并行模式用于评估多卡共享 PCIe/交换机瓶颈。
- 内存到内存测试（HBM/SRAM）有单独的预校准通过标准；测试插件会根据 `-gen`/`-lanes` 调整阈值。
- 在虚拟机环境上运行 PCI 测试可能不可靠，建议使用裸金属并导出 PCIe 拓扑进行校验。

**示例**（按需调整设备列表与迭代次数/时长）：

```bash
# 内存带宽并行测试
./hl_qual -gaudi2 -c all -rmod parallel -mb -memOnly -dis_mon

# PCI和SRAM带宽双向并行测试
./hl_qual -gaudi2 -c all -rmod parallel -mb -b -pciall -dis_mon

# PCI带宽瓶颈并行测试（20s, buffer 2GB）
./hl_qual -gaudi2 -c all -rmod parallel -t 20 -p -b -size 2147483648 -dis_mon
```

### 2.6 hl_qual 报告结构说明

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
7. **总结报告 (Closing Report)**：汇总全过程的统计指标（功率、时钟、温度），并给出单卡及系统的最终 <span style="color:green">**Passed**</span>/<span style="color:red">**Failed**</span> 判定。


详细的报告结构说明请参考官方文档 [hl_qual Report Structure](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/hl_qual_Report_Structure.html)

预计输出与测试失败调试方法请参考官方文档 [hl_qual Expected Output and Failure Debug](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/hl_qual_Expected_Output_and_Failure_Debug.html)


### 2.7 诊断工具

Intel Gaudi Qualification Tool 软件包中也集成了自动化诊断工具位于`/opt/habanalabs/qual/diag_tool` 旨在支持执行由多个 `hl_qual` 测试组成的复杂测试计划。

**主要功能**：
- **测试集自动化工具 (Test Plan Automation)**：允许用户通过配置文件定义并自动化执行一系列 `hl_qual` 测试用例，提高测试效率。
- **日志分析工具 (Log Analysis)**：对测试生成的复杂日志进行扫描、分析和汇总，帮助快速定位硬件故障或配置错误。
- **环境检测工具 (Qual Package Installation Validator)**：自动检查并验证当前系统中 `hl_qual` 软件包及其所有依赖组件是否已正确安装且版本兼容。
- **机架级脚本 (Rack Scale Script)**：提供支持在大规模机架（Rack）环境下进行统一的批量自动化测试和测试结果汇总分析。

详细的使用说明请参考官方文档 [Intel Gaudi Diagnostic Tool 指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/Diagnostic_Tool/index.html)。

## 3.0 集合通讯测试

### 3.1 hccl_demo 工具介绍
`hccl_demo` 是 Intel Habana 提供的一款 HCCL (Habana Collective Communication Library) 性能测试工具， 用于验证测试 Intel Gaudi 加速卡间 scale-up 以及 scale-out 的集合通信的带宽和时延。

**主要功能**：
- **测试类型**：支持如下的7种通讯原语的测试：`all_reduce`, `all_gather`, `broadcast`, `reduce`, `reduce_scatter`, `all2all`, `send_recv` 。
- **测试指标**：测量不同数据大小下集合通信操作的带宽和时延，帮助评估和优化多卡及多节点间的通信效率。
- **灵活配置**：允许用户通过命令行参数自定义测试的数据类型(float/bfloat16)、数据大小范围、迭代次数以及使用的集合通信算法等。


### 3.2 hccl_demo 编译

1.  **获取源码**：从 GitHub 克隆 `hccl_demo` 仓库。
    ```bash
    git clone https://github.com/HabanaAI/hccl_demo.git
    cd hccl_demo
    ```
2.  **编译**：使用 CMake 进行编译。
    ```bash
    不使用MPI时：
    make -j $(nproc)
    使用MPI时：
    MPI=1 make -j $(nproc)
    ```
    <span style="color:yellow">**提示**</span> 当切换MPI模式与非MPI模式时，请确保清理之前的编译文件，执行 `make clean` 后重新编译。


### 3.3 Host NIC Scale-Out 配置

若服务器配置了高速互联网卡（如 Mellanox CX6 / CX7）并连接至交换机，需要安装好相应的网卡驱动以及启用网卡端口并配置 IP Address。然后在测试环境中安装 libfabric 及 hccl_ofi_wrapper 库来使能 4 卡以上的通信互联。  
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
    make -j 32 && make install
    popd
    rm -rf /tmp/libfabric
    ```

    建议将通讯库的加载写入到 `~/.bashrc` ：

    ```bash
    export LD_LIBRARY_PATH=$LIBFABRIC_ROOT/lib:$LD_LIBRARY_PATH
    ```

#### 3.3.2 hccl_ofi_wrapper 安装步骤

1. 克隆 hccl_ofi_wrapper 仓库：

    ```bash
    git clone https://github.com/HabanaAI/hccl_ofi_wrapper.git
    ```

2. 定义`LIBFABRIC_ROOT` 环境变量：

    ```bash
    export LIBFABRIC_ROOT=/tmp/libfabric-1.20.0
    ```

3. 编译 hccl_ofi_wrapper：

    ```bash
    cd hccl_ofi_wrapper
    mkdir -j 10
    ```

4. 拷贝 `libhccl_ofi_wrapper.so` 文件到 `/user/lib/habanalabs/` 目录下：

    ```bash
    cp libhccl_ofi_wrapper.so /usr/lib/habanalabs/libhccl_ofi_wrapper.so
    ```

5. 加载 `libhccl_ofi_wrapper.so` 库：

    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/habanalabs/
    ```

### 3.4 hccl_demo 测试示例

`hccl_demo` 提供了 Python Wrapper 脚本 `run_hccl_demo.py` 用于简化测试的执行，执行参数说明请参照：[Python Wrapper Arguments](https://github.com/HabanaAI/hccl_demo?tab=readme-ov-file#python-wrapper-arguments)。


1. 单机 8 卡测试 all_reduce, 数据集大小 32 MB 测试网络和算法带宽（单服务器配备8张 Gaudi2E 卡）：

    ```bash
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1000 --ranks_per_node 8

    --- 输出示例 ---
    ...
    ###############################################################################
    [BENCHMARK] hcclAllReduce(src!=dst, count=8388608, dtype=float, iterations=1000)
    [BENCHMARK]     NW Bandwidth   : <Test results> GB/s
    [BENCHMARK]     Algo Bandwidth : <Test results> GB/s
    ###############################################################################
    ```

2. 单机 8 卡测试 all_reduce, 数据集大小范围 1MB - 32MB 测试网络和算法带宽（单服务器配备8张 Gaudi2E 卡）：

    ```bash
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size_range 1m 32m --test all_reduce --loop 1000 --ranks_per_node 8

    --- 输出示例 ---
    ...
    ###################################################
    [BANDWIDTH SUMMARY REPORT]
    (src!=dst, collective=all_reduce, iterations=1000)

    size              count             type              redop             time              algoBW            nw_bw             
    (B)               (elements)                                            (ms)              (GB/s)            (GB/s)            
    1048576           262144            float             sum               <time> ms       <results> GB/s   <results> GB/s   
    2097152           524288            float             sum               <time> ms       <results> GB/s   <results> GB/s   
    4194304           1048576           float             sum               <time> ms       <results> GB/s   <results> GB/s   
    8388608           2097152           float             sum               <time> ms       <results> GB/s   <results> GB/s   
    16777216          4194304           float             sum               <time> ms       <results> GB/s   <results> GB/s   
    33554432          8388608           float             sum               <time> ms       <results> GB/s   <results> GB/s   
    ```

3. 单机 4 卡测试 all_reduce, 数据集大小 32 MB 测试网络和算法带宽（单服务器配备8张 Gaudi2E 卡）：

    ```bash
    测试模组0 (卡0-3)：
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1000 --ranks_per_node 8 --custom_comm 0,1,2,3

    测试模组1 (卡4-7)：
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1000 --ranks_per_node 8 --custom_comm 4,5,6,7
    ```

4. 单机 8 卡测试 all_reduce, 数据集大小 32 MB 进程执行和设备通讯时延（单服务器配备8张 Gaudi2E 卡）：

    ```bash
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1000 --ranks_per_node 8 --measure latency

    --- 输出示例 ---
    ...
    #########################################################################################
    [BENCHMARK] hcclAllReduce(dataSize=33554432, count=8388608, dtype=float, iterations=1000)
    [BENCHMARK]     Host Latency   : <Test results> ms
    [BENCHMARK]     Device Latency : <Test results> ms
    #########################################################################################
    ```

5. 单机 8 卡测试 all_reduce, 数据集大小范围 1 MB - 32 MB 测试进程执行和设备通讯时延（单服务器配备8张 Gaudi2E 卡）：

    ```bash
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size_range 1m 32m --test all_reduce --loop 1000 --ranks_per_node 8 --measure latency

    [LATENCY SUMMARY REPORT]
    (src!=dst, collective=all_reduce, iterations=1000)

    size              count             type              redop             Host Latency      Device Latency    
    (B)               (elements)                                            (ms)              (ms)              
    1048576           262144            float             sum               <results> ms       <results> ms       
    2097152           524288            float             sum               <results> ms       <results> ms       
    4194304           1048576           float             sum               <results> ms       <results> ms       
    8388608           2097152           float             sum               <results> ms       <results> ms       
    16777216          4194304           float             sum               <results> ms       <results> ms       
    33554432          8388608           float             sum               <results> ms       <results> ms     
    ```

6. 双机 16 卡测试 all_reduce, 数据集大小 32 MB 测试网络和算法带宽（单服务器配备8张 Gaudi2E 卡）：

    ```bash
    python3 run_hccl_demo.py --test all_reduce --loop 1000 --size 32m -mpi --host 10.111.12.234:8,10.111.12.235:8

    or

    python3 run_hccl_demo.py --test all_reduce --loop 1000 --size 32m -mpi --host <path/to/hostfile.txt>
    ```

## 参考连接
- [Intel Gaudi v1.23.0 官方指南](https://docs.habana.ai/en/v1.23.0/index.html)
- [Intel Gaudi 驱动安装指南](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Driver_Installation.html)
- [Intel Gaudi 系统管理工具指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Embedded_System_Tools_Guide/System_Management_Interface_Tool.html#system-management-tools)
- [Intel Gaudi 环境安装指南](https://docs.habana.ai/en/v1.23.0/Installation_Guide/Additional_Installation/index.html)
- [Inetl Gaudi Qualification Tool 使用指南](https://docs.habana.ai/en/v1.23.0/Management_and_Monitoring/Qualification_Library/index.html#gaudi-qualification-library)
- [Intel Habana Communications Library GitHub](https://github.com/HabanaAI/HCL)
- [Intel HCCL Demo GIthub](https://github.com/HabanaAI/hccl_demo)
- [Intel Gaudi RDMA PerfTest Tool 使用指南](https://docs.habana.ai/en/latest/Management_and_Monitoring/RDMA_PerfTest_Tool/RDMA_PerfTest_Tool.html)