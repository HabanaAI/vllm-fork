# Environment setup

## Hardware Requirements
This is used to set up vLLM service on Intel(R) Gaudi(R) accelerator. Please refer to [Hardware and Network Requirements](https://docs.habana.ai/en/latest/Installation_Guide/Platform_Readiness.html#) to check your hardware readiness.

### Set CPU to Performance Mode
Please change the CPU setting to be performance optimization mode, enable CPU P-state and disable CPU C6-state in BIOS setup. Execute the command below in OS to make sure get the best CPU performance.

```
sudo echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo sysctl -w vm.nr_hugepages=15000
sudo echo 0 > /proc/sys/kernel/numa_balancing
```

## Software Requirements
* The supported OS are in [Supported Configurations and Components](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html#support-matrix)
* Refer to [Driver and Software Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install the Intel(R) Gaudi(R) driver and software stack (>= 1.21.3) on each node. Make sure `habanalabs-container-runtime` is installed.
* Refer to [Firmware Upgrade](https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html) to upgrade the Gaudi(R) firmware to 1.20.1 version or newer version on each node.
* Refer to [Configure Container Runtime](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#configure-container-runtime) to configure the `habana` container runtime on each node.

## Install vLLM
1. Start a container with the latest base image:

    ``` bash
    docker run -it --runtime=habana \
        -e HABANA_VISIBLE_DEVICES=all \
        -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
        --cap-add=sys_nice --net=host --ipc=host \
        vault.habana.ai/gaudi-docker/1.21.3/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
    ```

2. Install vLLM：

    ``` bash
    # install vllm
    git clone -b aice/v1.22.0 https://github.com/HabanaAI/vllm-fork
    pip install -r vllm-fork/requirements-hpu.txt
    VLLM_TARGET_DEVICE=hpu pip install -e vllm-fork --no-build-isolation

    # [optional] install vllm-hpu-extension to do calibration and run in fp8
    git clone -b aice/v1.22.0 https://github.com/HabanaAI/vllm-hpu-extension
    pip install -e vllm-hpu-extension --no-build-isolation
    ```

3. If you need use multimodal models like Qwen-VL, GLM-4V, we recommend using Pillow-SIMD instead of Pillow to improve the image processing performance.
To install Pillow-SIMD, run the following:

    ``` bash
    pip uninstall pillow
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
    ``` 

    > We also provide HPU MediaPipe for the image processing for Qwen-VL. Enable it by exporting `USE_HPU_MEDIA=true`. You may enable your models with this feature via referring to the changes in qwen.py.

4. Enter the scripts folder

    ``` bash
    cd scripts
    ```

## Steps to host vLLM service

### 1. Download the model weights
You may download the required model weight files from [HuggingFace](https://huggingface.co/) or [ModelScope](https://www.modelscope.cn/).

```bash
sudo apt install git-lfs
git-lfs install

# Option1: Download from HuggingFace
git clone https://huggingface.co/Qwen/Qwen2-72B-Instruct /models/Qwen2-72B-Instruct
# Option2: Download from ModelScope
git clone https://www.modelscope.cn/Qwen/Qwen2-72B-Instruct /models/Qwen2-72B-Instruct
```

### 2. Start the server
There are some system environment variables which need be set to get the best vLLM performance. We provide the sample script to set the recommended environment variables.

The script file "start_gaudi_vllm_server.sh" is used to start vLLM service. You may execute the command below to check its supported parameters.

``` bash
# to print the help info
bash start_gaudi_vllm_server.sh -h
```

The command output is like below.

```
Start a vLLM server for a huggingface model on Gaudi.

Usage: bash start_gaudi_vllm_server.sh <-w> [-t:m:a:d:q:i:p:o:b:g:u:e:l:c:sf] [-h]
Options:
-w  Weights of the model, str, could be model id in huggingface or local path.
    DO NOT change the model name as some of the parameters depend on it.
-t  tensor-parallel-size for vLLM, int, default=1.
    Also used to set EP size if it's enable by --enable-expert-parallel
-m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None
    Used to select HPUs and to set NUMA accordingly. It's recommended to set
    for cases with 4 or less HPUs.
-a  API server URL, str, 'IP:PORT', default=127.0.0.1:30001
-d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'
    Set to 'fp8' if -q or environment variable 'QUANT_CONFIG' is provided.
-q  Path to the quantization config file, str, default=None
    default=./quantization/<model_name_lower>/maxabs_quant_g2.json for -d 'fp8'
    The environment variable 'QUANT_CONFIG' will override this option.
-x  max-model-len for vllm, int, default=16384
    Make sure the range cover all the possible lengths from the benchmark/client.
-p  Max number of the prefill sequences, int, default=1
    Used to control the max batch size for prefill to balance the TTFT and throughput.
    The default value of 1 is used to optimize the TTFT.
    Set to '' to optimize the throughput for short prompts.
-b  max-num-seqs for vLLM, int, default=128
    Used to control the max batch size for decoding phase.
    It is recommended to set this value according to the 'Maximum concurrency'
    reported by a test run.
-g  max-seq-len-to-capture for vLLM, int, default=8192
    Used to control the maximum batched tokens to be captured in HPUgraph.
    Reduce this value could decrease memory usage, but not smaller than 2048.
-u  gpu-memory-utilization, float, default=0.9
    Used to control the GPU memory utilization. Reduce this value if OOM occurs.
-e  Extra vLLM server parameters, str, default=None
    Extra parameters that will pass to the vLLM server.
-l  Limit of the padding ratio, float, [0.0, 0.5], default=0.25
    The padding strategy ensures that padding_size/bucket_size <= limit.
    Smaller limit values result in more aggressive padding and more buckets.
    Set to 0.5 is equivalent to use the exponential bucketing.
    Set to 0.0 is equivalent to use the linear bucketing without padding limits.
-c  Cache HPU recipe to the specified path, str, default=None
    The recipe cache could be reused to reduce the warmup time.
-s  Skip warmup or not, bool, default=false
    Skip warmup to reduce startup time. Used in debug/dev environment only.
    DO NOT use in production environment.
-f  Enable high-level profiler or not, bool, default=false
-h  Help info
```

Here is a recommended example to start vLLM service on Qwen2-72B-Instruct model with 4 cards. Intel(R) Gaudi(R) module ID 0,1,2,3 are selected, max model length is 16384, data type is BF16 and the vLLM service port is 30001.
The model weight are the standard models files which can be downloaded from [HuggingFace](https://huggingface.co/) or [ModelScope](https://www.modelscope.cn/)

``` bash
bash start_gaudi_vllm_server.sh \
    -w "/models/Qwen2-72B-Instruct" \
    -t 4 \
    -m 0,1,2,3 \
    -a 127.0.0.1:30001 \
    -d bfloat16 \
    -x 16384 \
    -b 128
```

It will take 10 or more minutes to load and warm up the model. After completion, a typical output would be like below. vLLM server is ready at this time.

```
INFO 03-25 09:01:25 launcher.py:27 Route: /v1/score, Methods: POST 
INFO 03-25 09:01:25 launcher.py:27 Route: /v2/rerank, Methods: POST 
INFO 03-25 09:01:25 launcher.py:27 Route: /v2/rerank, Methods: POST 
INFO 03-25 09:01:25 launcher.py:27 Route: /invocations, Methods: POST 
INFO: Started server process [1167] 
INFO: Waiting for application startup. 
INFO: Application startup complete. 
INFO: Uvicorn running on http://127.0.0.1:30001 (Press CTRL+C to quit)
```

### 3. Run the benchmark
You may use these scripts to check the vLLM server inference performance. vLLM benchmark_serving.py file is used. Before running, please change the parameters in the script file, such as vLLM host, port, model weight path and so on.

``` bash
bash benchmark_serving_range.sh # to benchmark with specified input/output ranges, random dataset
bash benchmark_serving_sharegpt.sh # to benchmark with ShareGPT dataset
```

> The max-model-len passed to `start_gaudi_vllm_server.sh` must cover the following benchmark ranges to get expected performance.

> The parameters in the `benchmark_serving_range.sh` and `benchmark_serving_sharegpt.sh` must be modified to match the ones passed to `start_gaudi_vllm_server.sh`.

### 4. Run vLLM with FP8 using INC
Running vLLM with FP8 precision can be achieved using [Intel(R) Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8). To run vLLM with FP8 precision using INC, pass `-d fp8` and specify the path to your bfloat16 or float16 model with `-w <model_path>`. The model will be quantized to FP8 using calibration data obtained from the [FP8 Calibration Procedure](https://github.com/HabanaAI/vllm-hpu-extension/blob/aice/v1.22.0/calibration/README.md).

#### 1. Prepare the dataset
It's recommended to use [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) to do the calibration. We can download it to a local path by

```bash
python3 -m pip install hf_transfer huggingface_hub hf_xet
huggingface-cli download NeelNanda/pile-10k --repo-type dataset
```

or pass the dataset ID.

#### 2. Enter vllm-hpu-extension/calibration folder and do calibration
The calibration steps are integrated to the `calibrate_model.sh`.

``` bash
# to print the help info
bash calibrate_model.sh -h
```

The output help info is like below.

``` bash
Calibrate given MODEL_PATH for FP8 inference

usage: calibrate_model.sh <options>

  -m    - [required] huggingface stub or local directory of the MODEL_PATH
  -d    - [required] path to source dataset (details in README)
  -o    - [required] path to output directory for fp8 measurements
  -b    - batch size to run the measurements at (default: 32)
  -l    - limit number of samples in calibration dataset
  -t    - tensor parallel size to run at (default: 1); NOTE: if t > 8 then we need a multi-node setup
  -r    - rank of unified measurements, it should be smaller than original rank number and should be a factor of the original rank number
  -u    - use expert parallelism (default: False), expert parallelism unification rule is unique, card 1 expert measurement will be extended to card 0 if unified to x from 2x cards number
  -x    - expand measurement files to specific world size (default: not set)
  -e    - set this flag to enable enforce_eager execution
```

> [!IMPORTANT]
> **For Mixture of Experts (MoE) models**: The `-u` must be passed to enable Expert Parallelism (EP) except for [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct).

##### 1. Do the calibration for BF16 model
For the models which only have BF16 precision, like Qwen2.5-72B-Insturct, it can be calibrated on 4 Gaudi cards with the command below. The measured data will be saved into the `quantization` folder. With this measured data, vLLM can run this model with FP8 precision with 2 or 4 Gaudi cards. 

```bash
cd vllm-hpu-extension/calibration
./calibrate_model.sh \
     -m /models/Qwen2.5-72B-Instruct \
     -d NeelNanda/pile-10k \
     -o quantization \
     -t 4 \
     -r 2
```

##### 2. Do the calibration for FP8 model
For the new models which have the FP8 precision, like Qwen3-235B-A22B-FP8 or GLM-4.5-Air-FP8, the calibration is also required to optimize the performance. The calibration may directly be done on the FP8 model. The example below is to do the calibration on Qwen3-256B-A223-FP8 model with 8 cards. Then vLLM can run this model with FP8 precision with 4 or 8 Gaudi cards. 

``` bash
bash calibrate_model.sh \
     -m /models/Qwen3-235B-A22B-FP8 \
     -d NeelNanda/pile-10k \
     -o quantization \
     -t 8 -r 4 -u
```

If only 4 Gaudi cards are available, the Qwen3-235B-A22B-FP8 calibration can also be done with 4 cards. Then vLLM can run this model with FP8 precision with only 4 Gaudi cards. 

``` bash
bash calibrate_model.sh \
     -m /models/Qwen3-235B-A22B-FP8 \
     -d NeelNanda/pile-10k \
     -o quantization \
     -t 4 -u
```

##### 3. Do the calibration for pipeline parallelism mode
The `-x <TP_SIZE_WITH_PP>` must set to run the model with pipeline parallelism (PP), with the `TP_SIZE_WITH_PP` means the TP size when PP is enabled. Take GLM-4.5-FP8 with TP=4 and PP=2 as an example:

``` bash
bash calibrate_model.sh \
     -m /models/GLM-4.5-FP8 \
     -d NeelNanda/pile-10k \
     -o quantization \
     -t 8 -x 4 -u
```

> [!TIP]
> **For models with fp8 weights**: Gaudi2 supports `fp8_e4m3fnuz` instead of the `fp8_e4m3fn`, so far the fp8_e4m3fn weights will be loaded to the host memory first and be converted to `fp8_e4m3fnuz` before be moved to the HPU. The conversion is done with CPU and could be time consuming. Another option is to convert the `fp8_e4m3fn` weights offline with the script [convert_weights_for_gaudi2.py](https://github.com/HabanaAI/vllm-hpu-extension/blob/aice/v1.22.0/scripts/convert_weights_for_gaudi2.py) and set the environment variable `VLLM_HPU_CONVERT_TO_FP8UZ=false` for the benchmark and calibration sessions.

> [!WARNING]
> Do not set `VLLM_HPU_CONVERT_TO_FP8UZ=false` when using the original `fp8_e4m3fn` weights, and **do not forget** to set `VLLM_HPU_CONVERT_TO_FP8UZ=false` when using the converted `fp8_e4m3fnuz` weights.

#### 3. Make the Quantization folder
Create a quantization folder at the same level as start_gaudi_vllm_server.sh.

```bash
mkdir quantization
```

Copy the converted quantization files into the quantization folder:

```bash
cp -r vllm-hpu-extension/calibration/quantization/* quantization/
```

Note: Ensure that the subdirectory names under quantization match the modelPath suffixes in models.conf.
```bash
ls vllm-fork/scripts/quantization/
qwen3-235b-a22b-fp8  qwen2.5-72b-instruct
```


#### 4. Start vLLM service on Qwen2.5-72B-Instruct model with FP8 precision.
It will take much more time to do warm-up with FP8 precision. Suggest creating the warm-up cache files to accelerate the warm-up for next time.

```bash
bash start_gaudi_vllm_server.sh \
    -w "/models/Qwen2.5-72B-Instruct" \
    -t 2 \
    -m 0,1 \
    -a "127.0.0.1:30001" \
    -d fp8 \
    -b 128 \
    -x 16384 \
    -c /vllm_cache/Qwen2.5-32B-Instruct/
```

## Steps to run offline benchmark
 The script file "benchmark_throughput.sh" is used to run vLLM under offline mode. You may execute the command below to check its supported parameters.

``` bash
# to print the help info
bash benchmark_throughput.sh -h 
```

The command output is like below.

```
Benchmark vLLM throughput for a huggingface model on Gaudi.

Syntax: bash benchmark_throughput.sh <-w> [-t:m:d:q:i:p:o:b:r:j:n:g:u:e:l:c:sf] [-h]
options:
-w  Weights of the model, str, could be model id in huggingface or local path.
    DO NOT change the model name as some of the parameters depend on it.
-t  tensor-parallel-size for vLLM, int, default=1.
    Also used to set EP size if it's enable by --enable-expert-parallel
-m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None
    Used to select HPUs and to set NUMA accordingly. It's recommended to set
    for cases with 4 or less HPUs.
-d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'
    Set to 'fp8' if -q or environment variable 'QUANT_CONFIG' is provided.
-q  Path to the quantization config file, str, default=None
    default=./quantization/<model_name_lower>/maxabs_quant_g2.json for -d 'fp8'
    The environment variable 'QUANT_CONFIG' will override this option.
-i  Input length, int, default=1024
-p  Max number of the prefill sequences, int, default=1
    Used to control the max batch size for prefill to balance the TTFT and throughput.
    The default value of 1 is used to optimize the TTFT.
    Set to '' to optimize the throughput for short prompts.
-o  Output length, int, default=512
-b  max-num-seqs for vLLM, int, default=128
    Used to control the max batch size for decoding phase.
    It is recommended to set this value according to the 'Maximum concurrency'
    reported by a test run.
-r  random-range-ratio for benchmark_throughput.py, float, default=0.0
    The result range is [length * (1 - range_ratio), length * (1 + range_ratio)].
-j  Json path of the ShareGPT dataset, str, default=None
    set -j <sharegpt json path> will override -i, -o and -r
-n  Number of prompts, int, default=max_num_seqs*4
-g  max-seq-len-to-capture for vLLM, int, default=8192
    Used to control the maximum batched tokens to be captured in HPUgraph.
    Reduce this value could decrease memory usage, but not smaller than 2048.
-u  gpu-memory-utilization, float, default=0.9
    Used to control the GPU memory utilization. Reduce this value if OOM occurs.
-e  Extra vLLM server parameters, str, default=None
    Extra parameters passed to benchmarks/benchmark_throughput.py and vLLM engine.
-l  Limit of the padding ratio, float, [0.0, 0.5], default=0.25
    The padding strategy ensures that padding_size/bucket_size <= limit.
    Smaller limit values result in more aggressive padding and more buckets.
    Set to 0.5 is equivalent to use the exponential bucketing.
    Set to 0.0 is equivalent to use the linear bucketing without padding limits.
-c  Cache HPU recipe to the specified path, str, default=None
    The recipe cache could be reused to reduce the warmup time.
-s  Skip warmup or not, bool, default=false
    Skip warmup to reduce startup time. Used in debug/dev environment only.
    DO NOT use in production environment.
-f  Enable high-level profiler or not, bool, default=false
-h  Help info
```

Run offline benchmark with the ShareGPT dataset

``` bash
# an example to benchmark llama2-7b-chat with the sharegpt dataset
bash benchmark_throughput.sh -w "/models/Llama-2-7b-chat-hf" -j <sharegpt json>
```

Run offline benchmark with the random dataset, input length is 1024 and output length is 512.

``` bash
# an example to benchmark llama2-7b-chat with the fixed input length of 1024, output length of 512 and max_num_seqs of 64
bash benchmark_throughput.sh -w "/models/Llama-2-7b-chat-hf" -i 1024 -o 512 -b 64
```

## Handling of the long warm-up time
We can cache the recipe to disk and skip warm-up during the benchmark to save warm-up time. So, our customers and ourselves don’t have to wait for the long warm-up time, and we could get the best performance of vLLM on Gaudi.
### set the cache files path for online serving
Then the second warm-up can use the cached files to accelerate the warm-up. If the vLLM version, max_num_seqs, input range or output range is changed, the warm-up will be re-done.
The extra parameter is like "-c [cache_files_path]" and the full example command is like below.

``` bash
bash start_gaudi_vllm_server.sh \
    -w "/models/Qwen2-72B-Instruct" \
    -t 4 \
    -m 0,1,2,3 \
    -a "127.0.0.1:30001" \
    -d bfloat16 \
    -b 128 \
    -x 16384 \
    -c /data/Qwen2-72B-cache
```

### skip warm-up for online serving
You may and the parameter "-s" to skip the warm-up. vLLM server can be started very quickly. The warm-up is done during the inference serving and the performance may be impacted a little.

``` bash
bash start_gaudi_vllm_server.sh \
    -w "/models/Qwen2-72B-Instruct" \
    -t 4 \
    -m 0,1,2,3 \
    -a "127.0.0.1:30001" \
    -d bfloat16 \
    -b 128 \
    -x 16384 \
    -s
```

### For offline benchmark:
1. Run `benchmark_throughput.sh` with `-c <recipe path>` and without `-s` to create and save the recipe cache.
2. Release the cached recipe files along with the vllm code to the customer.
3. Run `benchmark_throughput.sh` with `-c <recipe path>` and with `-s` to skip warm-up.

> We can also skip warm-up at the 1st step and run the benchmark twice, one for warm-up and the other one for collecting of the performance data. This approach has the risk of some missing warm-up bucketing as the scheduling of the two rounds of benchmark may not be exactly the same.

## FAQs
### Handling of the accuracy issue
We found some models may have low lm_eval score when running with bf16 format. Please try to set `VLLM_FP32_SOFTMAX=true` and `VLLM_PROMPT_USE_FUSEDSDPA=false` to improve the accuracy.

> The models listed in the [Supported Configurations](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#supported-configurations) don't have this accuracy issue.

### Handling of not enough KV cache space warning
When there are warnings of "Sequence group xxx is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space.", please try to decrease the vLLM server "max_num_seqs"  or benchmarrk_serving.py "--max-concurrency" value, e.g. to 64. This warning can happen when running benchmark_throughtput with fixed input/output.

### About FusedSDPA
[FusedSDPA](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html#using-fused-scaled-dot-product-attention-fusedsdpa) could be used in vLLM prompt stage and it’s enabled by default to save device memory especially for long prompts. While it’s not compatible with Alibi yet, please disable it for models with Alibi.

### Handling of the long sequence request
For the long input/output cases, such as 20k/0.5k input/output, please modify the model length to be larger than `max(input_length) + max(output_length)`. For example, set `max_position_embeddings=32768` in the `config.json` file of LLaMA models.

### About fp8 benchmark
Please follow the [FP8 Calibration Procedure](https://github.com/HabanaAI/vllm-hpu-extension/tree/main/calibration#fp8-calibration-procedure) to get the quantization data before running of the benchmarks.

## Tuning vLLM on Gaudi
### Setup the bucketing
The `set_bucketing()` from `utils.sh` is used to setup the bucketing parameters according to the max_model_len, max_num_batched_tokens and max_num_seqs etc. The settings could also be override by manually set the corresponding ENVs. Please refer to [bucketing mechanism](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#bucketing-mechanism) for more details.

### Tuning the device memory usage
The environment variables `VLLM_GRAPH_RESERVED_MEM`, `VLLM_GRAPH_PROMPT_RATIO` and `VLLM_GPU_MEMORY_UTILIZATION` could be used to tune the detailed usage of device memory, please refer to [HPU Graph capture](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#hpu-graph-capture) for more details.

### Setup NUMA
vLLM is a CPU-heavy workload and the host processes are better to bound to the CPU cores and memory node of the selected devices if they are on the same NUMA node. The `set_numactl()` from `utils.sh` is used to setup the NUMA bounding for the module_id specified by `-m` according to the output of `hl-smi topo -c -N`. The script "start_gaudi_vllm_server.sh" has integrate "set_numactl()" to use the right NUMA node setting based on the module IDs.

``` {.}
modID   CPU Affinity    NUMA Affinity    
-----   ------------    -------------    
0       0-39, 80-119    0  
1       0-39, 80-119    0  
2       0-39, 80-119    0  
3       0-39, 80-119    0  
4       40-79, 120-159          1  
5       40-79, 120-159          1  
6       40-79, 120-159          1  
7       40-79, 120-159          1
```

### Profile the LLM engine
The following 4 ENVs are used to control the device profiling:
* `VLLM_ENGINE_PROFILER_ENABLED`, set to `true` to enable device profiler.
* `VLLM_ENGINE_PROFILER_WARMUP_STEPS`, number of steps to ignore for profiling.
* `VLLM_ENGINE_PROFILER_STEPS`, number of steps to capture for profiling.
* `VLLM_ENGINE_PROFILER_REPEAT`, number of cycles for (warmup + profile).

> Please refer to [torch.profiler.schedule](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) for more details about the profiler schedule arguments.

> The `step` in profiling means a step of the LLM engine, exclude the profile and warmup run in `HabanaModelRunner`.

> Please use the `-f` flag or `export VLLM_PROFILER_ENABLED=True` to enable the high-level vLLM profile and to choose the preferred steps to profile.

# Releases
## aice/v1.22.0
vllm-fork:
https://github.com/HabanaAI/vllm-fork/tree/aice/v1.22.0
vllm-hpu-extension:
https://github.com/HabanaAI/vllm-hpu-extension/tree/aice/v1.22.0
## Valided models
* DeepSeek-R1-Distill-Llama-70B (bf16 and fp8)
* DeepSeek-R1-Distill-Qwen-32B (bf16 and fp8)
* DeepSeek-R1-Distill-Qwen-14B (bf16 and fp8)
* DeepSeek-R1-Distill-Qwen-7B (bf16 and fp8)
* DeepSeek-R1-Distill-Llama-8B (bf16 and fp8)
* Qwen3-32B (bf16 and fp8)
* Qwen3-14B (bf16 and fp8)
* Qwen3-235B-A22B (bf16)
* Qwen3-30B-A3B (bf16 and fp8)
* Meta-Llama-3-70B-Instruct (bf16)
* Meta-Llama-3-8B-Instruct (bf16)
* Llama-3.1-70B-Instruct (bf16)
* Qwen2.5-72B-Instruct (bf16)
* Qwen2.5-32B-Instruct (bf16)
* Qwen2.5-14B-Instruct (bf16)
* Qwen2.5-7B-Instruct (bf16)
* Qwen2.5-3B-Instruct (bf16)
* Qwen2.5-1.5B-Instruct (bf16)
* QwQ-32B (bf16)
* Llama4 (bf16 and fp8)
* GLM-4.5 (bf16 and fp8)
* GLM-4.5-Air (bf16 and fp8)
* MiniMax-M2 (fp8)
* Qwen3-Next-80B-A3B (bf16 and fp8)
* multimodal models:
  - Qwen2.5 Omni
  - Qwen2-VL-7B-Instruct
  - Qwen3-VL
  - InternVL3.5
  
