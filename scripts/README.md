# Environment setup

## Hardware Requirements
This is used to set up vLLM service on Intel(R) Gaudi(R) accelerator. Please refer to [Hardware and Network Requirements](https://docs.habana.ai/en/latest/Installation_Guide/Platform_Readiness.html#) to check your hardware readiness. 

## Software Requirements
* The supported OS are in [Supported Configurations and Components](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html#support-matrix)
* Refer to [Driver and Software Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install the Intel(R) Gaudi(R) driver and software stack (>= 1.20.1) on each node. Make sure `habanalabs-container-runtime` is installed.
* Refer to [Firmware Upgrade](https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html) to upgrade the Gaudi(R) firmware to 1.20.1 version on each node.
* Refer to [Configure Container Runtime](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#configure-container-runtime) to configure the `habana` container runtime on each node.


## Install vLLM
1. Start a container with the latest base image:
``` bash
docker run -it --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice --net=host --ipc=host \
    vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
 ```

2. Install vLLM：
``` bash
git clone -b aice/v1.20.1 https://github.com/HabanaAI/vllm-fork
VLLM_TARGET_DEVICE=hpu pip install -e vllm-fork
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
### 1. Start the server
There are some system environment variables which need be set to get the best vLLM performance. We provide the sample script to set the recommended environment variables.

The script file "start_gaudi_vllm_server.sh" is used to start vLLM service. You may execute the command below to check its supported parameters.
``` bash
# to print the help info
bash start_gaudi_vllm_server.sh -h
```

The command output is like below. 
```
Start vllm server for a huggingface model on Gaudi.

Syntax: bash start_gaudi_vllm_server.sh <-w> [-n:m:u:p:d:i:o:t:l:b:e:c:sfza] [-h]
options:
w  Weights of the model, could be model id in huggingface or local path
n  Number of HPU to use, [1-8], default=1
m  Module IDs of the HPUs to use, [0-7], default=None
u  URL of the server, str, default=127.0.0.1
p  Port number for the server, int, default=30001
d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'
i  Input range, str, format='input_min,input_max', default='4,1024'
o  Output range, str, format='output_min,output_max', default='4,2048'
t  max_num_batched_tokens for vllm, int, default=8192
l  max_model_len for vllm, int, default=4096
b  max_num_seqs for vllm, int, default=128
e  number of scheduler steps, int, default=1
c  Cache HPU recipe to the specified path, str, default=None
s  Skip warmup or not, bool, default=false
f  Enable profiling or not, bool, default=false
z  Disable zero-padding, bool, default=false
a  Disable FusedFSDPA, bool, default=false
h  Help info
```

Here is a recommended example to start vLLM service on Qwen2-72B-Instruct model with 4 cards. Intel(R) Gaudi(R) module ID 0,1,2,3 are selected, input length range is 800 ~ 1024, output length range is 400 ~ 512, data type is BF16 and the vLLM service port is 30001. 
The model weight are the standard models files which can be downloaded from [HuggingFace](https://huggingface.co/) or [ModelScope](https://www.modelscope.cn/) 
``` bash
bash start_gaudi_vllm_server.sh -w "/models/Qwen2-72B-Instruct" \
    -n 4 \
    -m 0,1,2,3 \ 
    -b 128 \
    -i 800,1024 \
    -o 400,512 \
    -l 4096 \
    -t 8192 \
    -d bfloat16 \
    -p 30001
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

### 2. Run the benchmark
You may use these scripts to check the vLLM server inference performance. vLLM benchmark_serving.py file is used. 
``` bash
bash benchmark_serving_range.sh # to benchmark with specified input/output ranges, random dataset
bash benchmark_serving_sharegpt.sh # to benchmark with ShareGPT dataset
```

> The input/output ranges passed to `start_gaudi_vllm_server.sh` should cover the following benchmark ranges to get expected performance.

> The parameters in the `benchmark_serving_range.sh` and `benchmark_serving_sharegpt.sh` must be modified to match the ones passed to `start_gaudi_vllm_server.sh`.

### 3. Host vLLM service with FP8 precision
Running vLLM service with FP8 precision is achieved by using [Intel(R) Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package. For the detailed info, you may check [FP8 Calibration Procedure](https://github.com/HabanaAI/vllm-hpu-extension/blob/v1.21.0/calibration/README.md).
#### 1. Copy open_orca_gpt4_tokenized_llama.calibration_1000.pkl to vllm-hpu-extension/calibration folder
```bash
gzip -dk Gaudi-fp8-calibration/open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz
cp Gaudi-fp8-calibration/open_orca_gpt4_tokenized_llama.calibration_1000.pkl vllm-hpu-extension/calibration
```

#### 2. Enter vllm-hpu-externsion/calibration folder and do calibration
The example below is to calibrate Qwen2.5-72B-Instruct model for 2 Gaudi cards. The quantization files are copied into "quantization" folder.
```bash
cd vllm-hpu-extension/calibration
MODEL=/models/Qwen2.5-72B-Instruct
HPU_SIZE=2
./calibrate_model.sh -m $MODEL -d open_orca_gpt4_tokenized_llama.calibration_1000.pkl  -o quantization -t $HPU_SIZE
```

#### 3. Make the Quantization folder
Create a quantization folder at the same level as start_gaudi_vllm_server.sh.
```bash
mkdir quantization
```
Copy the converted quantization files into the quantization folder:
```bash
cp -r converted_quantization/* quantization/
```
Note: Ensure that the subdirectory names under quantization match the modelPath suffixes in models.conf. 
#### 4. Start vLLM service on Qwen2.5-72B-Instruct model with FP8 precision.
It will take much more time to do warm-up with FP8 precision. Suggest creating the warm-up cache files to accelerate the warm-up for next time. 
```bash
bash start_gaudi_vllm_server.sh -w "/models/Qwen2.5-72B-Instruct" \
    -n 2 \
    -m 0,1 \ 
    -b 128 \
    -i 800,1024 \
    -o 400,512 \
    -l 4096 \
    -t 8192 \
    -d fp8 \
    -p 30001 \
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
Benchmark vllm throughput for a huggingface model on Gaudi.

Syntax: bash benchmark_throughput.sh <-w> [-n:m:d:i:o:r:j:t:l:b:c:sfza] [-h]
options:
w  Weights of the model, could be model id in huggingface or local path
n  Number of HPU to use, [1-8], default=1
m  Module IDs of the HPUs to use, [0-7], default=None
d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'
i  Input length, int, default=1024
o  Output length, int, default=512
r  Ratio for min input/output length to generate an uniform distributed input/out length, float, default=1.0
j  Json path of the ShareGPT dataset, str, default=None
t  max_num_batched_tokens for vllm, int, default=8192
l  max_model_len for vllm, int, default=4096
b  max_num_seqs for vllm, int, default=128
p  number of prompts, int, default=1000
e  number of scheduler steps, int, default=1
c  Cache HPU recipe to the specified path, str, default=None
s  Skip warmup or not, bool, default=false
f  Enable profiling or not, bool, default=false
z  Disable zero-padding, bool, default=false
a  Disable FusedFSDPA, bool, default=false
h  Help info

Note: set -j <sharegpt json path> will override -i, -o and -r
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
bash start_gaudi_vllm_server.sh -w "/models/Qwen2-72B-Instruct" \
    -n 4 \
    -m 0,1,2,3 \ 
    -b 128 \
    -i 800,1024 \
    -o 400,512 \
    -l 4096 \
    -t 8192 \
    -d bfloat16 \
    -c /data/Qwen2-72B-cache \
    -p 30001
```
### skip warm-up for online serving
You may and the parameter "-s" to skip the warm-up. vLLM server can be started very quickly. The warm-up is done during the inference serving and the performance may be impacted a little. 
``` bash
bash start_gaudi_vllm_server.sh -w "/models/Qwen2-72B-Instruct" \
    -n 4 \
    -m 0,1,2,3 \ 
    -b 128 \
    -i 800,1024 \
    -o 400,512 \
    -l 4096 \
    -t 8192 \
    -d bfloat16 \
    -s \
    -p 30001
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
The `set_bucketing()` from `utils.sh` is used to setup the bucketing parameters according to the input/output range, max_num_batched_tokens and max_num_seqs etc. The settings could also be override by manually set the corresponding ENVs. Please refer to [bucketing mechanism](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#bucketing-mechanism) for more details.

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
- `VLLM_ENGINE_PROFILER_ENABLED`, set to `true` to enable device profiler.
- `VLLM_ENGINE_PROFILER_WARMUP_STEPS`, number of steps to ignore for profiling.
- `VLLM_ENGINE_PROFILER_STEPS`, number of steps to capture for profiling.
- `VLLM_ENGINE_PROFILER_REPEAT`, number of cycles for (warmup + profile).

> Please refer to [torch.profiler.schedule](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) for more deatils about the profiler schedule arguments.

> The `step` in profiling means a step of the LLM engine, exclude the profile and warmup run in `HabanaModelRunner`.

> Please use the `-f` flag or `export VLLM_PROFILER_ENABLED=True` to enable the high-level vLLM profile and to choose the preferred steps to profile.


