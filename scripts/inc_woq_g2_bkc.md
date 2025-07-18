## 0. Prerequisites

Please use the firmware and software stack mentioned [here](https://github.com/HabanaAI/vllm-fork/tree/deepseek_r1/scripts/quickstart).

> [!NOTE]
> The steps below have been verified on G2D and [Synapse v1.21.1](https://docs.habana.ai/en/v1.21.1/Release_Notes/GAUDI_Release_Notes.html#new-features-v1-21-1).

## 1. Installation

- vLLM

```bash
git clone -b deepseek_r1 https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
pip install -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation
```

- INC

```bash
pip install git+https://github.com/intel/neural-compressor.git@r1-woq
```

## 2. Convert the model files

```bash
cd vllm-fork
python ./scripts/convert_for_g2.py -i /path/to/official/model -o /path/to/converted/model/
```

This script 1) converts official model weights from `torch.float8_e4m3fn` format to `torch.float8_e4m3fnuz` format, and 2) copies other JSON and Python files into the target path.

## 3. Benchmark

> [!NOTE]
> For INC WoQ requantization, make sure to:
> 1) Specify the path to the measurement files in the quantization configuration JSON file.
>
> 2) Set the `QUANT_CONFIG` environment variable to point to this configuration file.
>

### Configure the Measurement Statistics Results

The environment variable `INC_MEASUREMENT_DUMP_PATH_PREFIX` specifies the root directory where measurement statistics were saved.
The final path is constructed by joining this root directory with the `dump_stats_path` defined in the quantization JSON file specified by the `QUANT_CONFIG` environment variable.

#### Example

If we download the measurements to `/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache`, we got below files:

```bash
user:vllm-fork$ pwd
/path/to/vllm-fork
user:vllm-fork$ ls -l  ./scripts/nc_workspace_measure_kvcache
-rw-r--r-- 1 user Software-SG 1949230 May 15 08:05 inc_measure_output_hooks_maxabs_0_8.json
-rw-r--r-- 1 user Software-SG  254451 May 15 08:05 inc_measure_output_hooks_maxabs_0_8_mod_list.json
-rw-r--r-- 1 user Software-SG 1044888 May 15 08:05 inc_measure_output_hooks_maxabs_0_8.npz
...
```

Then, we export `INC_MEASUREMENT_DUMP_PATH_PREFIX=/path/to/vllm-fork`, and INC will parse the full as below:

```
dump_stats_path (from config): "scripts/nc_workspace_measure_kvcache/inc_measure_output"
Resulting full path: "/path/to/vllm-fork/scripts/nc_workspace_measure_kvcache/inc_measure_output_hooks_maxabs_0_8.npz"
```

> [!CAUTION]
> Before running the benchmark, update the following variables in the single_16k_len_inc.sh script:
> - `model_path`
> - `QUANT_CONFIG`
> - `INC_MEASUREMENT_DUMP_PATH_PREFIX`

### 3.1 BF16 KV + Per-Channel Quantization

- Get calibration files

```bash
cd vllm-fork
huggingface-cli download Yi30/miki-k2-pile-g2-tp16-2nd-0717 --local-dir ./scripts/nc_workspace_measure_kvcache
```

- Running the Benchmark

AS Kimi-K2-Instruct requires at least two nodes for serving, please following the [Multi-Node Setup and Serving Deployment](https://github.com/HabanaAI/vllm-fork/tree/deepseek_r1/scripts/quickstart#multi-node-setup-and-serving-deployment) section to start the serving process.
