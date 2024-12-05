# vLLM with Intel® Gaudi® AI Accelerators

This README provides instructions on running vLLM with Intel Gaudi devices.

# Requirements and Installation

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the execution environment. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).

## Requirements

- OS: Ubuntu 22.04 LTS
- Python: 3.10
- Intel Gaudi accelerator
- Intel Gaudi software version 1.18.0

## Quick start using Dockerfile
```
$ docker build -f Dockerfile.hpu -t vllm-hpu-env  .
$ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --rm vllm-hpu-env
```

> [!TIP]
> If you're facing the following error: `docker: Error response from daemon: Unknown runtime specified habana.`, please refer to "Install optional packages" section of [Install Driver and Software](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html#install-driver-and-software) and "Configure Container Runtime" section of [Docker Installation] (https://docs.habana.ai/en/latest/Installation_Guide/Installation_Methods/Docker_Installation.html#configure-container-runtime). Make sure you have ``habanalabs-container-runtime`` package installed and that ``habana`` container runtime is registered.


## Build from source

### Environment verification
To verify that the Intel Gaudi software was correctly installed, run:

```{.console}
$ hl-smi # verify that hl-smi is in your PATH and each Gaudi accelerator is visible
$ apt list --installed | grep habana # verify that habanalabs-firmware-tools, habanalabs-graph, habanalabs-rdma-core, habanalabs-thunk and habanalabs-container-runtime are installed
$ pip list | grep habana # verify that habana-torch-plugin, habana-torch-dataloader, habana-pyhlml and habana-media-loader are installed
$ pip list | grep neural # verify that neural-compressor is installed
```

Refer to [System Verification and Final Tests](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html) for more details.

### Run Docker Image

It is highly recommended to use the latest Docker image from Intel Gaudi vault. Refer to the [Intel Gaudi documentation](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers) for more details.

Use the following commands to run a Docker image:

```{.console}
$ docker pull vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest
$ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest
```

### Build and Install vLLM-fork

Currently, the latest features and performance optimizations are developed in Gaudi's [vLLM-fork](https://github.com/HabanaAI/vllm-fork) and we periodically upstream them to vLLM main repo. To install latest [HabanaAI/vLLM-fork](https://github.com/HabanaAI/vllm-fork), run the following:

```{.console}
$ git clone https://github.com/HabanaAI/vllm-fork.git
$ cd vllm-fork
$ git checkout habana_main
$ pip install -r requirements-hpu.txt
$ python setup.py develop
```

# Supported Features
| **Feature** 	| **Description** 	| **References** 	|
|---	|---	|---	|
| Offline batched inference 	| Offline inference using LLM class from vLLM Python API 	| [Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#offline-batched-inference)<br>[Example](https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference.html) 	|
| Online inference via OpenAI-Compatible Server 	| Online inference using HTTP server that implements OpenAI Chat and Completions API 	| [Documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)<br>[Example](https://docs.vllm.ai/en/stable/getting_started/examples/openai_chat_completion_client.html) 	|
| HPU autodetection 	| HPU users do not need to specify the target platform, it will be detected automatically upon vLLM startup 	| N/A 	|
| Paged KV cache with algorithms enabled for Intel Gaudi accelerators 	| vLLM HPU backend contains a custom Paged Attention and cache operators implementations optimized for Gaudi devices. 	| N/A 	|
| Custom Intel Gaudi operator implementations 	| vLLM HPU backend provides optimized implementations of operators such as prefill attention, Root Mean Square Layer Normalization, Rotary Positional Encoding. 	| N/A 	|
| Tensor parallel inference (single-node multi-HPU) 	| vLLM HPU backend support multi-HPU inference across a single node with tensor parallelism with Ray and HCCL. 	| [Documentation](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)<br>[Example](https://docs.ray.io/en/latest/serve/tutorials/vllm-example.html)<br>[HCCL reference](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/index.html) 	|
| Inference with HPU Graphs 	| vLLM HPU backend uses HPU Graphs by default for optimal performance. When HPU Graphs are enabled, execution graphs will be recorded ahead of time, to be later replayed during inference, significantly reducing host overheads. 	| [Documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)<br>[vLLM HPU backend execution modes](https://docs.vllm.ai/en/stable/getting_started/gaudi-installation.html#execution-modes)<br>[Optimization guide](https://docs.vllm.ai/en/latest/getting_started/gaudi-installation.html#hpu-graph-capture) 	|
| Inference with torch.compile (experimental) 	| vLLM HPU backend experimentally supports inference with torch.compile. 	| [vLLM HPU backend execution modes](https://docs.vllm.ai/en/stable/getting_started/gaudi-installation.html#execution-modes) 	|
| Attention with Linear Biases (ALiBi) 	| vLLM HPU backend supports models utilizing Attention with Linear Biases (ALiBi) such as mpt-7b. 	| [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html) 	|
| INC quantization 	| vLLM HPU backend supports FP8 model and KV cache quantization and calibration with Intel Neural Compressor (INC). 	| [Documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html) 	|
| LoRA/MultiLoRA support 	| vLLM HPU backend includes support for LoRA and MultiLoRA on supported models. 	| [Documentation](https://docs.vllm.ai/en/stable/models/lora.html)<br>[Example](https://docs.vllm.ai/en/stable/getting_started/examples/multilora_inference.html)<br>[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html) 	|
| Multi-step scheduling support 	| vLLM HPU backend includes multi-step scheduling support for host overhead reduction, configurable by standard `--num-scheduler-seqs` parameter.  	| [Feature RFC](https://github.com/vllm-project/vllm/issues/6854) 	|
| Automatic prefix caching (experimental) 	| vLLM HPU backend includes automatic prefix caching (APC) support for more efficient prefills, configurable by standard `--enable-prefix-caching` parameter. 	| [Documentation](https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html)<br>[Details](https://docs.vllm.ai/en/stable/automatic_prefix_caching/details.html) 	|
| Speculative decoding (experimental) 	| vLLM HPU backend includes experimental speculative decoding support for improving inter-token latency in some scenarios, configurabie via standard `--speculative_model` and `--num_speculative_tokens` parameters. 	| [Documentation](https://docs.vllm.ai/en/latest/models/spec_decode.html)<br>[Example](https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference_mlpspeculator.html) 	|

# Unsupported Features

- Beam search
- AWQ quantization
- Prefill chunking (mixed-batch inferencing)

# Supported Configurations

The following configurations have been validated to be function with Gaudi2 devices. Configurations that are not listed may or may not work.

- [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b) with tensor parallelism on 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) with tensor parallelism on 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) with tensor parallelism on 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) with tensor parallelism on 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B) with tensor parallelism on 8x HPU, BF16 datatype with random or greedy sampling
- [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) with tensor parallelism on 8x HPU, BF16 datatype with random or greedy sampling
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) on single HPU or with tensor parallelism on 2x HPU, BF16 datatype with random or greedy sampling
- [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) with tensor parallelism on 2x HPU, BF16 datatype with random or greedy sampling

# Performance Tuning

## Execution modes

Currently in vLLM for HPU we support four execution modes, depending on selected HPU PyTorch Bridge backend (via `PT_HPU_LAZY_MODE` environment variable), and `--enforce-eager` flag.

| `PT_HPU_LAZY_MODE` | `enforce_eager` | execution mode     |
| ------------------ | --------------- | ------------------ |
| 0                  | 0               | torch.compile      |
| 0                  | 1               | PyTorch eager mode |
| 1                  | 0               | HPU Graphs         |
| 1                  | 1               | PyTorch lazy mode  |

> [!WARNING] 
> In 1.18.0, all modes utilizing `PT_HPU_LAZY_MODE=0` are highly experimental and should be only used for validating functional correctness. Their performance will be improved in the next releases. For obtaining the best performance in 1.18.0, please use HPU Graphs, or PyTorch lazy mode.

## Bucketing mechanism

Intel Gaudi accelerators work best when operating on models with fixed tensor shapes. [Intel Gaudi Graph Compiler](https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html#graph-compiler-and-runtime) is responsible for generating optimized binary code that implements the given model topology on Gaudi. In its default configuration, the produced binary code may be heavily dependent on input and output tensor shapes, and can require graph recompilation when encountering differently shaped tensors within the same topology. While the resulting binaries utilize Gaudi efficiently, the compilation itself may introduce a noticeable overhead in end-to-end execution. In a dynamic inference serving scenario, there is a need to minimize the number of graph compilations and reduce the risk of graph compilation occurring during server runtime. Currently it is achieved by "bucketing" model's forward pass across two dimensions - `batch_size` and `sequence_length`.

> [!NOTE]
> Bucketing allows us to reduce the number of required graphs significantly, but it does not handle any graph compilation and device code generation - this is done in warmup and HPUGraph capture phase.

Bucketing ranges are determined with 3 parameters - `min`, `step` and `max`. They can be set separately for prompt and decode phase, and for batch size and sequence length dimension. These parameters can be observed in logs during vLLM startup:

```{.}
INFO 08-01 21:37:59 hpu_model_runner.py:493] Prompt bucket config (min, step, max_warmup) bs:[1, 32, 4], seq:[128, 128, 1024]
INFO 08-01 21:37:59 hpu_model_runner.py:499] Generated 24 prompt buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 08-01 21:37:59 hpu_model_runner.py:504] Decode bucket config (min, step, max_warmup) bs:[1, 128, 4], seq:[128, 128, 2048]
INFO 08-01 21:37:59 hpu_model_runner.py:509] Generated 48 decode buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
```

`min` determines the lowest value of the bucket. `step` determines the interval between buckets, and `max` determines the upper bound of the bucket. Furthermore, interval between `min` and `step` has special handling - `min` gets multiplied by consecutive powers of two, until `step` gets reached. We call this the ramp-up phase and it is used for handling lower batch sizes with minimum wastage, while allowing larger padding on larger batch sizes.

Example (with ramp-up)

```{.}
min = 2, step = 32, max = 64
=> ramp_up = (2, 4, 8, 16)
=> stable = (32, 64)
=> buckets = ramp_up + stable => (2, 4, 8, 16, 32, 64)
```

Example (without ramp-up)

```{.}
min = 128, step = 128, max = 512
=> ramp_up = ()
=> stable = (128, 256, 384, 512)
=> buckets = ramp_up + stable => (128, 256, 384, 512)
```

In the logged scenario, 24 buckets were generated for prompt (prefill) runs, and 48 buckets for decode runs. Each bucket corresponds to a separate optimized device binary for a given model with specified tensor shapes. Whenever a batch of requests is processed, it is padded across batch and sequence length dimension to the smallest possible bucket.

> [!WARNING] 
> If a request exceeds maximum bucket size in any dimension, it will be processed without padding, and its processing may require a graph compilation, potentially significantly increasing end-to-end latency. The boundaries of the buckets are user-configurable via environment variables, and upper bucket boundaries can be increased to avoid such scenario.

As an example, if a request of 3 sequences, with max sequence length of 412 comes in to an idle vLLM server, it will be padded executed as `(4, 512)` prefill bucket, as `batch_size` (number of sequences) will be padded to 4 (closest batch_size dimension higher than 3), and max sequence length will be padded to 512 (closest sequence length dimension higher than 412). After prefill stage, it will be executed as `(4, 512)` decode bucket and will continue as that bucket until either batch dimension changes (due to request being finished) - in which case it will become a `(2, 512)` bucket, or context length increases above 512 tokens, in which case it will become `(4, 640)` bucket.

> [!NOTE]
> Bucketing is transparent to a client - padding in sequence length dimension is never returned to the client, and padding in batch dimension does not create new requests.

## Warmup

Warmup is an optional, but highly recommended step occurring before vLLM server starts listening. It executes a forward pass for each bucket with dummy data. The goal is to pre-compile all graphs and not incur any graph compilation overheads within bucket boundaries during server runtime. Each warmup step is logged during vLLM startup:

```{.}
INFO 08-01 22:26:47 hpu_model_runner.py:1066] [Warmup][Prompt][1/24] batch_size:4 seq_len:1024 free_mem:79.16 GiB
INFO 08-01 22:26:47 hpu_model_runner.py:1066] [Warmup][Prompt][2/24] batch_size:4 seq_len:896 free_mem:55.43 GiB
INFO 08-01 22:26:48 hpu_model_runner.py:1066] [Warmup][Prompt][3/24] batch_size:4 seq_len:768 free_mem:55.43 GiB
...
INFO 08-01 22:26:59 hpu_model_runner.py:1066] [Warmup][Prompt][24/24] batch_size:1 seq_len:128 free_mem:55.43 GiB
INFO 08-01 22:27:00 hpu_model_runner.py:1066] [Warmup][Decode][1/48] batch_size:4 seq_len:2048 free_mem:55.43 GiB
INFO 08-01 22:27:00 hpu_model_runner.py:1066] [Warmup][Decode][2/48] batch_size:4 seq_len:1920 free_mem:55.43 GiB
INFO 08-01 22:27:01 hpu_model_runner.py:1066] [Warmup][Decode][3/48] batch_size:4 seq_len:1792 free_mem:55.43 GiB
...
INFO 08-01 22:27:16 hpu_model_runner.py:1066] [Warmup][Decode][47/48] batch_size:2 seq_len:128 free_mem:55.43 GiB
INFO 08-01 22:27:16 hpu_model_runner.py:1066] [Warmup][Decode][48/48] batch_size:1 seq_len:128 free_mem:55.43 GiB
```

This example uses the same buckets as in *Bucketing mechanism* section. Each output line corresponds to execution of a single bucket. When bucket is executed for the first time, its graph is compiled and can be reused later on, skipping further graph compilations.

> [!TIP] 
> Compiling all the buckets might take some time and can be turned off with `VLLM_SKIP_WARMUP=true` environment variable. Keep in mind that if you do that, you may face graph compilations once executing a given bucket for the first time. It is fine to disable warmup for development, but it's highly recommended to enable it in deployment.

## HPU Graph capture

[HPU Graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) are currently the most performant execution method of vLLM on Intel Gaudi. When HPU Graphs are enabled, execution graphs will be traced (recorded) ahead of time (after performing warmup), to be later replayed during inference, significantly reducing host overheads. Recording can take large amounts of memory, which needs to be taken into account when allocating KV cache. Enabling HPU Graphs will impact the number of available KV cache blocks, but vLLM provides user-configurable variables to control memory management.

When HPU Graphs are being used, they share the common memory pool ("usable memory") as KV cache, determined by `gpu_memory_utilization` flag (`0.9` by default). Before KV cache gets allocated, model weights are loaded onto the device, and a forward pass of the model is executed on dummy data, to estimate memory usage. Only after that, `gpu_memory_utilization` flag is utilized - at its default value, will mark 90% of free device memory at that point as usable. Next, KV cache gets allocated, model is warmed up, and HPU Graphs are captured. Environment variable `VLLM_GRAPH_RESERVED_MEM` defines the ratio of memory reserved for HPU Graphs capture. With its default value (`VLLM_GRAPH_RESERVED_MEM=0.1`), 10% of usable memory will be reserved for graph capture (later referred to as "usable graph memory"), and the remaining 90% will be utilized for KV cache. Environment variable `VLLM_GRAPH_PROMPT_RATIO` determines the ratio of usable graph memory reserved for prefill and decode graphs. By default (`VLLM_GRAPH_PROMPT_RATIO=0.3`), both stages have equal memory constraints. Lower value corresponds to less usable graph memory reserved for prefill stage, e.g. `VLLM_GRAPH_PROMPT_RATIO=0.2` will reserve 20% of usable graph memory for prefill graphs, and 80% of usable graph memory for decode graphs.

> [!NOTE] 
> `gpu_memory_utilization` does not correspond to the absolute memory usage across HPU. It specifies the memory margin after loading the model and performing a profile run. If device has 100 GiB of total memory, and 50 GiB of free memory after loading model weights and executing profiling run, `gpu_memory_utilization` at its default value will mark 90% of 50 GiB as usable, leaving 5 GiB of margin, regardless of total device memory.

User can also configure the strategy for capturing HPU Graphs for prompt and decode stages separately. Strategy affects the order of capturing graphs. There are two strategies implemented: - `max_bs` - graph capture queue will sorted in descending order by their batch sizes. Buckets with equal batch sizes are sorted by sequence length in ascending order (e.g. `(64, 128)`, `(64, 256)`, `(32, 128)`, `(32, 256)`, `(1, 128)`, `(1,256)`), default strategy for decode - `min_tokens` - graph capture queue will be sorted in ascending order by the number of tokens each graph processes (`batch_size*sequence_length`), default strategy for prompt

When there's large amount of requests pending, vLLM scheduler will attempt to fill the maximum batch size for decode as soon as possible. When a request is finished, decode batch size decreases. When that happens, vLLM will attempt to schedule a prefill iteration for requests in the waiting queue, to fill the decode batch size to its previous state. This means that in a full load scenario, decode batch size is often at its maximum, which makes large batch size HPU Graphs crucial to capture, as reflected by `max_bs` strategy. On the other hand, prefills will be executed most frequently with very low batch sizes (1-4), which is reflected in `min_tokens` strategy.

> [!NOTE] 
> `VLLM_GRAPH_PROMPT_RATIO` does not set a hard limit on memory taken by graphs for each stage (prefill and decode). vLLM will first attempt to use up entirety of usable prefill graph memory (usable graph memory * `VLLM_GRAPH_PROMPT_RATIO`) for capturing prefill HPU Graphs, next it will attempt do the same for decode graphs and usable decode graph memory pool. If one stage is fully captured, and there is unused memory left within usable graph memory pool, vLLM will attempt further graph capture for the other stage, until no more HPU Graphs can be captured without exceeding reserved memory pool. The behavior on that mechanism can be observed in the example below.

Each described step is logged by vLLM server, as follows (negative values correspond to memory being released):

```{.}
INFO 08-02 17:37:44 hpu_model_runner.py:493] Prompt bucket config (min, step, max_warmup) bs:[1, 32, 4], seq:[128, 128, 1024]
INFO 08-02 17:37:44 hpu_model_runner.py:499] Generated 24 prompt buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 08-02 17:37:44 hpu_model_runner.py:504] Decode bucket config (min, step, max_warmup) bs:[1, 128, 4], seq:[128, 128, 2048]
INFO 08-02 17:37:44 hpu_model_runner.py:509] Generated 48 decode buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
INFO 08-02 17:37:52 hpu_model_runner.py:430] Pre-loading model weights on hpu:0 took 14.97 GiB of device memory (14.97 GiB/94.62 GiB used) and 2.95 GiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:52 hpu_model_runner.py:438] Wrapping in HPU Graph took 0 B of device memory (14.97 GiB/94.62 GiB used) and -252 KiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:52 hpu_model_runner.py:442] Loading model weights took in total 14.97 GiB of device memory (14.97 GiB/94.62 GiB used) and 2.95 GiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_worker.py:134] Model profiling run took 504 MiB of device memory (15.46 GiB/94.62 GiB used) and 180.9 MiB of host memory (475.4 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_worker.py:158] Free device memory: 79.16 GiB, 39.58 GiB usable (gpu_memory_utilization=0.5), 15.83 GiB reserved for HPUGraphs (VLLM_GRAPH_RESERVED_MEM=0.4), 23.75 GiB reserved for KV cache
INFO 08-02 17:37:54 hpu_executor.py:85] # HPU blocks: 1519, # CPU blocks: 0
INFO 08-02 17:37:54 hpu_worker.py:190] Initializing cache engine took 23.73 GiB of device memory (39.2 GiB/94.62 GiB used) and -1.238 MiB of host memory (475.4 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_model_runner.py:1066] [Warmup][Prompt][1/24] batch_size:4 seq_len:1024 free_mem:55.43 GiB
...
INFO 08-02 17:38:22 hpu_model_runner.py:1066] [Warmup][Decode][48/48] batch_size:1 seq_len:128 free_mem:55.43 GiB
INFO 08-02 17:38:22 hpu_model_runner.py:1159] Using 15.85 GiB/55.43 GiB of free device memory for HPUGraphs, 4.755 GiB for prompt and 11.095 GiB for decode (VLLM_GRAPH_PROMPT_RATIO=0.3)
INFO 08-02 17:38:22 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][1/24] batch_size:1 seq_len:128 free_mem:55.43 GiB
...
INFO 08-02 17:38:26 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][11/24] batch_size:1 seq_len:896 free_mem:48.77 GiB
INFO 08-02 17:38:27 hpu_model_runner.py:1066] [Warmup][Graph/Decode][1/48] batch_size:4 seq_len:128 free_mem:47.51 GiB
...
INFO 08-02 17:38:41 hpu_model_runner.py:1066] [Warmup][Graph/Decode][48/48] batch_size:1 seq_len:2048 free_mem:47.35 GiB
INFO 08-02 17:38:41 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][12/24] batch_size:4 seq_len:256 free_mem:47.35 GiB
INFO 08-02 17:38:42 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][13/24] batch_size:2 seq_len:512 free_mem:45.91 GiB
INFO 08-02 17:38:42 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][14/24] batch_size:1 seq_len:1024 free_mem:44.48 GiB
INFO 08-02 17:38:43 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][15/24] batch_size:2 seq_len:640 free_mem:43.03 GiB
INFO 08-02 17:38:43 hpu_model_runner.py:1128] Graph/Prompt captured:15 (62.5%) used_mem:14.03 GiB buckets:[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (4, 128), (4, 256)]
INFO 08-02 17:38:43 hpu_model_runner.py:1128] Graph/Decode captured:48 (100.0%) used_mem:161.9 MiB buckets:[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
INFO 08-02 17:38:43 hpu_model_runner.py:1206] Warmup finished in 49 secs, allocated 14.19 GiB of device memory
INFO 08-02 17:38:43 hpu_executor.py:91] init_cache_engine took 37.92 GiB of device memory (53.39 GiB/94.62 GiB used) and 57.86 MiB of host memory (475.4 GiB/1007 GiB used)
```

## Recommended vLLM Parameters

- We recommend running inference on Gaudi 2 with `block_size` of 128 for BF16 data type. Using default values (16, 32) might lead to sub-optimal performance due to Matrix Multiplication Engine under-utilization (see [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)).
- For max throughput on Llama 7B, we recommend running with batch size of 128 or 256 and max context length of 2048 with HPU Graphs enabled. If you encounter out-of-memory issues, see troubleshooting section.

## Environment variables

**Diagnostic and profiling knobs:**

- `VLLM_PROFILER_ENABLED`: if `true`, high level profiler will be enabled. Resulting JSON traces can be viewed in [perfetto.habana.ai](https://perfetto.habana.ai/#!/viewer). Disabled by default.
- `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION`: if `true`, will log graph compilations per each vLLM engine step, only when there was any - highly recommended to use alongside `PT_HPU_METRICS_GC_DETAILS=1`. Disabled by default.
- `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL`: if `true`, will log graph compilations per each vLLM engine step, always, even if there were none. Disabled by default.
- `VLLM_HPU_LOG_STEP_CPU_FALLBACKS`: if `true`, will log cpu fallbacks per each vLLM engine step, only when there was any. Disabled by default.
- `VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL`: if `true`, will log cpu fallbacks per each vLLM engine step, always, even if there were none. Disabled by default.
- `VLLM_REGIONAL_COMPILATION`: if `false`, turn off regional complation (when using torch.compile execution mode).

**Performance tuning knobs:**

- `VLLM_SKIP_WARMUP`: if `true`, warmup will be skipped, `false` by default
- `VLLM_GRAPH_RESERVED_MEM`: percentage of memory dedicated for HPUGraph capture, `0.1` by default
- `VLLM_GRAPH_PROMPT_RATIO`: percentage of reserved graph memory dedicated for prompt graphs, `0.3` by default
- `VLLM_GRAPH_PROMPT_STRATEGY`: strategy determining order of prompt graph capture, `min_tokens` or `max_bs`, `min_tokens` by default
- `VLLM_GRAPH_DECODE_STRATEGY`: strategy determining order of decode graph capture, `min_tokens` or `max_bs`, `max_bs` by default
- `VLLM_{phase}_{dim}_BUCKET_{param}` - collection of 12 environment variables configuring ranges of bucketing mechanism
  - `{phase}` is either `PROMPT` or `DECODE`
  - `{dim}` is either `BS`, `SEQ` or `BLOCK`
  - `{param}` is either `MIN`, `STEP` or `MAX`
  - Default values:
    - Prompt:

      - batch size min (`VLLM_PROMPT_BS_BUCKET_MIN`): `1`
      - batch size step (`VLLM_PROMPT_BS_BUCKET_STEP`): `min(max_num_seqs, 32)`
      - batch size max (`VLLM_PROMPT_BS_BUCKET_MAX`): `min(max_num_seqs, 64)`
      - sequence length min (`VLLM_PROMPT_SEQ_BUCKET_MIN`): `block_size`
      - sequence length step (`VLLM_PROMPT_SEQ_BUCKET_STEP`): `block_size`
      - sequence length max (`VLLM_PROMPT_SEQ_BUCKET_MAX`): `max_model_len`

    - Decode:

      - batch size min (`VLLM_DECODE_BS_BUCKET_MIN`): `1`
      - batch size step (`VLLM_DECODE_BS_BUCKET_STEP`): `min(max_num_seqs, 32)`
      - batch size max (`VLLM_DECODE_BS_BUCKET_MAX`): `max_num_seqs`
      - block size min (`VLLM_DECODE_BLOCK_BUCKET_MIN`): `block_size`
      - block size step (`VLLM_DECODE_BLOCK_BUCKET_STEP`): `block_size`
      - block size max (`VLLM_DECODE_BLOCK_BUCKET_MAX`): `max(128, (max_num_seqs*max_model_len)/block_size)`
-  `VLLM_HANDLE_TOPK_DUPLICATES`: if ``true``, will handle duplicates that are outside of top-k, ``false`` by default
-  `VLLM_CONFIG_HIDDEN_LAYERS`: configure how many hidden layers to run in a HPUGraph for model splitting among hidden layers when TP is 1. The default is 1. It helps with throughput improvement under inter-token latency limitation for some models.

Additionally, there are HPU PyTorch Bridge environment variables impacting vLLM execution:

- `PT_HPU_LAZY_MODE`: if `0`, PyTorch Eager backend for Gaudi will be used, if `1` PyTorch Lazy backend for Gaudi will be used, `1` is default
- `PT_HPU_ENABLE_LAZY_COLLECTIVES`: required to be `true` for tensor parallel inference with HPU Graphs

# Quantization, FP8 inference and model calibration process

> [!NOTE]
> Measurement files are required to run quantized models with vLLM on Gaudi accelerators. The FP8 model calibration procedure is described in the [vllm-hpu-extention](https://github.com/HabanaAI/vllm-hpu-extension/tree/main/calibration/README.md) package.

Once you've completed the model calibration process and collected the measurements, you can run FP8 inference with vLLM using the following command:
```bash
export QUANT_CONFIG=/path/to/quant/config/inc/meta-llama-3.1-405b-instruct/maxabs_measure_g3.json
vllm serve meta-llama/Llama-3.1-405B-Instruct --quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu --tensor_paralel_size 8
```

`QUANT_CONFIG` is an environment variable that points to the measurement or quantization configuration file. The measurement configuration file is used during the calibration procedure to collect measurements for a given model. The quantization configuration is used during inference.

> [!TIP]
> If you are just prototyping or testing your model with FP8, you can use the `VLLM_SKIP_WARMUP=true` environment variable to disable the warmup stage, which can take a long time. However, we do not recommend disabling this feature in production environments, as it causes a dramatic performance drop.

> [!TIP]
> When using FP8 models, you may experience timeouts caused by the long compilation time of FP8 operations. To mitigate this problem, you can use these two environment variables:
> - `VLLM_ENGINE_ITERATION_TIMEOUT_S` - to adjust the vLLM server timeout. You can set the value in seconds, e.g., 600 equals 10 minutes.
> - `VLLM_RPC_TIMEOUT` - to adjust the RPC protocol timeout used by the OpenAI-compatible API. This value is in microseconds, e.g., 600000 equals 10 minutes.

# Troubleshooting: Tweaking HPU Graphs

If you experience device out-of-memory issues or want to attempt inference at higher batch sizes, try tweaking HPU Graphs by following the below:

- Tweak `gpu_memory_utilization` knob. It will decrease the allocation of KV cache, leaving some headroom for capturing graphs with larger batch size. By default `gpu_memory_utilization` is set to 0.9. It attempts to allocate ~90% of HBM left for KV cache after short profiling run. Note that decreasing reduces the number of KV cache blocks you have available, and therefore reduces the effective maximum number of tokens you can handle at a given time.
- If this method is not efficient, you can disable `HPUGraph` completely. With HPU Graphs disabled, you are trading latency and throughput at lower batches for potentially higher throughput on higher batches. You can do that by adding `--enforce-eager` flag to server (for online inference), or by passing `enforce_eager=True` argument to LLM constructor (for offline inference).


# Changelog
## 1.19.0
### New features
 - Added fake HPU mode to Habana components with dummy habana_frameworks module. ([#250](https://github.com/HabanaAI/vllm-fork/pull/250))
 - Enabled HPU Graph capture even when warmup is skipped ([#320](https://github.com/HabanaAI/vllm-fork/pull/320))
 - Introduced vllm-hpu-extension, removed vllm.hpu directory and changed relevant imports ([#291](https://github.com/HabanaAI/vllm-fork/pull/291), [#323](https://github.com/HabanaAI/vllm-fork/pull/323))
 - Enabled async output processing for HPU ([#342](https://github.com/HabanaAI/vllm-fork/pull/342))
 - Enabled automatic BF16 usage on HPU instead of FP16 ([#361](https://github.com/HabanaAI/vllm-fork/pull/361))
 - Added padding-aware scheduling and option to limit prefill batch size ([#394](https://github.com/HabanaAI/vllm-fork/pull/394))
 - Overhauled HPU support of RotaryEmbedding ([#404](https://github.com/HabanaAI/vllm-fork/pull/404))
 - Added HPU specific arguments to benchmark_throughput ([#406](https://github.com/HabanaAI/vllm-fork/pull/406))
 - Added support for long context lengths with LoRA ([#418](https://github.com/HabanaAI/vllm-fork/pull/418))
 - Added support for various softmax normalization options ([#378](https://github.com/HabanaAI/vllm-fork/pull/378), [#420](https://github.com/HabanaAI/vllm-fork/pull/420))
 - Added initial support for automatic prefix caching ([#162](https://github.com/HabanaAI/vllm-fork/pull/162))
 - Added multi step scheduling HPU support with tensor parallelism support ([#441](https://github.com/HabanaAI/vllm-fork/pull/441), [#457](https://github.com/HabanaAI/vllm-fork/pull/457))
 - Added HPU support for speculative_decoding ([#375](https://github.com/HabanaAI/vllm-fork/pull/375), [#461](https://github.com/HabanaAI/vllm-fork/pull/461))
 - Aligned HPU fork with upstream code up to 6a585a2 (v0.6.3.post1) ([#259](https://github.com/HabanaAI/vllm-fork/pull/259), [#311](https://github.com/HabanaAI/vllm-fork/pull/311), [#340](https://github.com/HabanaAI/vllm-fork/pull/340), [#353](https://github.com/HabanaAI/vllm-fork/pull/353), [#465](https://github.com/HabanaAI/vllm-fork/pull/465), [#468](https://github.com/HabanaAI/vllm-fork/pull/468))
 
### Performance optimizations
 - Reduced default value of VLLM_GRAPH_RESERVED_MEM to 0.1 ([#292](https://github.com/HabanaAI/vllm-fork/pull/292))
 - Added attention performance optimizations: prefill cache write chunking, div_i32 removal from insert_or_update_cache ([#289](https://github.com/HabanaAI/vllm-fork/pull/289))
 - Optimized Qwen2 model on Gaudi ([#233](https://github.com/HabanaAI/vllm-fork/pull/233))
 - Optimized performance of top_p and top_k calculations ([#449](https://github.com/HabanaAI/vllm-fork/pull/449))
 - Removed CPU sync before sampler ([#414](https://github.com/HabanaAI/vllm-fork/pull/414))
 - Enabled Contiguous Paged Attention optimization ([#424](https://github.com/HabanaAI/vllm-fork/pull/424), [#433](https://github.com/HabanaAI/vllm-fork/pull/433))
 - Reduced block fragmentation ([#426](https://github.com/HabanaAI/vllm-fork/pull/426))
 - Enabled FusedSDPA prefill by default ([#447](https://github.com/HabanaAI/vllm-fork/pull/447), [#448](https://github.com/HabanaAI/vllm-fork/pull/448))
 - Offload logits processing to CPU when guided decoding is used ([#358](https://github.com/HabanaAI/vllm-fork/pull/358))
 - Enabled Dynamic MoE layer for Mixtral ([#425](https://github.com/HabanaAI/vllm-fork/pull/425))
 - Enabled INC patching matmuls in paged attention's block2batch and batch2block ([#500](https://github.com/HabanaAI/vllm-fork/pull/500))
 - Optimized multi-step scheduling deepcopy overhead ([#452](https://github.com/HabanaAI/vllm-fork/pull/452))
 - Enabled warmup for multi-step scheduling ([#501](https://github.com/HabanaAI/vllm-fork/pull/501))
 
### Bugfixes
 - Fixed LLaVA-1.5 multi-modal model inference ([#283](https://github.com/HabanaAI/vllm-fork/pull/283))
 - Fixed blocks number calculation for Flat Paged Attention ([#269](https://github.com/HabanaAI/vllm-fork/pull/269))
 - Fixed initialize_ray_cluster device_str bug ([#297](https://github.com/HabanaAI/vllm-fork/pull/297))
 - Fixed calculating slots for warmup ([#310](https://github.com/HabanaAI/vllm-fork/pull/310))
 - Removed padding block from a list of available blocks in allocators ([#313](https://github.com/HabanaAI/vllm-fork/pull/313))
 - Fixed seq_len for padding sequences ([#318](https://github.com/HabanaAI/vllm-fork/pull/318))
 - Fixed LoRA specific conditions in profile_run ([#317](https://github.com/HabanaAI/vllm-fork/pull/317))
 - Removed throwing "Failed to imported from vllm._C" warning on HPU ([#326](https://github.com/HabanaAI/vllm-fork/pull/326))
 - Fixed documentation build warnings ([#330](https://github.com/HabanaAI/vllm-fork/pull/330))
 - Fixed INC FP8 inference after rebase ([#333](https://github.com/HabanaAI/vllm-fork/pull/333))
 - Refined INC shutdown code ([#335](https://github.com/HabanaAI/vllm-fork/pull/335))
 - Fixed torch.compile issue of dispatch key set mismatch ([#299](https://github.com/HabanaAI/vllm-fork/pull/299))
 - Fixed runtime errors reported when using long input sequence lengths with LoRA ([#339](https://github.com/HabanaAI/vllm-fork/pull/339))
 - Fixed hpu_set_env call in load_model in vllm ([#364](https://github.com/HabanaAI/vllm-fork/pull/364))
 - Fixed LoRA tests ([#376](https://github.com/HabanaAI/vllm-fork/pull/376))
 - Removed constraints for bucket creation during warmup in LoRA ([#382](https://github.com/HabanaAI/vllm-fork/pull/382))
 - Fixed lora_manager tests with hpu_model_runner ([#386](https://github.com/HabanaAI/vllm-fork/pull/386))
 - Removed workaround added to resolve multi-card stall issue ([#387](https://github.com/HabanaAI/vllm-fork/pull/387))
 - Added workaround for RuntimeError: "fill_cpu" not implemented for 'Float8_e4m3fn' ([#402](https://github.com/HabanaAI/vllm-fork/pull/402))
 - Fixed SchedulerConfig params ([#459](https://github.com/HabanaAI/vllm-fork/pull/459))
 - Fixed multistep deepcopy overhead ([#452](https://github.com/HabanaAI/vllm-fork/pull/452))
 - Added option to disable duplicates in topk ([#464](https://github.com/HabanaAI/vllm-fork/pull/464))
 - Enabled lazy import of HPU-dependent components ([#363](https://github.com/HabanaAI/vllm-fork/pull/363))
 - Fixed bug: seed_everything function doesn't handle HPU ([#384](https://github.com/HabanaAI/vllm-fork/pull/384))
 - Removed redundant set_active_loras call during warmup ([#413](https://github.com/HabanaAI/vllm-fork/pull/413))
 - Fixed number of blocks when profiling contiguous paged attention ([#496](https://github.com/HabanaAI/vllm-fork/pull/496))
 - Fixed one_hot bug in torch compile mode ([#427](https://github.com/HabanaAI/vllm-fork/pull/427))


### Other
 - Updated SynapseAI version in README & Dockerfile ([#390](https://github.com/HabanaAI/vllm-fork/pull/390))
 - Updated documentation on support of FP8 ([#288](https://github.com/HabanaAI/vllm-fork/pull/288))
 - Added FP8 inference procedure ([#504](https://github.com/HabanaAI/vllm-fork/pull/504))
 - Fixed broken urls in gaudi-installation ([#473](https://github.com/HabanaAI/vllm-fork/pull/473))
 - Renamed vLLM components from Habana to HPU ([#359](https://github.com/HabanaAI/vllm-fork/pull/359))



## 1.18.0
### New features
 - Added support FP8 INC inference ([#144](https://github.com/HabanaAI/vllm-fork/pull/144))
 - Added support for FusedSDPA prefills ([#168](https://github.com/HabanaAI/vllm-fork/pull/168))
 - Enabled LoRA support for HPU ([#170](https://github.com/HabanaAI/vllm-fork/pull/170), [#247](https://github.com/HabanaAI/vllm-fork/pull/247))
 - Enabled buckets not warmed-up warnings ([#222](https://github.com/HabanaAI/vllm-fork/pull/222))
 - Enabled Flat Paged Attention optimization ([#169](https://github.com/HabanaAI/vllm-fork/pull/169))
 - Added disable_tensor_cache=True to HPUGraph capture ([#252](https://github.com/HabanaAI/vllm-fork/pull/252))
 - Added support for Mixtral quantization using INC ([#267](https://github.com/HabanaAI/vllm-fork/pull/267))
 - Added option to skip forward pass execution during warmup ([#227](https://github.com/HabanaAI/vllm-fork/pull/227))
 - Added PyTorch profiler integration ([#256](https://github.com/HabanaAI/vllm-fork/pull/256))
 - Added Dockerfile.hpu ([#200](https://github.com/HabanaAI/vllm-fork/pull/200))
 - Added topp/topk calculation sampler optimization ([#195](https://github.com/HabanaAI/vllm-fork/pull/195))
### Bugfixes
 - HPU Buckets now don't exceed token budget ([#206](https://github.com/HabanaAI/vllm-fork/pull/206))
 - Fixed bug causing incorrect lower bucket bounadry calculation ([#239](https://github.com/HabanaAI/vllm-fork/pull/239))
 - Fixed ALiBi support ([#254](https://github.com/HabanaAI/vllm-fork/pull/254))
 - Fixed HPU guided decoding crashes ([#236](https://github.com/HabanaAI/vllm-fork/pull/236))
 - Fixed incorrect handlign of large bucket minimums ([#235](https://github.com/HabanaAI/vllm-fork/pull/235))
 - Issued Llama-405b workaround for memory allocation error ([#184](https://github.com/HabanaAI/vllm-fork/pull/184))
 - Enabled dispersed dummy cache slots for avoiding caching issues ([#243](https://github.com/HabanaAI/vllm-fork/pull/243))
 - Eliminated Llama and GPTBigCode graph breaks in torch.compile mode ([#202](https://github.com/HabanaAI/vllm-fork/pull/202))

 


