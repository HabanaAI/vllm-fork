# vLLM-fork Warmup Strategy on Gaudi (HPU)

This document provides a brief overview of the warmup strategy used in the Habana vLLM-fork, specifically focusing on the use of **Bucketing** to optimize resource allocation and performance on Gaudi accelerators.

## Overview of Warmup Strategy

When running LLMs on Gaudi using the vLLM-fork, the initial model loading and compilation process can take some time. The warmup strategy is critical for ensuring subsequent performance is consistent and fast by pre-loading and compiling the necessary attention mechanisms for common sequence lengths.

The vLLM-fork utilizes a technique often referred to as **Bucketing (exponential or Linear)** for this purpose. This strategy involves pre-compiling attention kernels (Prompt and Decode) for specific token sequence lengths that grow exponentially (e.g., 128, 256, 512, 1024, etc.). This prepares the hardware to handle requests efficiently across a wide range of common input/output sizes.

### Linear vs Exponential

The `VLLM_EXPONENTIAL_BUCKETING=True` flag, enabled by default starting with the `1.21.0-post1` vLLM release, switches the bucketing strategy from linear to exponential. This can reduce the number of buckets and warmup time by up to 80%, while generally maintaining equivalent inference performance. For larger sequence lenght, it is recomended to use exponential to reduce the warmup time. Linear bucketing can be enabled by setting `VLLM_EXPONENTIAL_BUCKETING=False`.

### Configuration via Environment Variables

The behavior of the warmup phase and the underlying memory allocation are primarily controlled using specific environment variables. These variables allow users to tune the batch size limits, token block sizes, and the extent of the warmup buckets, directly influencing memory usage and maximum throughput.

For instance, variables control:

* **Maximum Model Length (`MAX_MODEL_LEN`)**: The longest total sequence (input + output) the model will handle.

* **Batch Size Limits (`PROMPT_BS_MAX`, `DECODE_BS_MAX`)**: The largest concurrent batch size allowed for prompt and decode operations.

* **Bucket Steps (`PROMPT_SQU_BUCKET_STEP`, `DECODE_BLOCK_BUCKET_STEP`)**: Defines the incremental size used for the exponential length buckets (often set to 128).

* **Block Bucket Max/Min**: Defines the limits for the total physical memory blocks reserved for the cache.

Adjusting these settings is essential for balancing memory usage and preventing OOM (Out-of-Memory) errors against optimizing your target latency and throughput requirements.

## Automated Configuration Calculation

Since manually calculating the optimal values for variables like `MAX_MODEL_LEN`, `DECODE_BLOCK_BUCKET_MAX`, and `PROMPT_SQU_BUCKET_MAX` can be complex, a utility script is provided to automate this process.

This script calculates the required environment variable values based on user-provided inputs, such as:

* **Input Length** and **Output Length**
* **Range Ratio** (to calculate minimum and maximum sequence lengths)
* A list of target **Batch Sizes** or concurrency levels

The script applies the necessary formulas, including ensuring the block bucket variables are rounded up to the nearest multiple of the block step (typically 128). This ensures the vLLM-fork is correctly configured for the desired load profile, which is critical for efficient warmup and overall memory management.

> [!NOTE]
> This script is only for languge models (not multimodals), and only for `vllm-fork` (as of date).
> Please make sure `max_model_len` output matches the vllm server ones. `range-ratio` arg should match the vllm server cmd (default 0).
> `bs-list` is the list of intended `concurancy` used in client side or `max_num_seqs` in server.

### Example

```sh
python  hpu-vllm-bucket-calc.py --input-len 8096 --output-len 1024 --range-ratio 0.8 --bs-list  4,8,16,32,64
```

```json
{
    "benchmark_inputs": {
        "base_input_len": 8096,
        "base_output_len": 1024,
        "range_ratio": 0.8,
        "concurrency_list": [
            4,
            8,
            16,
            32,
            64
        ]
    },
    "length_ranges": {
        "min_in_len": 1620,
        "max_in_len": 14573,
        "min_out_len": 205,
        "max_out_len": 1844
    },
    "calculated_variables": {
        "MAX_MODEL_LEN": 16417,
        "VLLM_PROMPT_BS_BUCKET_MIN": 1,
        "VLLM_PROMPT_BS_BUCKET_STEP": 4,
        "VLLM_PROMPT_BS_BUCKET_MAX": 64,
        "VLLM_PROMPT_SEQ_BUCKET_MIN": 128,
        "VLLM_PROMPT_SEQ_BUCKET_STEP": 128,
        "VLLM_PROMPT_SEQ_BUCKET_MAX": 16512,
        "VLLM_DECODE_BS_BUCKET_MIN": 4,
        "VLLM_DECODE_BS_BUCKET_STEP": 4,
        "VLLM_DECODE_BS_BUCKET_MAX": 64,
        "VLLM_DECODE_BLOCK_BUCKET_MIN": 128,
        "VLLM_DECODE_BLOCK_BUCKET_STEP": 128,
        "VLLM_DECODE_BLOCK_BUCKET_MAX": 8576
    }
}
```

```txt
export MAX_MODEL_LEN=16417
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=4
export VLLM_PROMPT_BS_BUCKET_MAX=64
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=128
export VLLM_PROMPT_SEQ_BUCKET_MAX=16512
export VLLM_DECODE_BS_BUCKET_MIN=4
export VLLM_DECODE_BS_BUCKET_STEP=4
export VLLM_DECODE_BS_BUCKET_MAX=64
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=128
export VLLM_DECODE_BLOCK_BUCKET_MAX=8576
```

## References

For detailed information on configuring the vLLM-fork for Gaudi, including a complete list of environment variables and the exponential bucketing methodology, please consult the official documentation:

* **Environment Variables**: <https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#environment-variables>

* **Warmup Methodology (Exponential Bucketing)**: <https://docs.habana.ai/en/latest/PyTorch/vLLM_Inference/Managing_vLLM_Warmup_Time.html#exponential-bucketing>

*This README is a high-level summary. Always refer to the official Habana documentation for the latest configuration details.*
