import json
import math
import argparse

"""
Example: python <script_path> --input-len 1024 --output-len 1024 --range-ratio 0.0 --bs-list  4,8,16,32,64 --bucket-size 128
"""


def calculate_benchmark_vars(input_len, output_len, range_ratio, bs_list, bucket_size):
    """
    Calculates various configuration parameters for LLM benchmarking
    based on length ranges, batch sizes, and model block constraints (bucket_size).

    Args:
        input_len (int): Base input length (L_in).
        output_len (int): Base output length (L_out).
        range_ratio (float): The ratio for length variability (R).
        bs_list (list): List of concurrency levels (batch sizes).
        bucket_size (int): The block/bucket step size (e.g., 128).
    """

    # --- 1. Calculate Length Ranges ---

    # Formula: [length * (1 - range_ratio), length * (1 + range_ratio)]
    min_ratio = 1.0 - range_ratio
    max_ratio = 1.0 + range_ratio

    # Calculate min/max input and output lengths (rounding up to nearest integer)
    min_in_len = math.ceil(input_len * min_ratio)
    max_in_len = math.ceil(input_len * max_ratio)
    min_out_len = math.ceil(output_len * min_ratio)
    max_out_len = math.ceil(output_len * max_ratio)

    # --- 2. Core Derived Lengths ---

    # max_model_len: max in + max out
    max_model_len = max_in_len + max_out_len

    # prompt_sq_bucket_max: ceil(max_model_len / bucket_size) * bucket_size
    prompt_sq_bucket_max = math.ceil(max_model_len / bucket_size) * bucket_size

    # --- 3. Batch Size (BS) Variables ---

    # BS range min/max are derived from the input list
    bs_min = min(bs_list)
    bs_max = max(bs_list)

    # prompt_bs_min: explicitly defined as 1
    prompt_bs_min = 1

    # prompt_bs_max: max bs range (for prompt phase)
    prompt_bs_max = bs_max

    # decode_bs_max: max bs range (for decode phase)
    decode_bs_max = bs_max

    # decode_bs_min: min bs range (formerly decode_min)
    decode_bs_min = bs_min

    # prompt/decode bs step: bs step range (using the difference between the first two elements)
    if len(bs_list) > 1:
        prompt_decode_bs_step = bs_list[1] - bs_list[0]
    else:
        # If only one BS is provided, use the BS itself.
        prompt_decode_bs_step = bs_min

    # --- 4. Bucket/Block Calculations ---

    # prompt_squ_bucket_step: explicitly set to bucket_size
    prompt_sq_bucket_step = bucket_size

    # decode_block_bucket_step: explicitly set to bucket_size
    decode_block_bucket_step = bucket_size

    # decode_block_bucket_max: bs-max * ceil(max_model_len / bucket_size) + 2*bucket_size
    blocks_per_max_request = math.ceil(max_model_len / bucket_size)
    raw_decode_block_bucket_max = bs_max * blocks_per_max_request + 2 * bucket_size
    decode_block_bucket_max = (
        math.ceil(raw_decode_block_bucket_max / decode_block_bucket_step)
        * decode_block_bucket_step
    )

    # decode_block_bucket_min: bs-min * ceil(min_in_len_with_range / bucket_size)
    blocks_per_min_prompt = math.ceil(min_in_len / bucket_size)
    raw_decode_block_bucket_min = bs_min * blocks_per_min_prompt
    # Ensure final result is divisible by decode_block_bucket_step (bucket_size)
    decode_block_bucket_min = (
        math.ceil(raw_decode_block_bucket_min / decode_block_bucket_step)
        * decode_block_bucket_step
    )

    # --- 5. Assemble Results ---

    results = {
        "benchmark_inputs": {
            "base_input_len": input_len,
            "base_output_len": output_len,
            "range_ratio": range_ratio,
            "concurrency_list": bs_list,
            "bucket_size": bucket_size,
        },
        "length_ranges": {
            "min_in_len": min_in_len,
            "max_in_len": max_in_len,
            "min_out_len": min_out_len,
            "max_out_len": max_out_len,
        },
        "calculated_variables": {
            "MAX_MODEL_LEN": max_model_len,
            "VLLM_PROMPT_BS_BUCKET_MIN": prompt_bs_min,
            "VLLM_PROMPT_BS_BUCKET_STEP": prompt_decode_bs_step,
            "VLLM_PROMPT_BS_BUCKET_MAX": prompt_bs_max,
            "VLLM_PROMPT_SEQ_BUCKET_MIN": prompt_sq_bucket_step,
            "VLLM_PROMPT_SEQ_BUCKET_STEP": prompt_sq_bucket_step,
            "VLLM_PROMPT_SEQ_BUCKET_MAX": prompt_sq_bucket_max,
            "VLLM_DECODE_BS_BUCKET_MIN": decode_bs_min,
            "VLLM_DECODE_BS_BUCKET_STEP": prompt_decode_bs_step,
            "VLLM_DECODE_BS_BUCKET_MAX": decode_bs_max,
            "VLLM_DECODE_BLOCK_BUCKET_MIN": decode_block_bucket_min,
            "VLLM_DECODE_BLOCK_BUCKET_STEP": decode_block_bucket_step,
            "VLLM_DECODE_BLOCK_BUCKET_MAX": decode_block_bucket_max,
        },
    }

    # Print output in JSON format
    print(json.dumps(results, indent=4))

    # Print variables as shell export commands
    for key, value in results["calculated_variables"].items():
        print(f"export {key}={value}")


# --- Execution with Command Line Arguments ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate benchmark configuration variables."
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Base input length (tokens). Default: 1024.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="Base output length (tokens). Default: 1024.",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for length variability (R). Default: 0.0.",
    )
    parser.add_argument(
        "--bs-list",
        type=str,
        default="16",
        help="Comma-separated list of batch sizes/concurrency levels. Default: 16.",
    )
    parser.add_argument( # New Argument
        "--bucket-size",
        type=int,
        default=128,
        help="The block/bucket step size for sequence and block calculations. Default: 128.",
    )

    args = parser.parse_args()

    # Convert comma-separated string to list of integers
    try:
        bs_list_parsed = [int(x.strip()) for x in args.bs_list.split(",")]
    except ValueError:
        print("Error: Batch size list must contain integers separated by commas.")
        exit(1)

    calculate_benchmark_vars(
        args.input_len, args.output_len, args.range_ratio, bs_list_parsed, args.bucket_size
    )