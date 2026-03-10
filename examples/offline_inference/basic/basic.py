# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

os.environ["PT_HPU_LAZY_MODE"] = "1"

from vllm import LLM, SamplingParams
# model_path = "/mnt/disk2/HF_models/Qwen3-30B-A3B"
model_path = "/mnt/disk2/HF_models/Step-3.5-Flash-FP8"
# model_path = "/mnt/disk2/HF_models/Step-3.5-Flash"
# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default=model_path,
    help="The model path.",
)
parser.add_argument("--tp-size", type=int, default=8, help="The number of threads.")
parser.add_argument(
    "--output-tokens", type=int, default=64, help="The number of output tokens."
)
parser.add_argument(
    "--max-model-length", type=int, default=1024, help="Max model length."
)
parser.add_argument("--enable-ep", action="store_true", help="Enable EP for MOE models")
args = parser.parse_args()

# os.environ["VLLM_SKIP_WARMUP"] = "true"
# os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
# os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
# os.environ["PT_HPU_WEIGHT_SHARING"] = "0"

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
# os.environ["VLLM_RAY_DISABLE_LOG_TO_DRIVER"] = "1"
# os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
os.environ["VLLM_MOE_N_SLICE"] = "1"
os.environ["VLLM_EP_SIZE"] = "8"
os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"

if __name__ == "__main__":
    # Sample prompts.
    prompts = [
        # "Hello, my name is",
        # "The president of the United States is",
        "The capital of France is",
        # "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=args.output_tokens
    )
    model = args.model
    if args.tp_size == 1:
        llm = LLM(
            model=model,
            tokenizer=model,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=args.max_model_length,
        )
    else:
        llm = LLM(
            model=model,
            tokenizer=model,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend="mp",
            trust_remote_code=True,
            max_model_len=args.max_model_length,
            enable_expert_parallel=True,
            dtype="bfloat16",
        )

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print()
    exit()
