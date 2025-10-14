# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

os.environ["PT_HPU_LAZY_MODE"] = "1"

from vllm import LLM, SamplingParams

# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    default="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    help="The model path.")
parser.add_argument("--tp-size",
                    type=int,
                    default=4,
                    help="The number of threads.")
parser.add_argument("--output-tokens",
                    type=int,
                    default=512,
                    help="The number of output tokens.")
parser.add_argument("--max-model-length",
                    type=int,
                    default=16384,
                    help="Max model length.")
parser.add_argument("--enable-ep",
                    action='store_true',
                    help="Enable EP for MOE models")
args = parser.parse_args()

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"

if __name__ == "__main__":

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
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
            distributed_executor_backend='mp',
            trust_remote_code=True,
            max_model_len=args.max_model_length,
            enable_expert_parallel=args.enable_ep,
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
