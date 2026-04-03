# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

os.environ["PT_HPU_LAZY_MODE"] = "1"

from vllm import LLM, SamplingParams

# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="facebook/opt-125m",
    help="The model path.",
)
parser.add_argument("--tp-size", type=int, default=2, help="The number of threads.")
parser.add_argument(
    "--output-tokens", type=int, default=512, help="The number of output tokens."
)
parser.add_argument(
    "--max-model-length", type=int, default=16384, help="Max model length."
)
parser.add_argument("--enable-ep", action="store_true", help="Enable EP for MOE models")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-p", type=float, default=0.95)
parser.add_argument(
    "--enable-thinking", action="store_true", help="Enable think mode for inference"
)
# Add example params
parser.add_argument("--chat-template-path", type=str)
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
    messages = []
    for idx in range(len(prompts)):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompts[idx]},
        ]
        messages.append(conversation)
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.output_tokens,
    )
    chat_template_path = args.chat_template_path
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
            enable_expert_parallel=args.enable_ep,
            dtype="bfloat16",
        )

    def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for idx in range(len(outputs)):
            prompt = prompts[idx]
            generated_text = outputs[idx].outputs[0].text
            print(f"Prompt: {prompt!r}\n")
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

    print("=" * 80)

    # A chat template can be optionally supplied.
    # If not, the model will use its default chat template.
    if chat_template_path is not None:
        with open(chat_template_path) as f:
            chat_template = f.read()

        outputs = llm.chat(
            messages,
            sampling_params,
            use_tqdm=False,
            chat_template=chat_template,
            chat_template_kwargs={"enable_thinking": args.enable_thinking},
        )
    else:
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": args.enable_thinking},
        )
    print_outputs(outputs)
