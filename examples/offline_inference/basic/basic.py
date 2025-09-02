# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)



from vllm import LLM, SamplingParams

import argparse

# Parse the command-line arguments.
model_path="/mnt/disk5/lmsys/gpt-oss-20b-bf16"
model_path = "/software/users/yiliu4/HF_HOME/lmsys/gpt-oss-20b-bf16"
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=model_path, help="The model path.")
parser.add_argument("--tokenizer", type=str, default=model_path, help="The model path.")
parser.add_argument("--tp_size", type=int, default=8, help="The number of threads.")
parser.add_argument("--ep_size", type=int, default=1, help="The number of threads.")
parser.add_argument("--inc", action="store_true", help="Use inc.")
args = parser.parse_args()




def main():
    # Create an LLM.
    param = {}
    if args.inc:
        param["quantization"] = "inc"
    if args.ep_size > 1:
        param["enable_expert_parallel"] = True
    
    llm = LLM(
        # model="facebook/opt-125m"
        # model="/mnt/disk5/lmsys/gpt-oss-20b-bf16",
        model=model_path,
        max_model_len=1024,
        max_num_seqs=32,
        tensor_parallel_size=args.tp_size,
        
        **param
        )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    import os
    if os.getenv("QUANT_CONFIG", None) is not None:
        llm.llm_engine.model_executor.shutdown()
    del llm
if __name__ == "__main__":
    main()
