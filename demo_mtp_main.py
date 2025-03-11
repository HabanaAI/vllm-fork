import torch
from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/mnt/weka/data/pytorch/DeepSeek-V3-bf16/", dtype=torch.bfloat16, tensor_parallel_size=1, trust_remote_code=True, enforce_eager=True, max_model_len=10)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


