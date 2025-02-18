from vllm import LLM
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
import PIL

import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("-m", "--model", help="model name or path")
# Parse the arguments
args = parser.parse_args()

# Load the image
filename = "/tmp/snowscat-H3oXiq7_bII-unsplash.jpg"
image = PIL.Image.open(filename)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

tested_models = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]
mdl = args.model

question = "Describe this image."
# Prompt example: https://docs.vllm.ai/en/v0.6.2/getting_started/examples/offline_inference_vision_language.html
if "Qwen2" in mdl:
    llm = LLM(model=mdl, enforce_eager=False)
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}g<|im_end|>\n<|im_start|>assistant\n"
elif "Llama-3.2" in mdl:
    llm = LLM(
        model=mdl,
        max_model_len=2048,
        max_num_seqs=64,
        tensor_parallel_size=1,
        num_scheduler_steps=32,
        max_num_prefill_seqs=4,
    )
    prompt = f"<|image|><|begin_of_text|>{question}"
else:
    print(f"{mdl} is not known model?")

outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
