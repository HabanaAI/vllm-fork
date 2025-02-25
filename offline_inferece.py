from vllm import LLM
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
import PIL

import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("-m", "--model", help="model name or path")
parser.add_argument("-i", "--image", help="type of image")
parser.add_argument(
    "--multiple_prompts", action="store_true", help="to run with multiple prompts"
)
parser.add_argument("--iter", help="number of iterations to run")

# Parse the arguments
args = parser.parse_args()

# Load the image

if args.image == "synthetic":
    image = ImageAsset("stop_sign").pil_image
    newsize = (225, 225)
    image = image.resize(newsize)
elif args.image == "snowscat":
    filename = "/tmp/snowscat-H3oXiq7_bII-unsplash.jpg"
    image = PIL.Image.open(filename)
else:
    print(f"unknow image {args.image}")
    exit

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

tested_models = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]
mdl = args.model
multiple_prompt = args.multiple_prompts

question = "Describe this image."
# Prompt example: https://docs.vllm.ai/en/v0.6.2/getting_started/examples/offline_inference_vision_language.html
if "Qwen2" in mdl:
    llm = LLM(
        model=mdl,
        enforce_eager=False,
        max_model_len=32768,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": 1},
    )
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}g<|im_end|>\n<|im_start|>assistant\n"

    if multiple_prompt:
        batch_data = [
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            {
                "prompt": "<|im_start|>system\nYou are a nice person.<|im_end|>\n<|im_start|>user\nTell me about you.<|im_end|>\n<|im_start|>assistant\n"
            },
            {"prompt": prompt, "multi_modal_data": {"image": image}},
        ]
    else:
        batch_data = {"prompt": prompt, "multi_modal_data": {"image": image}}

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

for i in range(int(args.iter)):
    print(f"==ITER : [{i}]")
    outputs = llm.generate(batch_data)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
