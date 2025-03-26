from vllm import LLM
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
import PIL
import cv2
from PIL import Image
import random

import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("-m", "--model", help="model name or path")
parser.add_argument("-i", "--image", help="type of image")
parser.add_argument("-v", "--video", help="Video Input")
parser.add_argument("-t", "--text_only", action="store_true", help="Text only pormpts")
parser.add_argument("--image_width", type=int, help="Image width size")
parser.add_argument("--image_height", type=int, help="Image width size")
parser.add_argument(
    "--multiple_prompts", action="store_true", help="to run with multiple prompts"
)
parser.add_argument("--iter", help="number of iterations to run")
parser.add_argument(
    "-gmu", "--gpu_mem_usage", type=float, default=0.9, help="GPU memory usage value"
)
parser.add_argument("-ris", "--random_img_size", action="store_true", help="Text only pormpts")

# Parse the arguments
args = parser.parse_args()

# Process video input and select specified number of evenly distributed frames


def sample_frames(path, num_frames):
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            pil_img = pil_img.resize((256, 256))
            frames.append(pil_img)
    video.release()
    return frames[:num_frames]


# Load the image / Video
if args.image:
    if args.image == "synthetic":
        image = ImageAsset("stop_sign").pil_image
    elif args.image == "snowscat":
        filename = "/tmp/snowscat-H3oXiq7_bII-unsplash.jpg"
        image = PIL.Image.open(filename)
    elif args.image:
        image = PIL.Image.open(args.image)

    if args.image_width and args.image_height:
        image = image.resize((args.image_width, args.image_height))

    if args.random_img_size:
        rand_width = random.randint(0.5 * args.image_width, 1 * args.image_width)
        rand_height = random.randint(0.5 * args.image_height, 1 * args.image_height)
        image2 = image.resize((rand_width, rand_height))

elif args.video:
    video = sample_frames(args.video, 50)
elif args.text_only:
    print(f"Running in text-only mode")
else:
    print(f"Unknow image/Video Input {args.image} {args.video}")
    exit

sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

tested_models = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]
mdl = args.model
multiple_prompt = args.multiple_prompts

question_list = [
    "Describe this video.",
    "Tell me about future of global warming.",
    "Can you describe Bernoulli's Equation?",
    "How does deepspeed work?",
]

if args.video:
    llm = LLM(
        model=mdl,
        enforce_eager=False,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_usage,
    )

    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{question_list[0]}<|im_end|>\n<|im_start|>assistant\n"

    if multiple_prompt:
        batch_data = [
            {"prompt": prompt, "multi_modal_data": {"video": video}},
            {
                "prompt": f"<|im_start|>system\nYou are a nice person.<|im_end|>\n<|im_start|>user\n{question_list[0]}<|im_end|>\n<|im_start|>assistant\n"
            },
            {"prompt": prompt, "multi_modal_data": {"video": video}},
        ]
    else:
        batch_data = {"prompt": prompt, "multi_modal_data": {"video": video}}
else:
    # Prompt example: https://docs.vllm.ai/en/v0.6.2/getting_started/examples/offline_inference_vision_language.html
    if "Qwen2" in mdl:
        llm = LLM(
            model=mdl,
            enforce_eager=False,
            max_model_len=32768,
            max_num_seqs=5,
            gpu_memory_utilization=args.gpu_mem_usage,
            limit_mm_per_prompt={"image": 1},
        )

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question_list[0]}<|im_end|>\n<|im_start|>assistant\n"

        if multiple_prompt:
            if args.text_only:
                batch_data = [
                    {
                        "prompt": f"<|im_start|>system\nYou are a nice person.<|im_end|>\n<|im_start|>user\n{question_list[1]}<|im_end|>\n<|im_start|>assistant\n"
                    },
                    {
                        "prompt": f"<|im_start|>system\nYou are a nice person.<|im_end|>\n<|im_start|>user\n{question_list[2]}<|im_end|>\n<|im_start|>assistant\n"
                    },
                    {
                        "prompt": f"<|im_start|>system\nYou are a nice person.<|im_end|>\n<|im_start|>user\n{question_list[3]}<|im_end|>\n<|im_start|>assistant\n"
                    },
                ]
            else:
                batch_data = [
                    {"prompt": prompt, "multi_modal_data": {"image": image}},
                    {
                        "prompt": f"<|im_start|>system\nYou are a nice person.<|im_end|>\n<|im_start|>user\n{question_list[1]}<|im_end|>\n<|im_start|>assistant\n"
                    },
                    {"prompt": prompt, "multi_modal_data": {"image": image}},
                ]
                if args.random_img_size:
                    batch_data2 = [
                        {"prompt": prompt, "multi_modal_data": {"image": image}},
                        {
                            "prompt": f"<|im_start|>system\nYou are a nice person.<|im_end|>\n<|im_start|>user\n{question_list[1]}<|im_end|>\n<|im_start|>assistant\n"
                        },
                        {"prompt": prompt, "multi_modal_data": {"image": image2}},
                    ]
        else:
            if args.text_only:
                batch_data = [
                    {
                        "prompt": f"<|im_start|>system\nYou are a nice person.<|im_end|>\n<|im_start|>user\n{question_list[1]}<|im_end|>\n<|im_start|>assistant\n"
                    }
                ]
            else:
                batch_data = {"prompt": prompt, "multi_modal_data": {"image": image}}
                if args.random_img_size:
                    batch_data2 = {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image2},
                    }

    elif "Llama-3.2" in mdl:
        llm = LLM(
            model=mdl,
            max_model_len=2048,
            max_num_seqs=64,
            tensor_parallel_size=1,
            num_scheduler_steps=32,
            max_num_prefill_seqs=4,
        )
        from vllm import TextPrompt

        batch_data = TextPrompt(prompt=f"<|image|><|begin_of_text|>{question_list[0]}")
        batch_data["multi_modal_data"] = {"image": image}
    else:
        print(f"{mdl} is not known model?")

for i in range(int(args.iter)):
    print(f"------ iteration: [{i}]")
    outputs = llm.generate(batch_data if i < 2 else batch_data2, sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
