from vllm import LLM
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from PIL import Image
import time
import cv2
import random
import argparse
import sys

MODALITY_VIDEO = "video"
MODALITY_IMAGE = "image"
MODALITY_TEXT = "text"


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


def get_images(index, random_shuffle=False):
    image_path = [
        "images/snowscat-H3oXiq7_bII-unsplash.jpg",
        "images/1920x1080.jpg",
        "images/200x200.jpg",
        "images/32x32.jpg",
        "images/800x600.jpg",
        "images/snowscat-H3oXiq7_bII-unsplash.jpg",
        "images/1920x1080.jpg",
        "images/200x200.jpg",
        "images/32x32.jpg",
        "images/800x600.jpg",
    ]
    if random_shuffle:
        random.shuffle(image_path)
    image_idx = index % len(image_path)
    return image_path[image_idx]


def get_multi_modal_prompt(args, modality, index=0):
    if modality == MODALITY_IMAGE:
        image_prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>"
            f"Describe this image"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        filename = get_images(index, args.shuffle_images)
        filename2 = get_images(index + 1, args.shuffle_images)
        image = Image.open(filename)
        image2 = Image.open(filename2)
        if args.image_width and args.image_height:
            image = image.resize((args.image_width, args.image_height))
        prompts = [
            {"prompt": image_prompt, "multi_modal_data": {"image": image}},
        ]
        if args.random_img_size:
            rand_width = random.randint(
                int(0.5 * args.image_width), 1 * args.image_width
            )
            rand_height = random.randint(
                int(0.5 * args.image_height), 1 * args.image_height
            )
            # rand_width = 112*int(rand_height/112)
            # rand_height = 112*int(rand_height/112)
            rand_width = 1008
            rand_height = 1008
            image2 = image2.resize((rand_width, rand_height))
            prompts = [
                {"prompt": image_prompt, "multi_modal_data": {"image": image}},
                {"prompt": image_prompt, "multi_modal_data": {"image": image}},
                {"prompt": image_prompt, "multi_modal_data": {"image": image2}},
                {"prompt": image_prompt, "multi_modal_data": {"image": image2}},
            ]
        if args.two_images_prompt:
            image_prompt_with_2 = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|>"
                f"<|image_pad|><|vision_end|>"
                f"Identify the similarities between these images."
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            image2 = image2.resize((140, 140))
            prompts = [
                {
                    "prompt": image_prompt_with_2,
                    "multi_modal_data": {"image": [image, image2]},
                }
            ]
        return prompts[index % len(prompts)]
    if modality == MODALITY_VIDEO:
        video_prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            f"<|vision_start|><|video_pad|><|vision_end|>"
            f"Describe this video."
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        video = sample_frames(args.video, 50)
        return {"prompt": video_prompt, "multi_modal_data": {"video": video}}
    if modality == MODALITY_TEXT:
        question_list = [
            "Tell me about future of global warming.",
            "Can you describe Bernoulli's Equation?",
            "How does deepspeed work?",
        ]
        return {"prompt": question_list[index % len(question_list)]}
    print(f"Unknow image/Video Input {args}")
    exit


def get_llm(args):
    model = args.model
    assert "Qwen2" in model
    return LLM(
        model=model,
        enforce_eager=False,
        max_model_len=32768,
        max_num_seqs=5,
        gpu_memory_utilization=args.gpu_mem_usage,
        limit_mm_per_prompt={
            "image": args.limit_mm_image,
            "video": args.limit_mm_video,
        },
    )


def get_modality_from_args(args):
    if args.text_only:
        return MODALITY_TEXT
    if args.image:
        return MODALITY_IMAGE
    if args.video:
        return MODALITY_VIDEO
    return MODALITY_TEXT


def prepare_input_data(args):
    first_modality = get_modality_from_args(args)
    input_data = [get_multi_modal_prompt(args, first_modality, index=0)]
    print(f"PROMPT is modal {first_modality}")
    modalities = [first_modality]
    if args.mix_prompts:
        modalities.append(MODALITY_TEXT)  # just mix with text for now
    for prompt_index in range(args.prompts - 1):
        i = prompt_index + 1
        print(f"PROMPT is modal {modalities[i % len(modalities)]}")
        new_input_data = get_multi_modal_prompt(
            args,
            modalities[i % len(modalities)],
            index=i,
        )
        input_data.append(new_input_data)
    return input_data


def validate_args(args):
    print(f"args: {args}")
    invalid_args = False
    if args.mix_prompt_lenght and args.multiple_prompts:
        invalid_args = True
    if invalid_args:
        print(
            f"Validaiton error: both mix_prompt_lenght and multiple_prompts agrs are passed, which are incompatible. Only use one"
        )
        sys.exit()


def main(args) -> int:
    validate_args(args)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    llm = get_llm(args)
    print(" ========= LLM started =========")

    input_data = prepare_input_data(args)
    print(f"Input data:", input_data)


    for i in range(args.iter):
        print(f"------ iteration: [{i}]")

        if args.multiple_prompts:
            iter_input_data = input_data
            print(f" -- Run batch with {len(input_data)} prompts")
        elif args.mix_prompt_lenght:
            iter_lenght = 2 if i != 2 else 1
            rand_idx = random.randint(0, int(len(input_data) / 2))
            # rand_idx = 0
            # : iter_lenght = 2 else: iter_lenght =1
            iter_input_data = input_data[rand_idx : iter_lenght + rand_idx]
            print(
                f" -- Run prompt lenght: {iter_lenght} with prompt: {iter_input_data}"
            )
        else:
            iter_input_data = input_data[i % len(input_data)]
            print(f" -- Run single prompt: {iter_input_data}")

        start_time = time.time()
        outputs = llm.generate(iter_input_data, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time

        num_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        throughput = num_tokens / elapsed_time
        print(f"-- generate time = {elapsed_time:0.2f}, num_token ={num_tokens}, throughput = {throughput:0.2f}")

        for i, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            print(f"[{i}/{len(outputs)}]: {generated_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument(
        "-m",
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="model name or path",
    )
    parser.add_argument("-i", "--image", action="store_true", help="Image input")
    parser.add_argument("-v", "--video", help="Video Input")
    parser.add_argument(
        "-t", "--text_only", action="store_true", help="Text only pormpts"
    )
    parser.add_argument("--image_width", type=int, help="Image width size")
    parser.add_argument("--image_height", type=int, help="Image width size")
    parser.add_argument(
        "--two_images_prompt", action="store_true", help="Prompt with two images"
    )
    parser.add_argument(
        "--shuffle_images", action="store_true", help="shuffle the image list"
    )
    parser.add_argument(
        "--mix_prompts",
        action="store_true",
        help="Run prompts with multiple modalities",
    )
    parser.add_argument(
        "--multiple_prompts", action="store_true", help="Run multiple prompts"
    )
    parser.add_argument(
        "--mix_prompt_lenght", action="store_true", help="Run multiple prompts"
    )
    parser.add_argument(
        "--iter", type=int, default=1, help="number of iterations to run"
    )
    parser.add_argument("--prompts", type=int, default=4, help="number of prompts")
    parser.add_argument(
        "--limit_mm_image", type=int, default=1, help="limit num of images"
    )
    parser.add_argument(
        "--limit_mm_video", type=int, default=1, help="limit num of images"
    )
    parser.add_argument(
        "-gmu",
        "--gpu_mem_usage",
        type=float,
        default=0.9,
        help="GPU memory usage value",
    )
    parser.add_argument(
        "-ris", "--random_img_size", action="store_true", help="Randomly resize image"
    )

    # Parse the arguments
    args = parser.parse_args()
    main(args)
