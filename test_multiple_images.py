import requests
from PIL import Image
from io import BytesIO
##from examples.offline_inference.vision_language import run_qwen2_5_vl
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

def run_qwen2_5_vl(question: str):

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        ##disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    placeholder = "<|image_pad|>"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


def fetch_image(url):
    """Fetch an image from a URL."""
    img_question = "What is the content of this image?"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert("RGB")
        # Input image and question
        return image, img_question
    else:
        raise Exception(f"Failed to fetch image from {url}")

def test_image_processing(url):
    """Test the image processing function with an image from a URL."""
    try:
        image, question = fetch_image(url)
        llm, prompt, stop_token_ids = run_qwen2_5_vl(question)

        # We set temperature to 0.2 so that outputs can be different
        # even when all prompts are identical when running batch inference.
        sampling_params = SamplingParams(temperature=0.2,
                                         max_tokens=64,
                                         stop_token_ids=stop_token_ids)

        # Single inference
        inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            }

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)

        print(f"Pass: {url}")
    except Exception as e:
        print(f"Fail: {url}")
        print(f"Error processing {url}: {e}")

def main():
    # List of image URLs with different sizes
    image_urls = [
        "https://picsum.photos/32/32",    # Small image (32x32 pixels)
        "https://picsum.photos/200/200",  # Medium image (200x200 pixels)
        "https://picsum.photos/800/600",  # Large image (800x600 pixels)
        "https://picsum.photos/1920/1080" # Very large image (1920x1080 pixels)
    ]
    for url in image_urls:
        test_image_processing(url)

if __name__ == "__main__":
    main()