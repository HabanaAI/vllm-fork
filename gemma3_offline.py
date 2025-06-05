# Example adapted from: https://docs.vllm.ai/en/latest/getting_started/examples/vision_language_multi_image.html
#eg PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false VLLM_PROMPT_BS_BUCKET_MIN=1 VLLM_PROMPT_BS_BUCKET_STEP=1 VLLM_PROMPT_BS_BUCKET_MAX=1 VLLM_PROMPT_SEQ_BUCKET_MIN=384 VLLM_PROMPT_SEQ_BUCKET_MAX=384 VLLM_DECODE_BS_BUCKET_MIN=1 VLLM_DECODE_BS_BUCKET_MAX=1 VLLM_DECODE_BLOCK_BUCKET_MIN=512 VLLM_DECODE_BLOCK_BUCKET_MAX=512 python multi_image_example.py --model google/gemma-3-27b-it --tensor-parallel-size 1 --num-images 1
# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""
import os
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser
import numpy as np
"""Images are:
1. Medical form
2. Duck
3. Lion
4. Blue Bird
5. Whale
6. Starfish
7. Snail
8. Bee on Purple Flower
9. 2 Dogs
10. Orange Cat
11. Gerbil
12. Rabbit
13. Horse and foal
"""
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/2/26/Ultramarine_Flycatcher_%28Ficedula_superciliaris%29_Naggar%2C_Himachal_Pradesh%2C_2013_%28cropped%29.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg/2560px-Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d4/Starfish%2C_Caswell_Bay_-_geograph.org.uk_-_409413.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/69/Grapevinesnail_01.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Texas_invasive_Musk_Thistle_1.jpg/1920px-Texas_invasive_Musk_Thistle_1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Huskiesatrest.jpg/2880px-Huskiesatrest.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/1920px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/30/George_the_amazing_guinea_pig.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Oryctolagus_cuniculus_Rcdo.jpg/1920px-Oryctolagus_cuniculus_Rcdo.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/98/Horse-and-pony.jpg",
]


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image.Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

def load_model(model_name: str, tp_size: int, max_model_len:int, question: str, batch_size:int, image_urls: list[str]) -> ModelRequestData:
    assert len(image_urls) % batch_size == 0
    engine_args = EngineArgs(
    model=model_name,
    max_model_len=max_model_len,
    max_num_batched_tokens=max_model_len,
    max_num_seqs=len(image_urls),
    tensor_parallel_size=tp_size,
    #gpu_memory_utilization=0.9,
    enforce_eager=True,   
        limit_mm_per_prompt={"image": int(len(image_urls)/batch_size)},
    )

    processor = AutoProcessor.from_pretrained(model_name)
    if batch_size==1:
        placeholders = [{"type": "image", "image": (url)} for url in image_urls]
        messages = [{
            "role":
            "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": question
                },
            ],
        }]
        prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        #breakpoint()
        # [Image.open('jr.png').convert("RGB")]
        requests = {"prompt":prompt,"multi_modal_data":{"image":[fetch_image(url) if type(url)==str else url for url in image_urls]}}

    else:
        chunks = np.array_split(image_urls, batch_size)
        requests = []
        for chunk in chunks:
            print("chunk....", chunk)
            placeholders = [{"type": "image", "image": (url)} for url in chunk]
            messages = [{
                "role":
                "user",
                "content": [
                    *placeholders,
                    {
                        "type": "text",
                        "text": question
                    },
                ],
            }]
            prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            import numpy
            
            requests.append({"prompt":prompt,"multi_modal_data":{"image":[fetch_image(url) if (type(url)==str) or (type(url)==numpy.str_) else url for url in chunk]}})
            #breakpoint()
            #print()
    return engine_args,requests

def run_generate(model_name: str, tp_size: int, max_model_len: int, question: str, batch_size: int, image_urls: list[str]):
    engine_args,requests = load_model(model_name, tp_size, max_model_len, question, batch_size, image_urls)

    engine_args = asdict(engine_args)
    #breakpoint()
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=8192)

    #breakpoint()
    outputs = llm.generate(requests,
        sampling_params=sampling_params
    )
    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(len(o.outputs[0].token_ids))
        print(generated_text)
        print("-*." * 50)

def parse_args():
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models that support multi-image input for text '
        'generation')
    parser.add_argument('--model-name',
                        '-m',
                        type=str,
                        default="google/gemma-3-4b-it",
                        choices=['google/gemma-3-4b-it','google/gemma-3-27b-it'],
                        help='Huggingface "model_type".')
    parser.add_argument('--tensor-parallel-size',
                        '-tp',
                        type=int,
                        default=1,
                        help='tensor parallel size.')
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        choices=list(range(0,
                           len(IMAGE_URLS))),  # the max number of images
        default=2,
        help="Number of images to use for the demo.")
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batches in which the images will be sent.")
    parser.add_argument('--max-model-len',
                        '-ml',
                        type=int,
                        default=8192,
                        help='Max-Model-Len.')
    return parser.parse_args()


def main(args: Namespace):
    model = args.model_name
    tp_size = args.tensor_parallel_size
    max_model_len = args.max_model_len


    image_urls = [IMAGE_URLS[idx % len(IMAGE_URLS)] for idx, i in enumerate(range(args.num_images * args.batch_size))]

    '''
    if args.num_images==1:
        QUESTION = "Extract all information from the provided image and provide it in a json format:"
        #batch_size=1
    elif args.num_images==0:
        QUESTION = "You are an AI designed to generate extremely long, detailed worldbuilding content. Your goal is to write a fictional encyclopedia with at least 4000 words of content. Do not stop early. Start by describing a fictional planet in detail. Include: \n1. Geography and climate zones (with rich, varied description).\n2. The history of all civilizations, from ancient to modern times.\n3. Cultures, belief systems, and mythologies along with rich detail about where such beliefs came from.\n4. Political structures and conflicts along with their history.\n5. Technology and magic systems (if any) spanning the last 1000 years, highlighting significant discoveries and figures.\n6. Major historical events and characters along with their geneology.\n\n Be descriptive, verbose, and never summarize. Write in a factual tone like an academic encyclopedia. Begin your entry below:"
        #batch_size=1
    else:
        QUESTION = "What is the content of each image? Once done, write a story that combines them all."
    '''
    if args.num_images==0:
        QUESTION = "You are an AI designed to generate extremely long, detailed worldbuilding content. Your goal is to write a fictional encyclopedia with at least 4000 words of content. Do not stop early. Start by describing a fictional planet in detail. Include: \n1. Geography and climate zones (with rich, varied description).\n2. The history of all civilizations, from ancient to modern times.\n3. Cultures, belief systems, and mythologies along with rich detail about where such beliefs came from.\n4. Political structures and conflicts along with their history.\n5. Technology and magic systems (if any) spanning the last 1000 years, highlighting significant discoveries and figures.\n6. Major historical events and characters along with their geneology.\n\n Be descriptive, verbose, and never summarize. Write in a factual tone like an academic encyclopedia. Begin your entry below:"
    else:
        QUESTION = "What is the content of each image? Once done, write a story that combines them all."
    batch_size = args.batch_size

    run_generate(model, tp_size, max_model_len, QUESTION, batch_size, image_urls)


if __name__ == "__main__":
    args = parse_args()
    main(args)
