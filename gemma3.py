import os
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple, Optional
#from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser
import sys
import torch



num_imgs = sys.argv[1]
num_imgs = [int(i) for i in num_imgs.split(',')]


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None


def generate_prompt(n):
    if n == 1:
        question = xxx
    elif n == 5:
        question = "Explain what is happening in these images. Explain the characters you see in great detail, what they are doing. What is the overall story, and why is it funny. Do you identify the people in the image. Explain in such great details so that someone could recreate the images by reading your answer."
    elif n == 0:
        question = "You are an AI designed to generate extremely long, detailed worldbuilding content. Your goal is to write a fictional encyclopedia with at least 4000 words of content. Do not stop early. Start by describing a fictional planet in detail. Include: \n1. Geography and climate zones (with rich, varied description).\n2. The history of all civilizations, from ancient to modern times.\n3. Cultures, belief systems, and mythologies along with rich detail about where such beliefs came from.\n4. Political structures and conflicts along with their history.\n5. Technology and magic systems (if any) spanning the last 1000 years, highlighting significant discoveries and figures.\n6. Major historical events and characters along with their geneology.\n\n Be descriptive, verbose, and never summarize. Write in a factual tone like an academic encyclopedia. Begin your entry below:"
        question00 = "Generate a list by repeating this 10000 times: hello, world, cat, dog"
        question2 = "Generate all numbers from 1 to 6000, separate them with commas"
    else:
        assert False

    
    image_urls = []
    if n == 1:
        image_urls = [Image.open('jr.png').convert("RGB")]
    elif n == 5:
        image_urls = [Image.open(f'popeye{i}.png').convert("RGB") for i in range(5)] # split this into 5 images: https://en.wikipedia.org/wiki/Popeye#/media/File:First_Popeye_Strip,_East_Liverpool_Review,_1929-01-17,_p12.jpg
    #question = "What is the name of the person in the form?"

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    return {
        "role":
        "user",
        "content": [
            *placeholders,  # remove for gemma2
            {
                "type": "text",
                "text": question
            },
        ],
    }, image_urls

messages = [generate_prompt(n) for n in num_imgs]

max_num_imgs = max([len(img_url) for msg, img_url in messages])


model_name = "google/gemma-3-4b-it"
#model_name = 'google/gemma-2-2b-it'
#model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
engine_args = EngineArgs(
    model=model_name,
    max_model_len=8192,
    max_num_seqs=2,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    enforce_eager=False,    
    limit_mm_per_prompt={"image": max_num_imgs},  # remove for gemma2
    #dtype="float32"  ## remove
)

processor = AutoProcessor.from_pretrained(model_name)

req_datas = []
for message, image_urls in messages:
    prompt = processor.apply_chat_template([message],
                                        tokenize=False,
                                        add_generation_prompt=True)
    req_data = ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=[url for url in image_urls],
    )
    req_datas += [req_data]

#breakpoint()
engine_args = asdict(req_data.engine_args) 
llm = LLM(**engine_args)
#breakpoint()
sampling_params = SamplingParams(temperature=0.0,
                                    max_tokens=8192,
                                    stop_token_ids=req_data.stop_token_ids)
num_prompts = 1
for i in range(num_prompts):
    #breakpoint()
    # remove "multi_modal_data" for gemma2
    llm_gen_inp = [{ "prompt": req_data.prompt, "multi_modal_data": { "image": req_data.image_data }, } for req_data in req_datas]
    outputs = llm.generate(
        llm_gen_inp,
        sampling_params=sampling_params,
#    lora_request=req_data.lora_requests,
    )
    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        
        print("-" * 50)
    breakpoint()
    print()

