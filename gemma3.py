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

num_imgs = int(sys.argv[1])

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None
image_urls = []
if num_imgs == 1:
    image_urls = [Image.open('jr.png').convert("RGB")]
elif num_imgs == 2:
    image_urls = [Image.open('jr.png').convert("RGB") for _ in range(2)]
#question = "What is the name of the person in the form?"
question = XXXXX
model_name = "google/gemma-3-27b-it"
#model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
engine_args = EngineArgs(
    model=model_name,
    max_model_len=8192,
    max_num_seqs=2,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    enforce_eager=True,    
    limit_mm_per_prompt={"image": len(image_urls)},
)
placeholders = [{"type": "image", "image": url} for url in image_urls]
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
processor = AutoProcessor.from_pretrained(model_name)
prompt = processor.apply_chat_template(messages,
                                        tokenize=False,
                                        add_generation_prompt=True)
req_data = ModelRequestData(
    engine_args=engine_args,
    prompt=prompt,
    image_data=[url for url in image_urls],
)
engine_args = asdict(req_data.engine_args) 
llm = LLM(**engine_args)
sampling_params = SamplingParams(temperature=0.0,
                                    max_tokens=1024,
                                    stop_token_ids=req_data.stop_token_ids)
num_prompts = 1
for i in range(num_prompts):
    outputs = llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image_data
            },
        },
        sampling_params=sampling_params,
#    lora_request=req_data.lora_requests,
    )
    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)

