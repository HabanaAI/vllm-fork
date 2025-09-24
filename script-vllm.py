#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Minimal multimodal test on vLLM + Gaudi (HPU) using Ovis2.5.
- Loads the model on HPU in bf16
- Feeds one image + one text prompt
- Prints the generated answer

Notes:
- This uses vLLM's Python API (no HTTP server).
- We pass the image via `multi_modal_data` and the text contains a "<image>" marker.
- Gaudi-specific env flags are optional but recommended for performance.
"""

import os
import io
import requests
from PIL import Image

# --- Optional but recommended for Gaudi performance ---
#os.environ.setdefault("HABANA_VISIBLE_DEVICES", "ALL")          # expose all HPUs
#os.environ.setdefault("PT_HPU_ENABLE_LAZY_MODE", "2")           # lazy mode can improve perf
#os.environ.setdefault("VLLM_USE_V1", "0")                        # keep v0 engine if your stack prefers it

from vllm import LLM, SamplingParams
import torch

# -------- Configs --------
# You can switch to your local path, e.g. "/home/.../Ovis2.5-9B"
MODEL_PATH = "/home/disk7/HF_MODELS/Ovis2.5-2B"

# Prompt & image from the official example you shared
IMAGE_URL = "https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/TIlymOb86R6_Mez3bpmcB.png"
QUESTION = "Calculate the sum of the numbers in the middle box in figure (c)."


# -------- Utilities --------
def load_image_from_url(url: str) -> Image.Image:
        """Load a PIL image from an HTTP URL into CPU memory."""
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

def hpu_available() -> bool:
        """Return True if torch.hpu exists and is available."""
        return hasattr(torch, "hpu") and torch.hpu.is_available()

def resize_for_vit(img: Image.Image, max_edge=448):
    w, h = img.size
    scale = max(w, h) / float(max_edge)
    if scale > 1:
        w2, h2 = int(round(w/scale)), int(round(h/scale))
        img = img.resize((w2, h2), Image.BICUBIC)
    return img


def main():
    
    # Sampling: adjust to your taste
    sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=512,       # generation budget
            stop=None
    )

    # -------- Build vLLM engine on HPU --------
    # Important: vLLM detects HPU if device="hpu" or device="auto" with HPU present.
    device_choice = "hpu" if hpu_available() else "auto"
    
    llm = LLM(
            model=MODEL_PATH,
            # tokenizer defaults to the same repo unless you override it
            trust_remote_code=True,          # Ovis2.5 custom processors
            dtype="bfloat16",                # Gaudi-friendly dtype
            device=device_choice,            # force HPU if available
            tensor_parallel_size=1,          # adjust if you have multiple HPUs and shard the model
            # Multimodal knobs (commonly safe defaults)
            # You may tune/extend according to your repo version:
            max_model_len=4096,       # ← 降到 4k/8k，先稳住
            block_size=8,             # ← OOM/碎片还在就开小一点
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 1},    # allow 1 image per prompt
            disable_mm_preprocessor_cache=False  # memoize preprocessing for repeated runs
    )
    
    # -------- Prepare a multimodal request --------
    # vLLM multimodal accepts a dict with:
    # - "prompt": text containing a "<image>" marker
    # - "multi_modal_data": {"image": [PIL.Image or file paths]}
    img = load_image_from_url(IMAGE_URL)
    img = resize_for_vit(img, max_edge=448)
    prompt = f"<image>\n{QUESTION}"
    
    requests_batch = [{
            "prompt": prompt,
            "multi_modal_data": {"image": [img]},
    }]
    
    # -------- Generate --------
    # vLLM returns a list of RequestOutput objects. Each contains .outputs (list of candidates).
    outputs = llm.generate(requests_batch, sampling_params)
    
    # -------- Print result --------
    for i, out in enumerate(outputs):
            # Usually one candidate is enough; pick the first
            text = out.outputs[0].text if out.outputs else ""
            print("-" * 50)
            print(f"Request #{i}")
            print("Prompt:", prompt)
            print("Answer:", text)
            print("-" * 50)
        
    """
    Troubleshooting tips (Gaudi + multimodal):
    - If you see the model repeating the prompt, ensure the <image> marker is present
        and that `multi_modal_data` correctly carries the PIL image.
    - Make sure your vLLM version supports your Ovis repo's remote code.
    - Keep bf16 on HPU; fp16 is not recommended.
    - If you run into capacity/shape issues, try lowering `max_tokens` or set
        `limit_mm_per_prompt={"image": 1}` explicitly (as above).
    - If you want to compare CPU/GPU vs HPU, change `device` to "auto" and run on a
        different machine; the request format stays the same.
    """
        
        
if __name__ == "__main__":
    main()