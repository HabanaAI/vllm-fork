#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Minimal DotsOCR offline inference on vLLM.
Image-only demo with correct prompt formatting for OCR tasks.
"""

import argparse

#from dots_ocr.utils.image_utils import fetch_image
from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

MODEL_NAME = "/home/disk6/HF_models/dotsocr"
TOKENIZER_NAME = "/home/disk6/HF_models/dotsocr"


def build_prompts(questions):
    """
    构建 OCR 任务的提示信息。此处问题对应 OCR 任务中的文本识别需求。
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME,
                                              trust_remote_code=True)
    # OCR任务中，问题改为针对图像内容的提问
    messages = [[{
        "role": "user",
        "content": f"<|img|><|imgpad|><|endofimg|>{q}"
    }] for q in questions]
    return tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)


def get_demo_data():
    """
    获取示例数据，返回待测试图像和 OCR 任务的提示问题。
    """
    # 在这里提供你想测试的图像路径
    #   image_path = "path_to_your_image.jpg"
    #   image = fetch_image(image_path)  # 使用 dotsocr 的图片加载工具
    image = Image.open('math.jpg').convert('RGB')
    questions = [
        "What is the content of this image?",  # 任务1：提取图像内容
        "Where is this image taken?",  # 任务2：如果需要识别图像地点
    ]
    return image, questions


def parse_args():
    """
    解析命令行参数，支持用户定义要使用的提示数量。
    """
    p = argparse.ArgumentParser("Minimal DotsOCR vLLM OCR demo")
    p.add_argument("--num-prompts", type=int, default=2)
    return p.parse_args()


def main():
    """
    主函数，执行模型推理并输出结果。
    """
    args = parse_args()
    image, questions = get_demo_data()
    prompts = build_prompts(questions)

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=max(1, args.num_prompts),
        limit_mm_per_prompt={"image": 1},
    )

    req_prompts = [prompts[i % len(prompts)] for i in range(args.num_prompts)]
    inputs = [{
        "prompt": p,
        "multi_modal_data": {
            "image": image
        }
    } for p in req_prompts]

    sampling = SamplingParams(temperature=0.2, max_tokens=64)
    outputs = llm.generate(inputs, sampling_params=sampling)

    print("-" * 50)
    for out in outputs:
        print(out.outputs[0].text.strip())  # 输出 OCR 结果
        print("-" * 50)


if __name__ == "__main__":
    main()
