# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.image_utils import fetch_image
from dots_ocr.utils.prompts import dict_promptmode_to_prompt


class DotsOCRParser:

    def __init__(self,
                 use_hf=False,
                 model_name='model',
                 ip='localhost',
                 port=8000,
                 temperature=0.1,
                 top_p=1.0):
        self.use_hf = use_hf
        self.model_name = model_name
        self.ip = ip
        self.port = port
        self.temperature = temperature
        self.top_p = top_p

    def _inference_with_hf(self, image, prompt):
        # 使用 Hugging Face 模型进行推理
        response = "HF model inference result"
        return response

    def _inference_with_vllm(self, image, prompt):
        # 使用 vLLM 进行推理
        response = inference_with_vllm(image,
                                       prompt,
                                       model_name=self.model_name,
                                       ip=self.ip,
                                       port=self.port,
                                       temperature=self.temperature,
                                       top_p=self.top_p)
        return response

    def get_prompt(self, prompt_mode, bbox=None, image=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            prompt = prompt + str(bbox)
        return prompt

    def parse_image(self, input_path, prompt_mode, bbox=None):
        origin_image = fetch_image(input_path)
        prompt = self.get_prompt(prompt_mode, bbox=bbox, image=origin_image)

        if self.use_hf:
            response = self._inference_with_hf(origin_image, prompt)
        else:
            response = self._inference_with_vllm(origin_image, prompt)

        return response


def main():
    # 设置要使用的 OCR 功能
    input_path = "path_to_your_image.jpg"
    prompt_mode = "prompt_layout_all_en"  # 选择不同的 prompt 模式
    # 示例：OCR区域（如果使用 "prompt_grounding_ocr"）
    bbox = [100, 150, 300, 400]

    # 实例化 DotsOCRParser 类
    parser = DotsOCRParser(use_hf=False)  # 根据需求选择使用 HF 或 vLLM
    result = parser.parse_image(input_path, prompt_mode, bbox)

    # 输出结果
    print("OCR Result:", result)


if __name__ == "__main__":
    main()
