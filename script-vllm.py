import os
import torch
import requests
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- 推荐但可选：Gaudi 性能相关开关 ---
# os.environ.setdefault("HABANA_VISIBLE_DEVICES", "ALL")
# os.environ.setdefault("PT_HPU_ENABLE_LAZY_MODE", "2")  # 懒执行性能更好

MODEL_PATH = "aidc-ai/ovis2.5-2b"
max_new_tokens = 3072

# 1) 加载模型到 HPU（bf16 对 Gaudi 友好） -> vLLM
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    tensor_parallel_size=int(os.getenv("TP_SIZE", "1")),
)

# 额外：需要一个 tokenizer 来产出聊天模板字符串
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)

# 2) 构造多模态输入（图片走 CPU 解码，后面交给 vLLM）
img = Image.open(requests.get(
    "https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/TIlymOb86R6_Mez3bpmcB.png",
    stream=True
).raw).convert("RGB")


# 3) 走仓库自带的预处理（trust_remote_code） -> 用 chat_template 生成 prompt
# 注：模板里通常用 <image> 占位；这里保持你的“图 + 同一句 prompt”
prompt = tokenizer.apply_chat_template(
    [{
        "role": "user",
        "content": [
            {"type": "text", "text": "<image>"},
            {"type": "text", "text": "Calculate the sum of the numbers in the middle box in figure (c)."},
        ],
    }],
    tokenize=False,
    add_generation_prompt=True,
)


# 5) 生成（vLLM：传入 prompts 和 images）
sampling_params = SamplingParams(max_tokens=max_new_tokens)
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": [img]},  # 传一张图；多张就放多张
}, sampling_params=sampling_params)

# 6) 解码（vLLM 返回对象里直接拿文本）
response = outputs[0].outputs[0].text
print(response)
