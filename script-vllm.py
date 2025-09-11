import os
import requests
from PIL import Image
from vllm import LLM, SamplingParams


MODEL_PATH = "aidc-ai/ovis2.5-2b"
tp = int(os.getenv("TP_SIZE", "1"))

# 1) 加载模型到 HPU（bf16 对 Gaudi 友好） -> vLLM
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    tensor_parallel_size=tp,
)

# 2) 构造多模态输入（图片走 CPU 解码，后面交给 vLLM）
img = Image.open(requests.get(
    "https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/TIlymOb86R6_Mez3bpmcB.png",
    stream=True
).raw).convert("RGB")

# 3) 直接手写 prompt，显式放 1 个 <image> 占位
prompt = "<image>\nCalculate the sum of the numbers in the middle box in figure (c)."

# 4) 生成（单图 -> images=[[img]]；与占位符一一对应）
sampling_params = SamplingParams(max_tokens=512)
outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": [img]},
    },
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)