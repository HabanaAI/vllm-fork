from vllm import LLM
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
import PIL

# Load the image
image = ImageAsset("stop_sign").pil_image

sampling_params = SamplingParams(temperature=0.8, top_p=0.95 )
mld = "Qwen/Qwen2.5-VL-3B-Instruct"
mld = "Qwen/Qwen2-VL-7B-Instruct"
llm = LLM(model=mld, enforce_eager=False)

prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
