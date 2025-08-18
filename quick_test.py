import time
import os

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["VLLM_QWEN_SPLIT_GRAPHS"] = "true"

from vllm import LLM
from vllm.assets.image import ImageAsset
from vllm import SamplingParams
from PIL import Image

# Load the image
newsize = (537, 612)
image = ImageAsset("stop_sign").pil_image
image = image.resize(newsize)
newsize = (672, 784)
image2 = ImageAsset("cherry_blossom").pil_image
image2 = image2.resize(newsize)
newsize = (1236, 1236)
image3 = ImageAsset("cherry_blossom").pil_image
image3 = image3.resize(newsize)

test_image = Image.open("test_image.jpg").resize((560, 784), Image.Resampling.LANCZOS)
test_image_d1 = Image.open("test_image.jpg").resize((580, 784), Image.Resampling.LANCZOS)
test_image_d2 = Image.open("test_image.jpg").resize((560, 840), Image.Resampling.LANCZOS)
test_image_d3 = Image.open("test_image.jpg").resize((580, 840), Image.Resampling.LANCZOS)
#test_image_distorted = Image.open("test_image.jpg").resize((308, 224), Image.Resampling.LANCZOS)


# Initialize the LLM with a multimodal model like LLaVA
# Qwen/Qwen2-VL-7B-Instruct
model2_5 = "Qwen/Qwen2.5-VL-3B-Instruct"
model = model2_5
enforce_eager = False

llm = LLM(
    model=model,
    max_model_len=32768,
    max_num_seqs=5,
    limit_mm_per_prompt={
        "image": 1,
        "video": 0
    },
    enforce_eager=enforce_eager,
)

prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
prompt2 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Can you please, tell me what do you see in this?<|im_end|>\n<|im_start|>assistant\n'
prompt_text_only = '<|im_start|>system\nYou are a nice sytem.<|im_end|>\n<|im_start|>user\nTell me about you.<|im_end|>\n<|im_start|>assistant\n'
prompt_question = 'Who is Barack Obama?\n'
prompt_new = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Can you tell me the street address on this image, without the city name and the state?.<|im_end|>\n<|im_start|>assistant\n'

batch_data = [
   {"prompt": prompt_new, "multi_modal_data": {"image": test_image}},
   {"prompt": prompt_new, "multi_modal_data": {"image": test_image_d1}},
   {"prompt": prompt_new, "multi_modal_data": {"image": test_image_d2}},
   {"prompt": prompt_new, "multi_modal_data": {"image": test_image_d3}},
    # {"prompt": prompt, "multi_modal_data": {"image": image}},
    # {"prompt": prompt_text_only},
    # {"prompt": prompt2, "multi_modal_data": {"image": image2}},
    # {"prompt": prompt_question},
    # {"prompt": prompt, "multi_modal_data": {"image": image3}},
    # {"prompt": prompt_question},
    # {"prompt": prompt, "multi_modal_data": {"image": image3}},
    # {"prompt": prompt_question},        
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=20, seed=2025)

REPS_TOTAL = len(batch_data)
throughputs = []
durations = []
answers = []
for i in range(REPS_TOTAL):
    print(f" ===================== ITER {i} ====================== ")

    start_time = time.time()
    outputs = llm.generate(batch_data[i], sampling_params=sampling_params)
    elapsed_time = time.time() - start_time

    num_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    throughput = num_tokens / elapsed_time
    
    durations.append(elapsed_time)
    throughputs.append(throughput)

    for o in outputs:
        generated_text = o.outputs[0].text
        answers.append(generated_text)
        print(" > OUTPUT", generated_text)

print(f" ===================== Answers ====================== ")
for ans in answers:
  print(" ", ans)

avg_throughputs = sum(throughputs) / REPS_TOTAL
print(f"Throughputs: {avg_throughputs:0.2f}")

