import time
import os

os.environ["VLLM_SKIP_WARMUP"] = "false"
os.environ["VLLM_QWEN_SPLIT_GRAPHS"] = "true"

from vllm import LLM
from vllm.assets.image import ImageAsset
from vllm import SamplingParams

# Load the image
newsize = (560, 560)
image = ImageAsset("stop_sign").pil_image
image = image.resize(newsize)
newsize = (672, 784)
image2 = ImageAsset("cherry_blossom").pil_image
image2 = image2.resize(newsize)

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

batch_data = [
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    {"prompt": prompt2, "multi_modal_data": {"image": image2}},
    {"prompt": prompt2, "multi_modal_data": {"image": image2}},    
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

REPS_TOTAL = len(batch_data)
throughputs = []
durations = []
for i in range(REPS_TOTAL):
    print(f" ===================== ITER {i} ====================== ")

    start_time = time.time()
    outputs = llm.generate(batch_data[i], sampling_params=sampling_params)
    elapsed_time = time.time() - start_time

    num_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    throughput = num_tokens / elapsed_time
    
    durations.append(elapsed_time)
    throughputs.append(f"{throughput:0.2f}")
    print(f"-- generate time = {elapsed_time:0.2f}, num_token ={num_tokens}, throughput = {throughput:0.2f}")

    for o in outputs:
        generated_text = o.outputs[0].text
        print(" > OUTPUT", generated_text)

print(f"Average Time Spend is {sum(durations) / REPS_TOTAL}")
print(f"Throughputs: {throughputs}")

# HPU Performance
# Average Time Spend is 1.4734559655189514
# Throughputs: ['151.43', '437.90', '557.05', '831.30']

# GPU Performance
# Average Time Spend is 1.1665586233139038
# Throughputs: ['172.24', '464.36', '1080.36', '1120.60']
