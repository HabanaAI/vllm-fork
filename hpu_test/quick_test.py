import time
import os

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ["PT_HPUGRAPH_DISABLE_TENSOR_CACHE=False "] = "False"

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
#model2_5 = "Qwen/Qwen2.5-VL-3B-Instruct"
#model = model2_5
model = "AIDC-AI/Ovis2.5-9B"
enforce_eager = False

kwargs={'trust_remote_code':'True'}
llm = LLM(
    model=model,
    max_model_len=8000,#32768,
    max_num_seqs=5,
    limit_mm_per_prompt={
        "image": 1,
        "video": 0
    },
    enforce_eager=enforce_eager,
    **kwargs
)

prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
prompt2 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Can you please, tell me what do you see in this?<|im_end|>\n<|im_start|>assistant\n'
prompt_text_only = '<|im_start|>system\nYou are a nice sytem.<|im_end|>\n<|im_start|>user\nTell me about you.<|im_end|>\n<|im_start|>assistant\n'
prompt_question = 'Who is Barack Obama?\n'
prompt_new = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Can you tell me the street address on this image, without the city name and the state?.<|im_end|>\n<|im_start|>assistant\n'

#OVIS
prompt_new='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>Please respond in English. Can you tell me the street address on this image, without the city name and the state?.<|im_end|>\n<|im_start|>assistant\n'
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
'''

prompt = '<|im_start|>user \n<image>\nCan you tell me the street address on this image, without the city name and the state?.<|im_end|>\n<|im_start|>assistant\n'
batch_data = [
   {"prompt": prompt, "multi_modal_data": {"image": test_image}},
]
'''
sampling_params = SamplingParams(temperature=0.0, max_tokens=20, seed=2025)

REPS_TOTAL = len(batch_data)
throughputs = []
durations = []
answers = []

from transformers import AutoTokenizer
placeholder = "<image>"
question = "Can you tell me the street address on this image, without the city name and the state?"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
messages = [
    [{"role": "user", "content": f"{placeholder}\n{question}"}]
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True#, enable_thinking=False
)



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

# print(f" ==================== batch data ==================== ")

# throughputs = []
# durations = []
# answers = []

# start_time = time.time()
# outputs = llm.generate(batch_data, sampling_params=sampling_params)
# elapsed_time = time.time() - start_time

# num_tokens = sum(len(o.prompt_token_ids) for o in outputs)
# throughput = num_tokens / elapsed_time

# durations.append(elapsed_time)
# throughputs.append(throughput)

# for o in outputs:
#     generated_text = o.outputs[0].text
#     answers.append(generated_text)
#     print(" > OUTPUT", generated_text)

# for ans in answers:
#   print(" ", ans)

# avg_throughputs = sum(throughputs) / REPS_TOTAL
# print(f"Throughputs: {avg_throughputs:0.2f}")

#  ===================== Answers ======================
#   The image depicts a street scene in what appears to be a Chinatown area, characterized by traditional Chinese
#   I am a large language model created by Alibaba Cloud. I am called Qwen. I am designed
#   The image shows a beautiful scene of cherry blossoms in full bloom, with pink flowers covering the branches
#   Barack Obama is an American politician who served as the 44th President of the United States
# Throughputs: 100.17 (No warmup) -> (Throughputs: 337.50 with warmUp)


# GPU 
#  ===================== Answers ======================
#   The image depicts a street scene in what appears to be an Asian neighborhood, likely in a Chinatown
#   I am a large language model created by Alibaba Cloud. I am called Qwen. I am designed
#   The image shows a beautiful scene of cherry blossoms in full bloom, with pink flowers covering the branches
#   Barack Obama is an American politician who served as the 44th President of the United States
# Throughputs: 725.20
