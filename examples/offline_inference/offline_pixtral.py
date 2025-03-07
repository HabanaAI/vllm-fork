from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta

model_name = "mistralai/Pixtral-Large-Instruct-2411"
# model_name = "mistralai/Pixtral-12B-2409"

def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, 'r') as file:
        system_prompt = file.read()
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)

def main():
    image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
    # image_url = "https://d2opxh93rbxzdn.cloudfront.net/original/2X/4/40cfa8ca1f24ac29cfebcb1460b5cafb213b6105.png"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Which of the depicted countries has the best food? Which the second and third and fourth?",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    if "Pixtral-Large-Instruct-2411" in model_name:
        SYSTEM_PROMPT = load_system_prompt(model_name, "SYSTEM_PROMPT.txt")
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    sampling_params = SamplingParams(max_tokens=512)

    # note that running this model on GPU requires over 300 GB of GPU RAM

    llm = LLM(model=model_name, max_model_len=1024*8, config_format="mistral", load_format="mistral", tokenizer_mode="mistral", tensor_parallel_size=1, limit_mm_per_prompt={"image": 4})

    outputs = llm.chat(messages, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)

if __name__ == '__main__':
    main()