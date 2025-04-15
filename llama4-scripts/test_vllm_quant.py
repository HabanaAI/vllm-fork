import argparse

from vllm import LLM, SamplingParams

import os
# os.environ['HABANA_LOGS']='llama4_habana_logs'
# os.environ['LOG_LEVEL_ALL']='3'
os.environ['PT_HPU_LAZY_MODE']='1'
# os.environ['PT_HPUGRAPH_DISABLE_TENSOR_CACHE']='1'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES']='true'
os.environ['PT_HPU_WEIGHT_SHARING']='0'
os.environ['VLLM_SKIP_WARMUP']='true'

def test_text(llm):
    # Sample conversations.
    conversations = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {"role": "system", "content": "Always answer with emojis"},
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1024)

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.chat(conversations, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-----------------------------------")
        print(f"Prompt: {prompt!r}\nGenerated text:\n {generated_text}\n")

        # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
def test_images(llm):
    # image_url = "https://d2opxh93rbxzdn.cloudfront.net/original/2X/4/40cfa8ca1f24ac29cfebcb1460b5cafb213b6105.png"
    image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)
    # Perform multi-image inference using llm.chat()
    messages = [
         {
             "role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": "what countries are on the map, specify only those that are with a text",
                 },
                 {"type": "image_url", "image_url": {"url": image_url}},
             ],
         },
     ]
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-----------------------------------")
        print(f"Prompt: {prompt!r}\nGenerated text:\n {generated_text}\n")
def test_images(llm):
    # image_url = "https://d2opxh93rbxzdn.cloudfront.net/original/2X/4/40cfa8ca1f24ac29cfebcb1460b5cafb213b6105.png"
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    # image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)
    # Perform multi-image inference using llm.chat()
    messages = [
         {
             "role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": "can you describe what do you see in this image?",
                 },
                 {"type": "image_url", "image_url": {"url": image_url}},
             ],
         },
     ]
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-----------------------------------")
        print(f"Prompt: {prompt!r}\nGenerated text:\n {generated_text}\n")


def test_images_2(llm):
    image_urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)
    # Perform multi-image inference using llm.chat()
    outputs = llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Can you describe how these two images are similar, and how they differ?",
                    },
                    *(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                        for image_url in image_urls
                    ),
                ],
            }
        ],
        sampling_params=sampling_params,
    )

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-----------------------------------")
        print(f"Prompt: {prompt!r}\nGenerated text:\n {generated_text}\n")



def test_completion(llm):
    # Define sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling parameters object
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    # Generate texts from the prompts
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Run inference with a specified model ID."
    )

    # Add an argument for the model ID
    parser.add_argument(
        "--model_id",
        type=str,
        help="The Hugging Face model ID to use (e.g., 'll-re/Llama-4-Scout-17B-16E-Instruct').",
        #default="/software/stanley/models/llama4-final-v2/Llama-4-Scout-17B-16E-Instruct/",
        default="/data/models/Llama-4-Scout-17B-16E-Instruct/",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the model_id argument
    model_id = args.model_id

    # Print or use the model_id as needed
    print(f"Using model: {model_id}")
    llm = LLM(
        model=model_id,
        dtype='bfloat16',
        seed=42,
        enforce_eager=True,
        max_model_len=16384,
        quantization="inc",
        tensor_parallel_size=8,
        limit_mm_per_prompt={"image": 5},
        enable_expert_parallel=True,
    )
    # print("---------Now start Completion test-----------")
    # test_text(llm)
    print("---------Now start Image test-----------")
    test_images(llm)
    # print("---------Now start multi Image test-----------")
    # test_images_2(llm)
    # if "instruct" in model_id.lower():
    #     print("---------Now start Instruct test-----------")
    #     test_text(llm)
    #     test_images(llm)
    # else:
    #     print("---------Now start Completion test-----------")
    #     test_completion(llm)


if __name__ == "__main__":
    main()
