from vllm import LLM
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
import PIL
import multiprocessing

def main():
    # Load the image
    image = ImageAsset("stop_sign").pil_image

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95 )
    # Initialize the LLM with a multimodal model like LLaVA
    # llava-hf/llava-1.5-7b-hf
    # Qwen/Qwen2-VL-7B-Instruct
    # meta-llama/Llama-3.2-11B-Vision-Instruct -> /root/sasarkar/clean_model_garden/models--meta-llama--Llama-3.2-11B-Vision-Instruct
    llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct", enforce_eager=False)
    #llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct", enforce_eager=True)
    #llm = LLM(model="llava-hf/llava-1.5-7b-hf")
    #llm = LLM(model="/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5/", tensor_parallel_size=2,)
    # Create the prompt with image data
    # llava prompt
    #prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT: <|image|>"
    #prompt = "<image>" * 576 + ("\nUSER: What is the content of this image?\nASSISTANT:")
    # qwen2-vl prompt
    prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
    outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}})

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()