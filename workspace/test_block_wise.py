import os

os.environ["USE_MMAP"] = "1"
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"
os.environ["VLLM_MOE_N_SLICE"] = "4"
os.environ["VLLM_EP_SIZE"] = "1"


import torch
import vllm
import transformers
from vllm import LLM, SamplingParams
from neural_compressor.torch.utils import get_used_hpu_mem_MB, get_used_cpu_mem_MB, logger

hpu_mem_0 = get_used_hpu_mem_MB()
cpu_mem_0 = get_used_cpu_mem_MB()
is_moe=False


def build_dataloader(tokenizer):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    num_samples = 1
    least_tokens = 2048
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    dataset = dataset.shuffle(seed=1)

    num_sample = 0
    prompts = []
    for data in dataset:
        prompt = data['text']
        tokens = tokenizer(prompt, return_tensors="pt")
        if len(tokens.input_ids[0]) < least_tokens:
            continue
        num_sample += 1
        if num_sample > num_samples:
            break
        prompts.append(prompt)
    
    dataloader = DataLoader(prompts, batch_size=1, shuffle=False)
    return dataloader

if __name__ == "__main__":
    with torch.no_grad():
        model_name_or_path = "facebook/opt-125m"
        model_name_or_path = "/mnt/weka/data/pytorch/llama2/Llama-2-7b-hf/"
        model_name_or_path = "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"
        # model_name_or_path = "/software/users/xinhe3/models/DeepSeek-R1-BF16-mini"
        # model_name_or_path = "deepseek-ai/DeepSeek-V2-Lite"
        config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        print(config.torch_dtype)

        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=-1,
            max_tokens=1,
            truncate_prompt_tokens=1000,
        )
        llm = LLM(
            model=model_name_or_path,
            quantization="inc",
            tensor_parallel_size=1,
            trust_remote_code=True,
            weights_load_device="cpu",
            gpu_memory_utilization=0.01,
            max_model_len=1024,
        )
        logger.info(f"Initialized vLLM model, Used HPU memory: {round((get_used_hpu_mem_MB()-hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Initialized vLLM model, Used CPU memory: {round((get_used_cpu_mem_MB()-cpu_mem_0)/1024, 3)} GiB")
        # out = llm.generate("hi, ", sampling_params)
        llm.llm_engine.model_executor.preprocess_calibration()
        llm.generate("hi, what's your ", sampling_params)
        llm.generate("I'm an engineer of ", sampling_params)
        llm.llm_engine.model_executor.postprocess_calibration()
        logger.info(f"Calibrated vLLM model, Used HPU memory: {round((get_used_hpu_mem_MB()-hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Calibrated vLLM model, Used CPU memory: {round((get_used_cpu_mem_MB()-cpu_mem_0)/1024, 3)} GiB")
        llm.llm_engine.model_executor.shutdown()


# QUANT_CONFIG=measure.json python test_block_wise.py




# init_llm(prepared) -> calibration -> shutdown(finialize)


# init_llm(prepared) -> calibration -> shutdown(finialize)

