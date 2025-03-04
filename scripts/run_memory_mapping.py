import os

os.environ["USE_MMAP"] = "1"  # memory mapping flag

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["VLLM_MOE_N_SLICE"] = "4"
os.environ["VLLM_EP_SIZE"] = "1"

os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"


import torch
import vllm
import transformers
from vllm import LLM, SamplingParams
from neural_compressor.torch.utils import get_used_hpu_mem_MB, get_used_cpu_mem_MB, logger

hpu_mem_0 = get_used_hpu_mem_MB()
cpu_mem_0 = get_used_cpu_mem_MB()


if __name__ == "__main__":
    with torch.no_grad():
        model_name_or_path = "facebook/opt-125m"
        # model_name_or_path = "deepseek-ai/DeepSeek-V2-Lite"
        config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        print(config.torch_dtype)

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=10,
        )
        llm = LLM(
            model=model_name_or_path,
            # tensor_parallel_size=2,
            trust_remote_code=True,
            weights_load_device="cpu",
            gpu_memory_utilization=0.1,  # limit memory usage of graph and kv cache.
            dtype=config.torch_dtype,
        )
        model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model.model
        print(model)
        logger.info(f"Initialized vLLM model, Used HPU memory: {round((get_used_hpu_mem_MB()-hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Initialized vLLM model, Used CPU memory: {round((get_used_cpu_mem_MB()-cpu_mem_0)/1024, 3)} GiB")

        model = model.to('hpu')
        out = llm.generate("hi, ", sampling_params)
        print('Output:', out[0].outputs[0].text)
        logger.info(f"Test vLLM model generation, Used HPU memory: {round((get_used_hpu_mem_MB()-hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Test vLLM model generation, Used CPU memory: {round((get_used_cpu_mem_MB()-cpu_mem_0)/1024, 3)} GiB")

        llm.llm_engine.model_executor.shutdown()


