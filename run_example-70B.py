from vllm import LLM, SamplingParams
import os
 
os.environ['VLLM_SKIP_WARMUP'] = 'true'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES']='true'
os.environ['PT_HPU_WEIGHT_SHARING']='0'

os.environ['VLLM_DELAYED_SAMPLING'] = 'true'
os.environ['VLLM_SOFTMAX_CONST_NORM'] = 'true'
os.environ['QUANT_CONFIG'] = 'quant_files/inc_main/meta-llama-3.1-70b-instruct/maxabs_quant_g3.json'
 
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
 
if __name__ == "__main__":
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
 
    model_path = "/data/Llama-3.1-70B-Instruct/"
    #model_path = "/data/Llama-3.1-405B-Instruct/"
 
    llm = LLM(model=model_path,
            #enforce_eager=True,
            dtype="bfloat16",
            use_v2_block_manager=True,
            tensor_parallel_size=1,
            max_model_len=16384,
            distributed_executor_backend='mp',
            gpu_memory_utilization=0.9,
            kv_cache_dtype="fp8_inc",
            quantization='inc',
            weights_load_device="cpu",)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")