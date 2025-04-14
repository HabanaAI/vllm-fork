from vllm import LLM, SamplingParams
import os

os.environ['VLLM_SKIP_WARMUP'] = 'true'
os.environ['PT_HPU_LAZY_MODE'] = '1'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES']='true'
os.environ['PT_HPU_WEIGHT_SHARING']='0'
#os.environ['HABANA_LOGS']="vllm_inc_debug"
#os.environ["LOG_LEVEL_ALL"]="3"
os.environ['VLLM_MLA_DISABLE_REQUANTIZATION']='1'
os.environ["QUANT_CONFIG"] = "inc_quant_with_fp8kv_config.json"
#os.environ["LOGLEVEL"] = "DEBUG"

 
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
 
if __name__ == "__main__":
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
 
    # Create an LLM.
    #model_path = "/software/data/disk10/hub/models--deepseek-ai--deepseek-r1/snapshots/a157fa3d494497a54586a333a23df6c2143e7697/"
    model_path = "/software/data/disk7/models/DeepSeek-R1-G2/"
 
    llm = LLM(model=model_path,
            trust_remote_code=True,
            enforce_eager=True,
            dtype="bfloat16",
            use_v2_block_manager=True,
            max_model_len=1024,
            max_num_seqs=1,
            tensor_parallel_size=8,
            distributed_executor_backend='mp',
            gpu_memory_utilization=0.8,
            kv_cache_dtype="fp8_inc",
            seed=2024)
 
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    if os.environ.get("QUANT_CONFIG", None) is not None:
        llm.llm_engine.model_executor.shutdown()
