import os
os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ['VLLM_CONTIGUOUS_PA'] = 'false'
os.environ['VLLM_MLA_DISABLE_REQUANTIZATION']='1'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES']='true'
os.environ['PT_HPU_WEIGHT_SHARING']='0'
os.environ['VLLM_MLA_PERFORM_MATRIX_ABSORPTION']='0'
os.environ['VLLM_MTP_PRINT_ACCPET_RATE']='0'
os.environ['PT_HPU_LAZY_MODE']='1'
os.environ['VLLM_DELAYED_SAMPLING']='false'
#os.environ['VLLM_USE_V1']='1'


if __name__ == "__main__":

    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    model_name = "/home/HF_models/llama-3-8b"
    llm = LLM(model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            dtype="bfloat16",
            use_v2_block_manager=True,
            tensor_parallel_size=1,
            max_model_len=1024,
            num_scheduler_steps=1,
            gpu_memory_utilization=0.5,
            max_num_seqs=128,
            enable_chunked_prefill=True,
            max_num_batched_tokens=128,
            seed=2024)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

