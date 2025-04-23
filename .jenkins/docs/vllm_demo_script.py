from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m")
llm.start_profile()
outputs = llm.generate(["San Francisco is a"])
llm.stop_profile()
print(outputs)