from vllm import LLM
llm = LLM("meta-llama/Llama-3.1-8B-Instruct", quantization="inc")
...
# Call llm.generate on the required prompts and sampling params.
...
llm.llm_engine.model_executor.shutdown()

