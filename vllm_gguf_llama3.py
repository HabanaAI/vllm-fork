#from transformers import AutoTokenizer
import torch
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import load_chat_template


# In this script, we demonstrate how to pass input to the chat method:
conversation = [
   {
      "role": "system",
      "content": "You are a helpful assistant"
   },
   {
      "role": "user",
      "content": "Hello"
   },
   {
      "role": "assistant",
      "content": "Hello! How can I assist you today?"
   },
   {
      "role": "user",
      "content": "Write an essay about the importance of higher education.",
   },
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

#tokenizer = AutoTokenizer.from_pretrained("/software/users/rsshaik/frameworks/Models/llama3.1/Meta-Llama-3.1-8B/")


# Create an LLM.
llm = LLM(model="/software/users/rsshaik/frameworks/repos/GGUF/llama_3.1_8B/llama_3.1_FP16.gguf",
        tokenizer="unsloth/Meta-Llama-3.1-8B-Instruct",
        #max_model_len=32768,
        dtype=torch.bfloat16)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
chat_template_llama = load_chat_template("examples/tool_chat_template_llama3.1_json.jinja")

outputs = llm.chat(conversation, sampling_params, chat_template=chat_template_llama, chat_template_content_format="openai")

# Print the outputs.
for output in outputs:
   prompt = output.prompt
   generated_text = output.outputs[0].text
   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

