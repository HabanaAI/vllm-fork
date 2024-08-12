import argparse
import contextlib
import random
import time

import torch

import os

random.seed(42)


from vllm.entrypoints.chat_utils import ConversationMessage, load_chat_template
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams


def generate_random_coding_question():
    questions = [
        "How do you reverse a string in Python?",
        "Write a Python function to check if a number is prime.",
        "Explain the difference between a list and a tuple in Python.",
        "Write a Python script to merge two dictionaries.",
        "What is the use of the 'with' statement in Python?",
        "Write a Python program to find the factorial of a number using recursion.",
        "How do you handle exceptions in Python?",
        "Write a Python class to implement a basic calculator.",
        "Explain the concept of decorators in Python.",
        "Write a Python function to sort a list of tuples based on the second element."
    ]
    return random.choice(questions)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_conversation():
    assistant_adjectives = ['an enthusiastic', 'a knowledgeable', 'a curious', 'a patient', 'an insightful', 'a clever']
    assistants = ['coder', 'developer', 'programmer', 'software engineer', 'tech enthusiast', 'Python expert']
    sys_message = ConversationMessage(role='system', content=f'You are {random.choice(assistant_adjectives)} {random.choice(assistants)}. You enjoy sharing your knowledge about Python programming.')
    user_message = ConversationMessage(role='user', content=generate_random_coding_question())
    return [sys_message, user_message]

def main():
    parser = argparse.ArgumentParser(
                    prog='vllm_offline_test',
                    description='Tests vLLM offline mode')
    parser.add_argument('-n', '--batch-size', type=int, default=4)
    parser.add_argument('-w', '--world-size', type=int, default=1)
    parser.add_argument('-m', '--model-path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('-e', '--enforce-eager', action='store_true')
    parser.add_argument('-p', '--profiling', action='store_true')
    parser.add_argument('-g', '--gpu-mem-utilization', type=float, default=0.5)
    parser.add_argument('-b', '--block-size', type=int, default=128)
    parser.add_argument('-l', '--max-seq-len-to-capture', type=int, default=2048)
    parser.add_argument('--chat-template', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--warmup', type=int, default=0, help='Number of warmup iterations to skip')
    parser.add_argument('--fp8', type=str2bool, nargs='?', const=True, default=False, help='Boolean flag to enable fp8')
    parser.add_argument('--measure', type=str2bool, nargs='?', const=True, default=False, help='Boolean flag to enable fp8 measurements')

    args = parser.parse_args()

    batch_size = args.batch_size
    world_size = args.world_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    temperature = args.temperature
    block_size = args.block_size
    enforce_eager = args.enforce_eager
    gpu_mem_utilization = args.gpu_mem_utilization
    profiling = args.profiling
    max_tokens = args.max_tokens
    provided_chat_template = args.chat_template
    warmup = args.warmup

    model_path = args.model_path

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    os.environ['EXPERIMENTAL_WEIGHT_SHARING'] = "0"
    os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = "true"

    if args.measure:
        print("Starting the measurements:")
        os.environ.setdefault('QUANT_CONFIG', "./test_jsons/test_measure.json")
        llm = LLM(model=model_path, enforce_eager=enforce_eager, swap_space=0, dtype=torch.bfloat16, tensor_parallel_size=world_size, block_size=block_size,
                    max_num_seqs=batch_size, gpu_memory_utilization=gpu_mem_utilization, max_seq_len_to_capture=max_seq_len_to_capture, max_model_len=max_seq_len_to_capture,
                    quantization="inc")
    elif args.fp8:
        print("Running in fp8:")
        os.environ.setdefault('QUANT_CONFIG', "./test_jsons/test_hw_quant.json")
        llm = LLM(model=model_path, enforce_eager=enforce_eager, swap_space=0, dtype=torch.bfloat16, tensor_parallel_size=world_size, block_size=block_size,
                  max_num_seqs=batch_size, gpu_memory_utilization=gpu_mem_utilization, max_seq_len_to_capture=max_seq_len_to_capture, max_model_len=max_seq_len_to_capture,
                  quantization="inc",  kv_cache_dtype="fp8_inc", weights_load_device="cpu")
    else:
        # Create an LLM.
        print("Running in bf16:")
        llm = LLM(model=model_path, enforce_eager=enforce_eager, swap_space=0, dtype=torch.bfloat16, tensor_parallel_size=world_size, block_size=block_size,
                max_num_seqs=batch_size, gpu_memory_utilization=gpu_mem_utilization, max_seq_len_to_capture=max_seq_len_to_capture, max_model_len=max_seq_len_to_capture)

    chat_template = load_chat_template(provided_chat_template)
    tokenizer = llm.llm_engine.get_tokenizer()
    conversations = [get_conversation() for _ in range(batch_size)]
    prompts = [tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template,
            ) for conversation in conversations]

    # Warmup iterations
    for _ in range(warmup):
        _ = llm.generate(prompts, sampling_params)

    # Measure performance for the final iteration
    start = time.time()
    profile_ctx = contextlib.nullcontext()
    if profiling:
        profile_ctx = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('vllm_logs', use_gzip=True), with_stack=True, with_modules=True, record_shapes=False, profile_memory=False)
    with profile_ctx as profiler:
        outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    if args.measure:
        llm.finish_measurements()

    # Print the outputs.
    total_time = end - start
    num_tokens = 0
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        tokens = output.outputs[0].token_ids
        generated_text = output.outputs[0].text
        num_tokens += len(tokens)
        print("Conversation:")
        for message in conversations[idx]:
            print(f'\t{message["role"]!r}: {message["content"]!r}')
        print(f"{idx}-Prompt:\n\t{prompt!r}\nGenerated text:\n\t{generated_text!r}\ngen_len: {len(tokens)}\n")
    print(f"Gen tput: {num_tokens/total_time:.3f} tokens/s; Total tokens: {num_tokens}; total time: {total_time:.3f} seconds")

if __name__ == '__main__':
    main()
