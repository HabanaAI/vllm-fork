# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import nullcontext

import pytest
from vllm_test_utils import BlameResult, blame

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    V1 only supports xgrammar so this is irrelevant.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


def run_normal_opt125m(enforce_eager):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM without guided decoding as a baseline.
    llm = LLM(model="/mnt/weka/data/pytorch/llama3.2/Meta-Llama-3.2-1B",
              dtype="bfloat16",
              enforce_eager=enforce_eager,
              gpu_memory_utilization=0.3)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Destroy the LLM object and free up the GPU memory.
    del llm
    cleanup_dist_env_and_memory()


def run_normal(enforce_eager):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM without guided decoding as a baseline.
    llm = LLM(model="distilbert/distilgpt2",
              dtype="bfloat16",
              enforce_eager=enforce_eager,
              gpu_memory_utilization=0.3)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Destroy the LLM object and free up the GPU memory.
    del llm
    cleanup_dist_env_and_memory()


def run_lmfe(sample_regex, enforce_eager):
    # Create an LLM with guided decoding enabled.
    llm = LLM(model="distilbert/distilgpt2",
              enforce_eager=enforce_eager,
              dtype="bfloat16",
              guided_decoding_backend="lm-format-enforcer",
              gpu_memory_utilization=0.3)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(
        prompts=[
            f"Give an example IPv4 address with this regex: {sample_regex}"
        ] * 2,
        sampling_params=sampling_params,
        use_tqdm=True,
        guided_options_request=dict(guided_regex=sample_regex))

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_lazy_outlines(sample_regex, enforce_eager):
    """If users don't use guided decoding, outlines should not be imported.
    """
    # make sure outlines is not imported
    module_name = "outlines"
    # In CI, we only check finally if the module is imported.
    # If it is indeed imported, we can rerun the test with `use_blame=True`,
    # which will trace every function call to find the first import location,
    # and help find the root cause.
    # We don't run it in CI by default because it is slow.
    use_blame = False
    context = blame(
        lambda: module_name in sys.modules) if use_blame else nullcontext()
    with context as result:
        run_normal(enforce_eager)
        run_lmfe(sample_regex, enforce_eager)
    if use_blame:
        assert isinstance(result, BlameResult)
        print(f"the first import location is:\n{result.trace_stack}")
    assert module_name not in sys.modules, (
        f"Module {module_name} is imported. To see the first"
        f" import location, run the test with `use_blame=True`.")
