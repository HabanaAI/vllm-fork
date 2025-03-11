# SPDX-License-Identifier: Apache-2.0
"""Compare the outputs of HF and vLLM when using greedy sampling.

It tests chunked prefill. Chunked prefill can be enabled by
enable_chunked_prefill=True. If prefill size exceeds max_num_batched_tokens,
prefill requests are chunked.

Run `pytest tests/models/test_chunked_prefill_hpu.py`.
"""

import pytest

from ..models.utils import check_outputs_equal

MODELS = [
    "facebook/opt-125m",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("chunked_prefill_token_size", [
    4,
])
@pytest.mark.parametrize("enforce_eager", [True])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    chunked_prefill_token_size: int,
    enforce_eager: bool,
    tensor_parallel_size: int,
) -> None:
    """
    Checks exact match decode between huggingface model and vllm runner with
    chunked prefill.
    """

    max_num_seqs = chunked_prefill_token_size

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with vllm_runner(
            model,
            dtype=dtype,
            enable_chunked_prefill=True,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
