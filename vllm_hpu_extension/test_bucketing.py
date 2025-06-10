# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import pytest

import vllm_hpu_extension.bucketing.linear as linear
from vllm_hpu_extension.bucketing.exponential import (
    HPUExponentialBucketingContext)


@pytest.fixture
def hpu_linear_bucketing_context():
    ctx = linear.HPUBucketingContext(
        max_num_seqs=128,
        max_num_prefill_seqs=16,
        block_size=128,
        max_num_batched_tokens=4096,
        max_model_len=2048,
        max_prompt_seq=1024,
        use_merged_prefill=False,
    )
    ctx.num_hpu_blocks = 1024
    return ctx


@pytest.fixture
def hpu_exponential_bucketing_context():
    ctx = HPUExponentialBucketingContext(
        max_num_seqs=128,
        max_num_prefill_seqs=16,
        block_size=128,
        max_num_batched_tokens=4096,
        max_model_len=4096,
        use_merged_prefill=False,
    )
    ctx.num_hpu_blocks = 1024
    return ctx


@pytest.fixture
def bucketing_cls(request):
    return request.param


@pytest.fixture
def reset_singleton(bucketing_cls):
    singleton = type(bucketing_cls)
    instances = singleton._instances
    signatures = singleton._instances_argspec
    singleton._instances = {}
    singleton._instances_argspec = {}
    yield
    singleton._instances = instances
    singleton._instances_argspec = signatures


@pytest.mark.parametrize(
    "bucketing_cls",
    [linear.HPUBucketingContext, HPUExponentialBucketingContext],
    indirect=True)
def test_singleton_same_args(bucketing_cls, reset_singleton):
    context1 = bucketing_cls(
        max_num_seqs=128,
        max_num_prefill_seqs=8,
        block_size=128,
        max_num_batched_tokens=4096,
        max_model_len=2048,
        max_prompt_seq=1024,
        use_merged_prefill=False,
    )
    context2 = bucketing_cls(
        max_num_seqs=128,
        max_num_prefill_seqs=8,
        block_size=128,
        max_num_batched_tokens=4096,
        max_model_len=2048,
        max_prompt_seq=1024,
        use_merged_prefill=False,
    )
    assert context1 is context2


@pytest.mark.parametrize(
    "bucketing_cls",
    [linear.HPUBucketingContext, HPUExponentialBucketingContext],
    indirect=True)
def test_singleton_different_args(bucketing_cls, reset_singleton):
    context1 = bucketing_cls(
        max_num_seqs=128,
        max_num_prefill_seqs=32,
        block_size=128,
        max_num_batched_tokens=4096,
        max_model_len=2048,
        max_prompt_seq=1024,
        use_merged_prefill=False,
    )
    with pytest.raises(AssertionError) as e_info:
        context2 = bucketing_cls(
            max_num_seqs=256,
            max_num_prefill_seqs=16,
            block_size=128,
            max_num_batched_tokens=4096,
            max_model_len=2048,
            max_prompt_seq=1024,
            use_merged_prefill=False,
        )


@pytest.mark.parametrize(
    "bucketing_cls",
    [linear.HPUBucketingContext, HPUExponentialBucketingContext],
    indirect=True)
def test_singleton_get_instance_no_init(bucketing_cls, reset_singleton):
    with pytest.raises(AssertionError) as e_info:
        context = bucketing_cls.get_instance()


def test_singleton_get_instance():
    context1 = linear.HPUBucketingContext(max_num_seqs=128,
                                          max_num_prefill_seqs=16,
                                          block_size=128,
                                          max_num_batched_tokens=4096,
                                          max_model_len=2048,
                                          max_prompt_seq=1024,
                                          use_merged_prefill=False)
    context2 = linear.HPUBucketingContext.get_instance()
    assert context1 is context2


def test_read_bucket_settings(monkeypatch):
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_MIN", "1")
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_STEP", "16")
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_MAX", "64")
    config = linear.read_bucket_settings("prompt",
                                         "bs",
                                         min=1,
                                         step=32,
                                         max=128)
    assert config == [1, 16, 64]


def test_read_bucket_settings_empty_flags():
    config = linear.read_bucket_settings("prompt",
                                         "bs",
                                         min=1,
                                         step=32,
                                         max=128)
    assert config == [1, 32, 128]


def test_warmup_range():
    config = (2, 64, 128)
    result = linear.warmup_range(config)
    assert result == [2, 4, 8, 16, 32, 64, 128]


def test_generate_prompt_buckets():
    bs_bucket_config = (1, 4, 16)
    seq_bucket_config = (512, 512, 1024)
    max_num_batched_tokens = 2048
    buckets, omitted_buckets = linear.generate_prompt_buckets(
        bs_bucket_config, seq_bucket_config, max_num_batched_tokens)
    assert len(buckets) == 5
    assert len(omitted_buckets) == 7
    assert all(bs * seq <= max_num_batched_tokens for bs, seq in buckets)


def test_generate_decode_buckets():
    bs_bucket_config = [1, 32, 128]
    blocks_bucket_config = [128, 128, 2048]
    max_blocks = 1024
    buckets = linear.generate_decode_buckets(bs_bucket_config,
                                             blocks_bucket_config, max_blocks)
    assert len(buckets) == 72
    assert all(blocks <= max_blocks for _, blocks in buckets)


def test_linear_next_pow2():
    assert linear.next_pow2(15, 1) == 16
    assert linear.next_pow2(16, 1) == 16
    assert linear.next_pow2(17, 1) == 32
    assert linear.next_pow2(129, 1) == 256


def test_linear_round_up():
    assert linear.round_up(5, 4) == 8
    assert linear.round_up(12, 5) == 15


def test_linear_find_bucket():
    config = (1, 32, 128)
    assert linear.find_bucket(5, config) == 8
    assert linear.find_bucket(9, config) == 16
    assert linear.find_bucket(64, config) == 64
    assert linear.find_bucket(65, config) == 96


@pytest.mark.parametrize(
    "hpu_bucketing_context_fixture",
    ["hpu_linear_bucketing_context", "hpu_exponential_bucketing_context"])
def test_generate_prompt_buckets_method(hpu_bucketing_context_fixture,
                                        request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    hpu_bucketing_context.generate_prompt_buckets()
    assert len(hpu_bucketing_context.prompt_buckets) > 0


@pytest.mark.parametrize(
    "hpu_bucketing_context_fixture",
    ["hpu_linear_bucketing_context", "hpu_exponential_bucketing_context"])
def test_generate_decode_buckets_method(hpu_bucketing_context_fixture,
                                        request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    hpu_bucketing_context.generate_decode_buckets(256)
    assert len(hpu_bucketing_context.decode_buckets) > 0


@pytest.mark.parametrize("hpu_bucketing_context_fixture, ref_max_shape",
                         [("hpu_linear_bucketing_context", (16, 1024)),
                          ("hpu_exponential_bucketing_context", (16, 4096))])
def test_get_max_prompt_shape(hpu_bucketing_context_fixture, ref_max_shape,
                              request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    max_shape = hpu_bucketing_context.get_max_prompt_shape()
    assert max_shape == ref_max_shape


@pytest.mark.dependency(depends=["test_generate_prompt_buckets_method"])
@pytest.mark.parametrize("hpu_bucketing_context_fixture, ref_padded_size",
                         [("hpu_linear_bucketing_context", 8),
                          ("hpu_exponential_bucketing_context", 8)])
def test_get_padded_prompt_batch_size(hpu_bucketing_context_fixture,
                                      ref_padded_size, request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    padded_size = hpu_bucketing_context.get_padded_prompt_batch_size(5)
    assert padded_size == ref_padded_size


@pytest.mark.dependency(depends=["test_generate_decode_buckets_method"])
@pytest.mark.parametrize("hpu_bucketing_context_fixture, ref_padded_size",
                         [("hpu_linear_bucketing_context", 8),
                          ("hpu_exponential_bucketing_context", 8)])
def test_get_padded_decode_batch_size(hpu_bucketing_context_fixture,
                                      ref_padded_size, request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    padded_size = hpu_bucketing_context.get_padded_decode_batch_size(5)
    assert padded_size == ref_padded_size


@pytest.mark.dependency(depends=["test_generate_prompt_buckets_method"])
@pytest.mark.parametrize("hpu_bucketing_context_fixture, ref_padded_len",
                         [("hpu_linear_bucketing_context", 128),
                          ("hpu_exponential_bucketing_context", 128)])
def test_get_padded_prompt_seq_len(hpu_bucketing_context_fixture,
                                   ref_padded_len, request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    padded_len = hpu_bucketing_context.get_padded_prompt_seq_len(100)
    assert padded_len == ref_padded_len


@pytest.mark.dependency(depends=["test_generate_decode_buckets_method"])
@pytest.mark.parametrize("hpu_bucketing_context_fixture, ref_padded_blocks",
                         [("hpu_linear_bucketing_context", 128),
                          ("hpu_exponential_bucketing_context", 128)])
def test_get_padded_decode_num_blocks(hpu_bucketing_context_fixture,
                                      ref_padded_blocks, request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    padded_blocks = hpu_bucketing_context.get_padded_decode_num_blocks(100)
    assert padded_blocks == ref_padded_blocks


@pytest.mark.parametrize("hpu_bucketing_context_fixture, ref_padded_size",
                         [("hpu_linear_bucketing_context", 8),
                          ("hpu_exponential_bucketing_context", 8)])
def test_get_padded_batch_size(hpu_bucketing_context_fixture, ref_padded_size,
                               request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    padded_size = hpu_bucketing_context.get_padded_batch_size(5,
                                                              is_prompt=True)
    assert padded_size == ref_padded_size


@pytest.mark.parametrize("hpu_bucketing_context_fixture, ref_padded_value",
                         [("hpu_linear_bucketing_context", 128),
                          ("hpu_exponential_bucketing_context", 128)])
def test_get_padded_seq_or_block(hpu_bucketing_context_fixture,
                                 ref_padded_value, request):
    hpu_bucketing_context = request.getfixturevalue(
        hpu_bucketing_context_fixture)
    padded_value = hpu_bucketing_context.get_padded_seq_or_block(
        100, is_prompt=True)
    assert padded_value == ref_padded_value
