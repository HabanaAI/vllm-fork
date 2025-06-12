# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm_hpu_extension.config import (All, Disabled, Engine, Env,
                                       FirstEnabled, Hardware, Kernel,
                                       ModelType, Not, Value, VersionRange,
                                       boolean, choice, split_values_and_flags,
                                       to_dict)
from vllm_hpu_extension.kernels import block_softmax_adjustment, fsdpa


def get_user_flags():
    flags = [
        Env('VLLM_USE_V1', boolean),
        Env('VLLM_ENABLE_EXPERIMENTAL_FLAGS', boolean),
        Env('VLLM_EXPONENTIAL_BUCKETING', boolean),
        Env('VLLM_PROMPT_BS_BUCKET_MIN', int),
        Env('VLLM_PROMPT_BS_BUCKET_STEP', int),
        Env('VLLM_PROMPT_BS_BUCKET_MAX', int),
        Env('VLLM_PROMPT_BS_BUCKET_LIMIT', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_MIN', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_STEP', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_MAX', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_LIMIT', int),
        Env('VLLM_DECODE_BS_BUCKET_MIN', int),
        Env('VLLM_DECODE_BS_BUCKET_STEP', int),
        Env('VLLM_DECODE_BS_BUCKET_MAX', int),
        Env('VLLM_DECODE_BS_BUCKET_LIMIT', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_MIN', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_STEP', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_MAX', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_LIMIT', int),

        # Non-vllm flags that are also important to print
        Env('EXPERIMENTAL_WEIGHT_SHARING', str),
        Env('PT_HPU_WEIGHT_SHARING', str),
    ]
    return to_dict(flags)


def get_experimental_flags():
    flags = [
        Env('VLLM_PT_PROFILE', str),
        Env('VLLM_PROFILE_PROMPT', str),
        Env('VLLM_PROFILE_DECODE', str),
    ]
    return to_dict(flags)


def get_features():
    supported_attn_impls = ['flex_impl', 'fsdpa_impl', 'naive_impl']
    features = [
        Value('fp32_alibi_biases',
              True,
              env_var='VLLM_ALIBI_USE_FLOAT32_BIASES'),
        Value('fp32_softmax', ModelType('qwen2')),
        Value(
            'fused_block_softmax_adjustment',
            All(VersionRange(">=1.22.0.101"), Hardware('gaudi3'),
                Kernel(block_softmax_adjustment), Not(ModelType('qwen2')))),
        Value('flex_impl', False, env_var='VLLM_PROMPT_USE_FLEX_ATTENTION'),
        Value('fsdpa_impl',
              All(Kernel(fsdpa), Not(ModelType('mllama'))),
              env_var='VLLM_PROMPT_USE_FUSEDSDPA'),
        Value('naive_impl', True),
        Value('prompt_attn_impl',
              FirstEnabled(*supported_attn_impls),
              env_var_type=choice(*supported_attn_impls)),
        Value('skip_warmup', False),
        Value('use_contiguous_pa',
              Disabled('prefix_caching'),
              env_var='VLLM_CONTIGUOUS_PA'),
        Value('use_delayed_sampling',
              Engine('v0'),
              env_var='VLLM_DELAYED_SAMPLING'),
        Value('use_bucketing', True, env_var='VLLM_ENABLE_BUCKETING'),
    ]
    return split_values_and_flags(features)
