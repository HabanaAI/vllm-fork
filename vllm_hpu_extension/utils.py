# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import os
from functools import lru_cache, wraps

import habana_frameworks.torch as htorch
import torch

from vllm_hpu_extension.runtime import get_config


@lru_cache(maxsize=None)
def is_fake_hpu() -> bool:
    return os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0'


def with_mark_steps(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        htorch.core.mark_step()
        result = fn(*args, **kwargs)
        del args
        del kwargs
        htorch.core.mark_step()
        return result

    return wrapped


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


class VLLMKVCache(torch.nn.Module):

    def __init__(self):
        super(VLLMKVCache, self).__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa

    def forward(self, input, cache, slot_mapping):
        # In cross-attention kv cache forward inputs are None in decode
        # We don't want to store them in the cache in such case
        if input is not None:
            cache.index_copy_(0, slot_mapping, input)
        return cache

    def fetch_from_cache(self, cache, blocks):
        if self.use_contiguous_pa:
            return cache[:blocks.size(0)]
        else:
            return cache.index_select(0, blocks)


class VLLMFP8KVCache(VLLMKVCache):

    def __init__(self, input_scale=1.0):
        super(VLLMKVCache, self).__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        self.input_scale = input_scale
        self.output_scale = 1.0 / self.input_scale

    def quant_input(self, input):
        return torch.ops.hpu.cast_to_fp8_v2(input, self.input_scale, False, False, torch.float8_e4m3fn)[0]
    
    def dequant_output(self, output):
        return torch.ops.hpu.cast_from_fp8(output, self.output_scale, torch.bfloat16)

    def forward(self, input, *args, **kwargs):
        qinput = self.quant_input(input)
        return super().forward(qinput, *args, **kwargs)

    def fetch_from_cache(self, quant_cache, blocks, permutations=None):
        if permutations:
            output_cache = super().fetch_from_cache(quant_cache, blocks,
                                                        permutations)
            for i in range(len(output_cache)):
                output_cache[i] = self.dequant_output(output_cache[i])
            return output_cache
        output_cache = super().fetch_from_cache(quant_cache, blocks)
        return self.dequant_output(output_cache)


class FP8Matmul(torch.nn.Module):

    def __init__(self, scale_input=1.0, scale_other=1.0,):
        super().__init__()
        self.scale_input = scale_input
        self.scale_other = scale_other

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(
            x, scale, False, False, torch.float8_e4m3fn
        )[0]

    def matmul_fp8(
        self, x, other, out_dtype, scale_input_inv=None, scale_other_inv=None
    ):
        return torch.ops.hpu.fp8_gemm_v2(
            A=x,
            trans_A=False,
            B=other,
            trans_B=False,
            D=None,
            out_dtype=out_dtype,
            A_scale_inv=scale_input_inv,
            B_scale_inv=scale_other_inv,
            bias=None,
            accumulate=False,
        )

    def forward(self, input, other):
        qinput = self.quant_input(input, self.scale_input)
        qother = self.quant_input(other, self.scale_other)
        output = self.matmul_fp8(
            qinput,
            qother,
            out_dtype=torch.bfloat16,
            scale_input_inv=1.0 / self.scale_input,
            scale_other_inv=1.0 / self.scale_other,
        )
        return output


class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'fusedSDPA kernel is None'
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
    ):
        return self._hpu_kernel_fsdpa.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            softmax_mode,
            recompute_mode,
            valid_sequence_lengths,
            padding_side,
        )


