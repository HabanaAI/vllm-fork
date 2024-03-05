###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from typing import Tuple
import torch
import habana_frameworks.torch as htorch


def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, is_prompt=False):
    """
    key: [num_tokens, num_heads, head_size]
    value: [num_tokens, num_heads, head_size]
    key_cache: [num_heads, head_size, block_size] * num_blocks
    value_cache: [num_heads, head_size, block_size] * num_blocks
    slot_mapping: [num_tokens]
    """
    block_size = key_cache.size(-1)
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = torch.fmod(slot_mapping, block_size)
    MAX_SLOTS_IN_GRAPH = 16
    slots_in_graph = 0

    for idx, offset, k, v in zip(block_indices, block_offsets, key, value):
        tmp_key = key_cache.index_select(0, idx)
        tmp_key.index_copy_(-1, offset, k.view(1, *k.shape, 1))
        key_cache.index_copy_(0, idx, tmp_key)

        tmp_value = value_cache.index_select(0, idx)
        tmp_value.index_copy_(-1, offset, v.view(1, *v.shape, 1))
        value_cache.index_copy_(0, idx, tmp_value)

        slots_in_graph += 1
        if slots_in_graph % MAX_SLOTS_IN_GRAPH == 0:
            htorch.core.mark_step()

    htorch.core.mark_step()


def swap_blocks(src, dst, block_mapping):
    index_src = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    index_dst = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    for src_idx, dst_idx in block_mapping.items():
        index_src[0] = src_idx
        index_dst[0] = dst_idx
        dst.index_put_([index_dst], src.index_select(0, index_src))
    if dst.device.type == 'hpu':
        htorch.core.mark_step()
        torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, block_mapping):
    index_src = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    index_dst = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    for src, dsts in block_mapping.items():
        index_src[0] = src
        for dst in dsts:
            index_dst[0] = dst
            for key_cache in key_caches:
                key_cache.index_copy_(0, index_dst, key_cache.index_select(0, index_src))
            for value_cache in value_caches:
                value_cache.index_copy_(0, index_dst, value_cache.index_select(0, index_src))
        if key_caches[0].device.type == 'hpu':
            htorch.core.mark_step()
