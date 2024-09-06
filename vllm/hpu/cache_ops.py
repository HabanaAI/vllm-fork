###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import habana_frameworks.torch as htorch
import torch


def reshape_and_cache(key,
                      value,
                      key_cache,
                      value_cache,
                      slot_mapping,
                      dtype,
                      is_prompt=False):
    num_blocks = key_cache.size(0)
    block_size = key_cache.size(1)
    slot_mapping = slot_mapping.flatten()
    indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offsets = torch.fmod(slot_mapping, block_size)
    num_slots_requested = slot_mapping.size(0)
    num_slots_available = num_blocks * block_size
    # NOTE(kzawora): HPU PT bridge crashes with
    # RuntimeError: Invalid inputs for scatter_nd_onnx
    # on index_put when num_slots_requested > num_slots_available.
    # This case might occur when we have little kv cache blocks and
    # lots of padding, or are doing warmup.
    # This loop is a workaround for this issue. Please remove it
    # once key_cache.index_put_(indices, offsets), key) works.
    num_kv_cache_passes = torch.div(num_slots_requested,
                                    num_slots_available).ceil().int().item()
    for i in range(num_kv_cache_passes):
        start_idx = i * num_slots_available
        end_idx = (i + 1) * num_slots_available
        key_cache.index_put_(
            (indices[start_idx:end_idx], offsets[start_idx:end_idx]),
            key[start_idx:end_idx])
        value_cache.index_put_(
            (indices[start_idx:end_idx], offsets[start_idx:end_idx]),
            value[start_idx:end_idx])


def prepare_to_cache(cache, slot_mapping):
    num_blocks = cache.size(0)
    block_size = cache.size(1)
    slot_mapping = slot_mapping.flatten()
    indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offsets = torch.fmod(slot_mapping, block_size)
    num_slots_requested = slot_mapping.size(0)
    num_slots_available = num_blocks * block_size
    # NOTE(kzawora): HPU PT bridge crashes with
    # RuntimeError: Invalid inputs for scatter_nd_onnx
    # on index_put when num_slots_requested > num_slots_available.
    # This case might occur when we have little kv cache blocks and
    # lots of padding, or are doing warmup.
    # This loop is a workaround for this issue. Please remove it
    # once key_cache.index_put_(indices, offsets), key) works.
    num_kv_cache_passes = torch.div(num_slots_requested,
                                    num_slots_available).ceil().int().item()

    return num_kv_cache_passes, num_slots_available, indices, offsets


def insert_or_update_cache(input, cache, num_kv_cache_passes,
                           num_slots_available, block_indices, block_offsets):
    for i in range(num_kv_cache_passes):
        start_idx = i * num_slots_available
        end_idx = (i + 1) * num_slots_available
        cache.index_put_((block_indices[start_idx:end_idx],
                          block_offsets[start_idx:end_idx]),
                         input[start_idx:end_idx])


def swap_blocks(src, dst, block_mapping):
    if block_mapping.numel() == 0:
        return

    block_mapping = block_mapping.transpose(0, 1)
    src_indices = block_mapping[0]
    dst_indices = block_mapping[1]

    dst.index_put_(dst_indices, src.index_select(0, src_indices))

    htorch.core.mark_step()
    torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, block_mapping):
    if block_mapping.numel() == 0:
        return

    block_mapping = block_mapping.transpose(0, 1)
    src = block_mapping[0]
    dst = block_mapping[1]

    for key_cache, value_cache in zip(key_caches, value_caches):
        key_cache.index_copy_(0, dst, key_cache.index_select(0, src))
        value_cache.index_copy_(0, dst, value_cache.index_select(0, src))

    htorch.core.mark_step()
