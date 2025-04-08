# SPDX-License-Identifier: Apache-2.0
"""Utils for model executor."""
from typing import Any, Dict, Optional

import torch

import os

use_mmap = os.getenv("USE_MMAP", "0") == "1"


def set_random_seed(seed: int) -> None:
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)


def set_expert_data(param, expert_id, param_data, idx=None, cat_dim=0):
    """function to avoid copy for tensors to concatenate, used by use_mmap."""
    def to(self, *args, **kwargs):
        """function to overwrite Parameter.to() method, used by use_mmap."""
        sorted_items = sorted(self.cached_tensors.items())
        sorted_tensors = [tensor for idx, tensor in sorted_items]
        if isinstance(sorted_tensors[0], dict):
            sorted_tensor_dicts = sorted_tensors
            for i, tensor_dict in enumerate(sorted_tensor_dicts):
                sorted_items = sorted(tensor_dict.items())
                sorted_tensors_tmp = [tensor for idx, tensor in sorted_items]
                concatenated = torch.cat(sorted_tensors_tmp, dim=self.cat_dim)
                sorted_tensors[i] = concatenated.unsqueeze(0)
        else:
            sorted_tensors = [tensor for tensor in sorted_tensors]
        concatenated = torch.cat(sorted_tensors, dim=0)  # dim=0 for expert id
        self.cached_tensors = {}
        return concatenated.to(*args, **kwargs)

    if param.data.shape != param_data.shape:
        from types import MethodType
        setattr(param, "cat_dim", cat_dim)
        setattr(param, "to", MethodType(to, param))
        if not hasattr(param, "cached_tensors"):
            param.cached_tensors = {}
        if idx is None:
            param.cached_tensors[expert_id] = param_data
        else:
            if expert_id not in param.cached_tensors:
                param.cached_tensors[expert_id] = {}
            param.cached_tensors[expert_id][idx] = param_data
    else:
        param.data = param_data


def set_param_data(param, idx, param_data, cat_dim=0):
    """function to avoid copy for tensors to concatenate, used by use_mmap."""
    def to(self, *args, **kwargs):
        """function to overwrite Parameter.to() method, used by use_mmap."""
        if self.cached_tensors == {}:  # cat already
            return self.data.to(*args, **kwargs)
        sorted_items = sorted(self.cached_tensors.items())
        sorted_tensors = [tensor for idx, tensor in sorted_items]
        concatenated = torch.cat(sorted_tensors, dim=self.cat_dim)
        self.cached_tensors = {}
        return concatenated.to(*args, **kwargs)

    if param.data.shape != param_data.shape:
        from types import MethodType
        setattr(param, "cat_dim", cat_dim)
        setattr(param, "to", MethodType(to, param))
        if not hasattr(param, "cached_tensors"):
            param.cached_tensors = {}
        param.cached_tensors[idx] = param_data
    else:
        param.data = param_data


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        from vllm.platforms import current_platform
        if current_platform.is_tpu() and key == "weight_loader":
            value = _make_synced_weight_loader(value)
        setattr(weight, key, value)


def _make_synced_weight_loader(original_weight_loader):

    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        torch._sync(param)

    return _synced_weight_loader
