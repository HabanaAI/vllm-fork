"""Utils for model executor."""
from typing import Any, Dict, Optional

import torch
import os

use_mmap = os.getenv("USE_MMAP", "0") == "1"


def set_random_seed(seed: int) -> None:
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)


def to(self, *args, **kwargs):
    """function to overwrite Parameter.to() method, used by use_mmap."""
    concatenated = torch.cat(self.cached_tensors, dim=0).to(*args, **kwargs)
    self.data.copy_(concatenated).to(*args, **kwargs)
    return self.data


def set_param_data(param, param_data):
    """function to avoid copy for tensors to concatenate, used by use_mmap."""
    if param.data.shape != param_data.shape:
        from types import MethodType
        setattr(param, "to", MethodType(to, param))
        if not hasattr(param, "cached_tensors"):
            setattr(param, "cached_tensors", [param_data])
        else:
            param.cached_tensors.append(param_data)
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
