import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import Dict, Optional, Union, Any

from vllm.platforms import current_platform

if current_platform.is_hpu():
    import habana_frameworks.torch as htorch  # noqa: F401


class HpuCommunicator:

    def __init__(self, group: ProcessGroup, rank_in_group: int):
        if not current_platform.is_hpu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank_in_group = rank_in_group

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # FIXME(kzawora): this is a workaround for a bug in Habana PT bridge
        # occurring when PT_HPU_ENABLE_LAZY_COLLECTIVES=true env var is used
        # (which is required for tensor parallel HPUGraph inference)
        htorch.core.mark_step()
        dist.all_reduce(x, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += x.dim()
        input_size = x.size()
        # Allocate output tensor.
        output_tensor = torch.empty((world_size, ) + input_size,
                                    dtype=x.dtype,
                                    device=x.device)
        # All-gather.
        htorch.core.mark_step()
        dist.all_gather_into_tensor(output_tensor, x, group=self.group)
        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        
        if not dist.is_initialized() or self.world_size == 1:
            return tensor_dict

        all_gather_size = (1 if all_gather_group is None else
                           all_gather_group.world_size)
        all_gather_rank = (0 if all_gather_group is None else
                           all_gather_group.rank_in_group)
        
        group = self.group

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `send_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.send_object(metadata_list, dst=dst)
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue

            # send-allgather: send only a slice, then do allgather.
            if (all_gather_group is not None
                    and tensor.numel() % all_gather_size == 0):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            # use group for GPU tensors
            htorch.core.mark_step()
            torch.distributed.send(tensor,
                                   dst=self.ranks[dst],
                                   group=group)
        return None

    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:

        htorch.core.mark_step()
        dist.recv_tensor_dict(src, all_gather_group)
