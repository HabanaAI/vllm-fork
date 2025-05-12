from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

import habana_frameworks.torch as htorch
import habana_frameworks.torch.utils as htutils

from vllm.distributed.device_communicators.pyhccl_wrapper import (
    HCCLLibrary, buffer_type, hpuStream_t, hcclComm_t, hcclDataTypeEnum,
    hcclRedOpTypeEnum, hcclUniqueId)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class PyHcclCommunicator:

    def __init__(
        self,
        group: StatelessProcessGroup,
        device: Union[int, str, torch.device],
        library_path: Optional[str] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyHcclCommunicator to.
            library_path: the path to the HCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        self.rank = group.rank
        self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return
        try:
            self.hccl = HCCLLibrary(library_path)
        except Exception:
            # disable because of missing HCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        logger.info("vLLM is using hccl==%s", self.hccl.hcclGetVersion())

        if self.rank == 0:
            # get the unique id from HCCL
            self.unique_id = self.hccl.hcclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = hcclUniqueId()

        self.unique_id = group.broadcast_obj(self.unique_id, src=0)
        if isinstance(device, int):
            device = torch.device(f"hpu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device
        
        htorch.core.mark_step()
        self.comm: hcclComm_t = self.hccl.hcclCommInitRank(
            self.world_size, self.unique_id, self.rank)

        # A small all_reduce for warmup.
        data = torch.ones(1, device=device)
        out = self.all_reduce(data)
        del data
        torch.hpu.synchronize()

    def all_reduce(self,
                   in_tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None) -> torch.Tensor:
        if self.disabled:
            return None
        assert in_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}")

        out_tensor = torch.empty_like(in_tensor)

        htorch.core.mark_step()
        torch.hpu.synchronize()

        self.hccl.hcclAllReduce(buffer_type(htutils.experimental._data_ptr(in_tensor)),
                                buffer_type(htutils.experimental._data_ptr(out_tensor)),
                                in_tensor.numel(),
                                hcclDataTypeEnum.from_torch(in_tensor.dtype),
                                hcclRedOpTypeEnum.from_torch(op), self.comm,
                                hpuStream_t(htutils.experimental._compute_stream()))
        htorch.core.mark_step()
        torch.hpu.synchronize()
        return out_tensor

    def all_gather(self,
                   output_tensor: torch.Tensor,
                   input_tensor: torch.Tensor,
                   stream=None):
        if self.disabled:
            return
        assert input_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        htorch.core.mark_step()
        torch.hpu.synchronize()
        self.hccl.hcclAllGather(
            buffer_type(htutils.experimental._data_ptr(input_tensor)),
            buffer_type(htutils.experimental._data_ptr(output_tensor)), input_tensor.numel(),
            hcclDataTypeEnum.from_torch(input_tensor.dtype), self.comm,
            hpuStream_t(htutils.experimental._compute_stream()))
        htorch.core.mark_step()
        torch.hpu.synchronize()

    def reduce_scatter(self,
                       output_tensor: torch.Tensor,
                       input_tensor: torch.Tensor,
                       op: ReduceOp = ReduceOp.SUM,
                       stream=None):
        if self.disabled:
            return
        assert input_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        
        htorch.core.mark_step()
        torch.hpu.synchronize()
        self.hccl.hcclReduceScatter(
            buffer_type(htutils.experimental._data_ptr(input_tensor)),
            buffer_type(htutils.experimental._data_ptr(output_tensor)), output_tensor.numel(),
            hcclDataTypeEnum.from_torch(input_tensor.dtype),
            hcclRedOpTypeEnum.from_torch(op), self.comm,
            hpuStream_t(htutils.experimental._compute_stream()))
        htorch.core.mark_step()
        torch.hpu.synchronize()

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        
        htorch.core.mark_step()
        torch.hpu.synchronize()
        self.hccl.hcclSend(buffer_type(htutils.experimental._data_ptr(tensor)), tensor.numel(),
                           hcclDataTypeEnum.from_torch(tensor.dtype), dst,
                           self.comm, hpuStream_t(htutils.experimental._compute_stream()))
        htorch.core.mark_step()
        torch.hpu.synchronize()

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        
        htorch.core.mark_step()
        torch.hpu.synchronize()
        self.hccl.hcclRecv(buffer_type(htutils.experimental._data_ptr(tensor)), tensor.numel(),
                           hcclDataTypeEnum.from_torch(tensor.dtype), src,
                           self.comm, hpuStream_t(htutils.experimental._compute_stream()))
        htorch.core.mark_step()
        torch.hpu.synchronize()

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        sendbuff = buffer_type(htutils.experimental._data_ptr(tensor))
        recvbuff = buffer_type(htutils.experimental._data_ptr(tensor))

        htorch.core.mark_step()
        torch.hpu.synchronize()
        self.hccl.hcclBroadcast(sendbuff, recvbuff, tensor.numel(),
                                hcclDataTypeEnum.from_torch(tensor.dtype), src,
                                self.comm, hpuStream_t(htutils.experimental._compute_stream()))
        htorch.core.mark_step()
        torch.hpu.synchronize()
