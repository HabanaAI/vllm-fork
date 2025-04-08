# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import gguf
import time
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.utils import set_weight_attrs


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, ) -> None:
        pass

    def __repr__(self) -> str:
        return ("GGUFConfig()")

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return GGUFEmbeddingMethod(self)
        return None


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def _fuse_mul_mat(x: torch.Tensor, qweight: torch.Tensor,
                  qweight_type: int) -> torch.Tensor:
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    # enable MMVQ in contiguous batching with batch_size=1
    if x.shape[0] == 1 and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape)
        y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(
            f"Unsupported GGUF quantization type: {qweight_type}")
    return y


class Dequantizer:

    def custom_view(self, x):
        low = x[:,0].to(torch.int32).unsqueeze(1)
        high = x[:,1].to(torch.int32).unsqueeze(1)

        bits = (high << 8) | low

        sign = (bits >> 15) & 0x1
        exponent = (bits >> 10) & 0x1F
        fraction = bits & 0x3FF

        result = torch.zeros_like(bits, dtype=torch.float32, device=torch.device('hpu'))

        #Normal numbers
        normals = (exponent > 0) & (exponent < 0x1F)
        n_values = (1 + fraction / 1024) * torch.pow(2.0,exponent - 15)
        result = torch.where(normals, n_values, result)
        
        #subnormal
        subnormals = (exponent == 0) & (fraction != 0)
        s_values = (fraction / 1024) * (2.0**(-14))
        result = torch.where(subnormals, s_values, result)

        # inf and NaN
        inf_mask = (exponent == 0x1F) & (fraction == 0)
        result = torch.where(inf_mask, torch.tensor(float('inf'),device=torch.device('hpu')), result)
        nan_mask = (exponent == 0x1F) & (fraction != 0)
        result = torch.where(nan_mask, torch.tensor(float('nan'),device=torch.device('hpu')), result)

        # Apply sign
        result = torch.where(sign==1, result * -1, result)
        # print("HPU result ==> ",result, result.device)
        return result


    def get_scale_min(self, scales):

        n_blocks = scales.shape[0]
        scales = scales.to(torch.uint8).view(n_blocks, 3, 4)

        d, m, m_d = torch.tensor_split(scales, 3, dim=-2)

        #compute scale and min_vales using bitwise operations
        sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
        min_vals = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

        return sc.reshape(n_blocks, 8), min_vals.reshape(n_blocks, 8)

    
    def dequantize_blocks_q4_k(self, blocks):
    
        n_blocks = blocks.shape[0]
        d, dmin, scales, qs = torch.split(blocks, [2, 2, 12, 128], dim=1)

        sc, m = self.get_scale_min(scales)
        # d, dmin = d.clone().view(torch.float16).float(), dmin.clone().view(torch.float16).float()
        d = self.custom_view(d.clone())
        dmin = self.custom_view(dmin.clone())

        #compute dequantized values
        d = (d * sc.float()).reshape(n_blocks, -1, 1)
        dm = (dmin * m.float()).reshape(n_blocks, -1, 1)

        #Bitwise operations for quantized values
        qs = qs.reshape(n_blocks, -1, 1, 32)
        qs = qs >> torch.tensor([0, 4], dtype=torch.uint8, device=torch.device('hpu')).reshape(1,1,2,1)
        qs = (qs & 0x0F).reshape(n_blocks, -1, 32).float()

        return (d * qs - dm).reshape(n_blocks, 256)

    
    def dequantize_blocks_q6_k(self, blocks):

        n_blocks = blocks.shape[0]
        ql, qh, scales, d = torch.split(blocks, [128, 64, 16, 2], dim=1)
    
        # scales, d = scales.clone().view(torch.int8).float(), d.clone().view(torch.float16).float()
        scales = scales.clone().view(torch.int8).float()
        d = self.custom_view(d.clone())
        d = (d * scales).reshape(n_blocks, 16, 1)

        #compute dequantized values
        ql = ql.reshape(n_blocks, -1, 1, 64)
        ql = ql >> torch.tensor([0, 4], dtype=torch.uint8, device=torch.device('hpu')).reshape(1, 1, 2, 1)
        ql = (ql & 0x0F).reshape(n_blocks, -1, 32)

        qh = qh.reshape(n_blocks, -1, 1, 32)
        qh = qh >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=torch.device('hpu')).reshape(1, 1, 4, 1)
        qh = (qh & 0x03).reshape(n_blocks, -1, 32)

        q = (ql | (qh << 4)).to(torch.int8) - 32
        q = q.reshape(n_blocks, 16, -1).to(torch.float32)

        return (d * q).reshape(n_blocks, 256)


    def dequantize_rows(self, rows, qw_type): #type_size, block_size

        block_size, type_size = gguf.GGML_QUANT_SIZES[qw_type]

        rows = rows.to(torch.uint8)
        shape = rows.shape

        n_blocks = rows.numel() // type_size
        blocks = rows.reshape(n_blocks, type_size)
        if qw_type==12:
            blocks = self.dequantize_blocks_q4_k(blocks)
        elif qw_type == 14:
            blocks = self.dequantize_blocks_q6_k(blocks)

        assert blocks.dtype == torch.float32
        assert blocks.shape[-1] == block_size
        res = blocks.reshape(*shape[:-1], shape[-1] // type_size * block_size)
        # print("5555555555  ", res)
        return res

    def dequantize_grouped_rows(self, qw, qw_type, otype):

        block_size, type_size = gguf.GGML_QUANT_SIZES[qw_type]
        shape = qw.shape
        rows = qw.reshape(-1, qw.shape[-1])
        n_groups = max(rows.shape[0] // 16, 1)  # Ensure at least one group

        out = torch.cat([self.dequantize_rows(group, qw_type).flatten() for group in torch.tensor_split(rows, n_groups)],
                        dim=0).to(otype)

        return out.reshape(*shape[:-1], shape[-1] // type_size * block_size)


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config
        self.dequantizer = Dequantizer()

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight, {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            })
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(torch.empty(len(output_partition_sizes),
                                             dtype=torch.uint8),
                                 requires_grad=False)
        set_weight_attrs(
            qweight_type, {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True
            })
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)


    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        shard_id = getattr(layer.qweight, "shard_id", None)

        if shard_id:
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight.unbind(0)

            result = []
            for idx in shard_id:
                q_idx = layer.qweight.shard_id_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]

                if qweight_type in KQUANT_TYPES:
                    dequant_weight = self.dequantizer.dequantize_grouped_rows(qweight[q_idx], qweight_type, otype=torch.float32)
                    res = x @ dequant_weight.T
                    result.append(res)
                else:
                    result.append(_fuse_mul_mat(x, qweight[q_idx], qweight_type))
            out = torch.cat(result, axis=len(result[0].shape)-1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            # print("333333  ", layer, qweight.shape, qweight_type)

            if qweight_type in KQUANT_TYPES:
                dequant_weight = self.dequantizer.dequantize_grouped_rows(qweight.data, qweight_type, otype=torch.float32)  # Doubt 
                out = x @ dequant_weight.T
            else:
                out = _fuse_mul_mat(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """
    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.dequantizer = Dequantizer()


    def embedding(self, layer: torch.nn.Module,
                  x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight              #Doubt
        qweight_type = layer.qweight_type.weight_type

        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type] #256,144 - Q4 and 256, 210 - Q6
        hidden_size = qweight.shape[1] // type_size * block_size

        if qweight_type < 2:
            return torch.embedding(qweight, x)
        dequant_qweight = self.dequantizer.dequantize_grouped_rows(qweight.data, qweight_type, otype=torch.float32)
        res = torch.embedding(dequant_qweight, x)
        return res
        # x_flat = x.flatten()
        # quant = torch.index_select(qweight, dim=0, index=x_flat)
        # dequant = ops.ggml_dequantize(quant, qweight_type, hidden_size,
        #                               x_flat.shape[0])
        #return dequant.view(*x.shape, hidden_size)


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: List[torch.Tensor]

    def materialize_nested(self) -> Parameter:
        dtype = {data.dtype for data in self.data_container}
        assert len(dtype) == 1, ValueError(
            f"Data container has mixed dtypes: {dtype}")
        dtype = next(iter(dtype))
        nested_data = torch.nested.nested_tensor(self.data_container,
                                                 device=self.device,
                                                 dtype=dtype)
        
        # for i in range(len(self.data_container)):
        #     print(self.data_container[i].shape)

        # max_rows = max(tensor.size(0) for tensor in self.data_container)
        # max_cols = max(tensor.size(1) for tensor in self.data_container)

        # padded_data = [torch.cat([tensor.to(device=self.device),torch.zeros(max_rows-tensor.size(0), tensor.size(1), device=self.device, dtype=dtype)],dim=0) for tensor in self.data_container]

        # padded_data = [torch.cat([tensor.to(device=self.device), torch.zeros(tensor.size(0), max_cols-tensor.size(1), device=self.device, dtype=dtype)],dim=1) for tensor in padded_data]

        # nested_data = torch.stack(padded_data)
        # print("Nested_data shape  = ", nested_data.shape)
        # print("--------------------------------------------")

        self.data_container.clear()
        param = torch.Tensor._make_subclass(self.cls_to_become,
                                            nested_data,
                                            require_grad=False)
        for k, v in self.__dict__.items():
            setattr(param, k, v)
        return param
