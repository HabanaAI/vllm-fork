# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import importlib.util
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin,
    prepare_moe_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, all_close_1d, cutlass_block_fp8_supported,
    cutlass_fp8_supported, maybe_create_device_identity,
    normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if current_platform.is_hpu():
    import vllm_hpu_extension.ops as hpu_ops
    from vllm_hpu_extension.ops import scaled_fp8_quant
    ops.scaled_fp8_quant = scaled_fp8_quant

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


def _is_col_major(x: torch.Tensor) -> bool:
    assert x.dim() == 3
    b, m, n = x.shape
    return x.stride(0) == m * n and x.stride(1) == 1 and x.stride(2) == m


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[list[str]] = None,
        weight_block_size: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now.")
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions")
            if activation_scheme != "dynamic":
                raise ValueError("The block-wise quantization only supports "
                                 "dynamic activation scheme for now, but got "
                                 f"{activation_scheme} activation scheme.")
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"],
                                                 None)
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme,
                   ignored_layers=ignored_layers,
                   weight_block_size=weight_block_size)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> Optional[str]:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = False
        self.cutlass_block_fp8_supported = False
        if current_platform.is_cuda_alike():
            self.cutlass_fp8_supported = cutlass_fp8_supported()
            self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()

        self.use_marlin = False
        if not current_platform.is_hpu():
            # For GPUs that lack FP8 hardware support, we can leverage the
            # Marlin kernel for fast weight-only FP8 quantization
            self.use_marlin = (not current_platform.has_device_capability(89)
                               or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        # AITER is only supported on ROCm and only for FP8_FNUZ
        # and at the moment are MI300 series
        self.use_aiter_and_is_supported = (current_platform.is_rocm()
                                           and envs.VLLM_ROCM_USE_AITER
                                           and envs.VLLM_ROCM_USE_AITER_LINEAR
                                           and current_platform.is_fp8_fnuz())

        self.block_quant = self.quant_config.weight_block_size is not None
        self.fp8_linear = Fp8LinearOp(
            # Default to using per_token quantization if cutlass is supported
            use_per_token_if_dynamic=cutlass_fp8_supported())

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        maybe_create_device_identity()

        layer.quant_config = self.quant_config
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        if current_platform.is_hpu() and envs.VLLM_HPU_CONVERT_TO_FP8UZ:
            weight_loader = self._gaudi_weight_wrapper(weight_loader)

        if self.block_quant:
            tp_size = get_tensor_model_parallel_world_size()
            assert self.quant_config.weight_block_size is not None
            layer.weight_block_size = self.quant_config.weight_block_size
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if (tp_size > 1
                    and input_size // input_size_per_partition == tp_size
                    and input_size_per_partition % block_k != 0):
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")
            # Required by column parallel or enabling merged weights
            if (tp_size > 1 and output_size // output_size_per_partition
                    == tp_size) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}.")

        # WEIGHT
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)

        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if not self.block_quant:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes),
                                     dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                layer.register_parameter("weight_scale", scale)
            else:
                assert self.quant_config.activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                # The weight_scale_inv name is intentional for deepseekv3
                layer.register_parameter("weight_scale_inv", scale)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(data=torch.empty(
                    len(output_partition_sizes), dtype=torch.float32),
                                                weight_loader=weight_loader)

                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "input_scale"})
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def _gaudi_weight_wrapper(self, weight_loader):
        """Wrapper for Gaudi weight conversion."""

        def wrapper(*args, **kwargs):
            # args[0] is parameter, args[1] is loaded_weight
            # weights will be always in fp8, but scales will be in fp32,
            # so we can detect it by dtype
            loaded_weight = args[1]
            if loaded_weight.dtype == torch.float8_e4m3fn:
                loaded_weight = (loaded_weight.float() * 0.5).to(
                    torch.float8_e4m3fn)
            else:
                loaded_weight = (loaded_weight.data * 2.0)
            args = (args[0], loaded_weight) + args[2:]

            weight_loader(*args, **kwargs)

        return wrapper

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_FP8_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight

    def dequant_block_fp8_weight(self, layer) -> torch.Tensor:
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight
        dequant_weight = hpu_ops.dequant_block_fp8_weight_naive(
            layer.weight,
            layer.weight_scale_inv.data,
            self.quant_config.weight_block_size,
            original_M=layer.orig_M,
            original_N=layer.orig_N,
            do_unpad=True,
        )
        return dequant_weight

    def process_weights_after_loading(self, layer: Module) -> None:
        size_k_first = True
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            size_k_first = False
            if current_platform.is_hpu():
                layer = hpu_ops.fp8_block_linear_postprocess_weights(
                    layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
                return
            if current_platform.is_fp8_fnuz():
                weight, weight_scale_inv, _ = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=layer.weight,
                        weight_scale=layer.weight_scale_inv)
            else:
                weight = layer.weight.data
                weight_scale_inv = layer.weight_scale_inv.data

            weight = self._maybe_pad_weight(weight)

            # Torch.compile cannot use Parameter subclasses.
            layer.weight = Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = Parameter(weight_scale_inv,
                                               requires_grad=False)

        # If checkpoint not serialized fp8, quantize the weights.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

        # If checkpoint is fp8, handle that there are N scales for N
        # shards in a fused module
        else:
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                    requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = torch.nn.Parameter(layer.input_scale.data,
                                                       requires_grad=False)

            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            if not self.use_marlin:
                # Dequant -> Quant with max scale so we can run per tensor.
                if current_platform.is_fp8_fnuz():
                    weight, weight_scale, input_scale = \
                        normalize_e4m3fn_to_e4m3fnuz(
                            weight=weight,
                            weight_scale=weight_scale,
                            input_scale=layer.input_scale)
                    if input_scale is not None:
                        layer.input_scale = Parameter(input_scale,
                                                      requires_grad=False)

                weight_scale, weight = requantize_with_max_scale(
                    weight=weight,
                    weight_scale=weight_scale,
                    logical_widths=layer.logical_widths,
                )

            weight = self._maybe_pad_weight(weight)
            # Update layer with new values.
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)

        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer, size_k_first)
            # Activations not quantized for marlin.
            del layer.input_scale

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)

        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            if current_platform.is_hpu():
                return hpu_ops.apply_block_fp8_linear_hpu(
                    input=x,
                    layer=layer,
                    block_size=self.quant_config.weight_block_size,
                    bias=bias,
                    do_unpad=True,
                    force_channel_fp8=envs.VLLM_HPU_FORCE_CHANNEL_FP8,
                )
            return torch.ops.vllm.apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
                use_aiter_and_is_supported=self.use_aiter_and_is_supported,
            )

        if self.block_quant and current_platform.is_hpu():
            if layer.weight_scale.dim() > 1:
                weight_scale = layer.weight_scale.transpose(0, 1)
            else:
                weight_scale = layer.weight_scale
            return hpu_ops.apply_fp8_linear_hpu(input=x,
                                                weight=layer.weight,
                                                weight_scale=weight_scale,
                                                input_scale=layer.input_scale,
                                                bias=bias,
                                                trans_B=False)
        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        from vllm.model_executor.layers.fused_moe import fused_experts
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                           or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        # Check for DeepGemm support.
        self.allow_deep_gemm = False
        if envs.VLLM_USE_DEEP_GEMM:
            if not has_deep_gemm:
                logger.warning_once("Failed to import DeepGemm kernels.")
            elif not self.block_quant:
                logger.warning_once("Model is not block quantized. Not using "
                                    " DeepGemm kernels")
            elif (current_platform.is_cuda()
                  and current_platform.has_device_capability(90)):
                logger.info_once("Using DeepGemm kernels for Fp8MoEMethod.")
                self.allow_deep_gemm = True
            else:
                logger.warning_once(
                    "DeepGemm not supported on the current platform.")

        self.topk_indices_dtype = None
        self.fused_experts = functools.partial(  # type: ignore
            fused_experts,
            use_fp8_w8a8=True,
            block_shape=self.quant_config.weight_block_size,
            allow_deep_gemm=self.allow_deep_gemm)

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        layer.weight_block_size = None
        if current_platform.is_hpu() and envs.VLLM_HPU_CONVERT_TO_FP8UZ:
            extra_weight_attrs["weight_loader"] = self._gaudi_weight_wrapper(
                extra_weight_attrs.get("weight_loader"))
        layer.quant_config = self.quant_config
        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            layer.weight_block_size = self.quant_config.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}.")
            if (tp_size > 1
                    and intermediate_size_per_partition % block_k != 0):
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32),
                                                  requires_grad=False)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) //
                         block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.
             value} if self.block_quant else
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8.")

            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def _gaudi_weight_wrapper(self, weight_loader):
        """Wrapper for Gaudi weight conversion."""

        def wrapper(*args, **kwargs):
            # args[0] is parameter, args[1] is loaded_weight
            # weights will be always in fp8, but scales will be in fp32,
            # so we can detect it by dtype
            loaded_weight = args[1]
            if loaded_weight.dtype == torch.float8_e4m3fn:
                loaded_weight.data = (loaded_weight.data.float() * 0.5).to(
                    torch.float8_e4m3fn)
            else:
                loaded_weight.data = (loaded_weight.data * 2.0)
            args = (args[0], loaded_weight) + args[2:]
            weight_loader(*args, **kwargs)

        return wrapper

    def process_weights_after_loading(self, layer: Module) -> None:
        # Lazy import to avoid importing triton too early.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            is_rocm_aiter_moe_enabled, shuffle_weights)

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

        # TODO (rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            if current_platform.is_hpu():
                import vllm_hpu_extension.ops as hpu_ops
                layer = hpu_ops.fp8_block_moe_prepare_weights(
                    layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
                return
            if current_platform.is_fp8_fnuz():
                w13_weight, w13_weight_scale_inv, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale_inv,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale_inv, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale_inv,
                        layer.w2_input_scale)
            else:
                w13_weight = layer.w13_weight.data
                w13_weight_scale_inv = layer.w13_weight_scale_inv.data
                w2_weight = layer.w2_weight
                w2_weight_scale_inv = layer.w2_weight_scale_inv

            # torch.compile() cannot use Parameter subclasses.
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(w13_weight_scale_inv,
                                                   requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(w2_weight_scale_inv,
                                                  requires_grad=False)
            if self.rocm_aiter_moe_enabled:
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight.data, layer.w2_weight.data)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)

            # DeepGemm scales need to be transposed and aligned.  We try to do
            # it ahead of time for performance reasons.
            if self.allow_deep_gemm:
                # Lazy import to avoid CUDA initialization problems.
                import deep_gemm as dg
                if _is_col_major(layer.w13_weight_scale_inv):
                    layer.w13_weight_scale_inv = \
                        dg.get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
                if _is_col_major(layer.w2_weight_scale_inv):
                    layer.w2_weight_scale_inv = \
                        dg.get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()

        # If checkpoint is fp16, quantize in place.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data,
                                          dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
                layer.local_num_experts,
                dtype=torch.float32,
                device=w13_weight.device),
                                                        requires_grad=False)
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], layer.w2_weight_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w2_weight.data[expert, :, :])
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
            if self.rocm_aiter_moe_enabled:
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight, layer.w2_weight)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)
        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if (layer.w13_input_scale is None
                        or layer.w2_input_scale is None):
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None.")
                if (not all_close_1d(layer.w13_input_scale)
                        or not all_close_1d(layer.w2_input_scale)):
                    logger.warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer.")
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False)
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False)
            if current_platform.is_hpu():
                import vllm_hpu_extension.ops as hpu_ops
                layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)
                return
            if current_platform.is_fp8_fnuz():
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale,
                        layer.w2_input_scale)
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                      requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False)
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                     requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                           requires_grad=False)
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False)

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            if self.rocm_aiter_moe_enabled:
                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight, layer.w2_weight)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)

            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)

        if self.use_marlin:
            prepare_moe_fp8_layer_for_marlin(layer, False)
            # Activations not quantized for marlin.
            del layer.w13_input_scale
            del layer.w2_input_scale

    def select_gemm_impl(self, prepare_finalize):

        from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import (  # noqa: E501
            BatchedTritonOrDeepGemmExperts)
        from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
            TritonOrDeepGemmExperts)

        assert not self.use_marlin and not self.rocm_aiter_moe_enabled, (
            "Marlin and ROCm AITER are not supported with all2all yet.")

        experts: Optional[Union[BatchedTritonOrDeepGemmExperts,
                                TritonOrDeepGemmExperts]] = None
        max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
        use_batched_experts = max_num_tokens_per_rank is not None

        if use_batched_experts:
            experts = BatchedTritonOrDeepGemmExperts(
                max_num_tokens=max_num_tokens_per_rank,
                world_size=prepare_finalize.world_size,
                dp_size=prepare_finalize.dp_size,
                use_fp8_w8a8=True,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=self.quant_config.weight_block_size,
                allow_deep_gemm=self.allow_deep_gemm,
            )
        else:
            experts = TritonOrDeepGemmExperts(
                use_fp8_w8a8=True,
                block_shape=self.quant_config.weight_block_size,
                allow_deep_gemm=self.allow_deep_gemm,
            )

        assert experts is not None
        return experts

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        if current_platform.is_hpu():
            return self.forward_hpu(
                layer,
                x,
                use_grouped_topk,
                top_k,
                router_logits,
                renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                apply_router_weight_on_input=apply_router_weight_on_input,
                activation=activation)
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
        )

        if self.rocm_aiter_moe_enabled:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
                rocm_aiter_fused_experts)
            return rocm_aiter_fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                use_fp8_w8a8=True,
                apply_router_weight_on_input=apply_router_weight_on_input,
                w1_scale=(layer.w13_weight_scale_inv
                          if self.block_quant else layer.w13_weight_scale),
                w2_scale=(layer.w2_weight_scale_inv
                          if self.block_quant else layer.w2_weight_scale),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                block_shape=self.quant_config.weight_block_size)
        elif self.use_marlin:
            assert activation == "silu", (
                f"{activation} not supported for Marlin MoE.")
            assert not apply_router_weight_on_input, (
                "Apply router weight on input not supported for Marlin MoE.")
            return torch.ops.vllm.fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                quant_type_id=scalar_types.float8_e4m3fn.id,
                global_num_experts=global_num_experts,
                expert_map=expert_map)
        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            use_fp8_w8a8=True,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=(layer.w13_weight_scale_inv
                      if self.block_quant else layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale_inv
                      if self.block_quant else layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
            allow_deep_gemm=self.allow_deep_gemm,
        )

    def forward_hpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if use_grouped_topk or custom_routing_function is not None:
            topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)
        if layer.dp_size > 1:
            cu_tokens_across_dp_cpu = get_forward_context(
            ).dp_metadata.cu_tokens_across_dp_cpu

            topk_ids_across_dp = get_forward_context(
            ).dp_metadata.topk_ids_across_dp
            topk_ids = layer.multicast_fn(topk_ids, cu_tokens_across_dp_cpu,
                                          topk_ids_across_dp)

            topk_weights_across_dp = get_forward_context(
            ).dp_metadata.topk_weights_across_dp
            topk_weights = layer.multicast_fn(topk_weights,
                                              cu_tokens_across_dp_cpu,
                                              topk_weights_across_dp)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        output = layer.moe_op(
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation=activation,
        )
        return output.view(*input_shape)


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
