# ==-------------------------------------------------------------------------==
# VLLM-HPU-EXT PATCH Start
# ==-------------------------------------------------------------------------==
import logging
import os
import torch
from typing import Callable, Optional, Tuple
import habana_frameworks.torch as htorch

logging.basicConfig(level=logging.INFO)


class MoeFP8Matmul(torch.nn.Module):
    def __init__(
        self,
        block_size: Tuple[int, int] = (128, 128),
        high_precision=torch.bfloat16,
    ):
        super().__init__()
        self.block_size = block_size
        self.high_precision = high_precision
        self.is_dequantized = False

    def set_weight(self, w: torch.Tensor):
        self.weight = w

    def set_scale_inv_fp8(self, scale_inv_fp8: torch.Tensor):
        self.scale_inv_fp8 = scale_inv_fp8

    def set_high_precision(self, high_precision=torch.bfloat16):
        self.high_precision = high_precision

    def set_weight_block_size(self, block_size: Tuple[int, int] = (128, 128)):
        self.block_size = block_size

    def get_dequant_weight(self):
        # FIXME: (Yi) move `dequant_block_fp8_weight_naive` to extension as well.
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            dequant_block_fp8_weight_naive,
        )

        return dequant_block_fp8_weight_naive(
            self.weight,
            self.scale_inv_fp8,
            block_size=self.block_size,
            dtype=self.high_precision,
        )

    def forward(self, state, expert_id, w):
        raise NotImplementedError()

    def dequant_block_fp8_weight(self, layer: "MoeFP8Matmul") -> torch.Tensor:
        # This function is called by INC during either the measurement or quantization phase.
        # - In the quantization phase, INC requantizes the BF16 weight to FP8 and updates the weight.
        # - In the measurement phase, INC only measures the BF16 weight without updating it.
        # Tracking the BF16 weight can lead to Out of Memory (OoM) issues, so we avoid storing it.
        # If the weight has already been updated, we return it directly.
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight

        dequant_weight = layer.get_dequant_weight()
        layer.is_dequantized = True
        return dequant_weight

    def get_dequant_weights_func(
        self,
    ) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
        return self.dequant_block_fp8_weight


class VllmMixtureOfExpertsOpFP8(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        global_num_experts: int = 0,
        experts_min: int = 0,
        experts_max: int = 8,
    ):
        super().__init__()
        self.w13_list = torch.nn.ModuleList(
            [MoeFP8Matmul() for _ in range(num_experts)]
        )
        self.w2_list = torch.nn.ModuleList(
            [MoeFP8Matmul() for _ in range(num_experts)]
        )
        self.enable_moe_chunk = (
            os.environ.get("VLLM_SUPPORT_MOE_CHUNK", "false").lower() == "true"
        )
        self.chunk_size_list = [
            int(x)
            for x in os.environ.get(
                "PT_HPU_MOE_CHUNK", "64,128,512,1024,1536,2048,4096"
            ).split(",")
            if x.strip()
        ]
        self.token_boundary_list = [
            int(x)
            for x in os.environ.get(
                "PT_HPU_MOE_TOKEN_BOUNDARY", "64,128,1536,1736,2048,3072,4096"
            ).split(",")
            if x.strip()
        ]
        assert len(self.chunk_size_list) == len(self.token_boundary_list), (
            f"chunk_size_list({len(self.chunk_size_list)}) and "
            f"token_boundary_list({len(self.token_boundary_list)}) must be the same length"
        )
        logger = logging.getLogger()
        logger.info("token_boundary_list is:%s",self.token_boundary_list)
        logger.info("chunk_size_list is:%s",self.chunk_size_list)

        self.num_experts = num_experts
        self.global_num_experts = global_num_experts
        self.experts_min = experts_min
        self.experts_max = experts_max

    def _get_extra_kwargs(self, tokens_num: int):
        if self.enable_moe_chunk:
            chunk_size = self.chunk_size_list[-1]
            for idx, threshold in enumerate(self.token_boundary_list):
                if tokens_num <= threshold:
                    chunk_size = self.chunk_size_list[idx]
                    break
            kwargs = {
                "chunk_size": chunk_size,
                "total_experts": self.global_num_experts,
            }
        else:
            kwargs = {}
        return kwargs

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
    ):
        min_expert = self.experts_min
        max_expert = self.experts_max
        w13_list_slice = []
        w2_list_slice = []
        tokens_num, _ = x.shape
        kwargs = self._get_extra_kwargs(tokens_num)
        for j in range(self.num_experts):
            w13_list_slice.append(self.w13_list[j].get_dequant_weight())
            w2_list_slice.append(self.w2_list[j].get_dequant_weight())

        final_hidden_states = torch.ops.hpu.mixture_of_experts(
            hidden_states=x,
            expert_routing_table=topk_ids.to(torch.int64),
            router_weights=topk_weights.to(x.dtype),
            w12=w13_list_slice,
            w3=w2_list_slice,
            permuted_weights=True,
            activation="silu",
            experts_min=min_expert,
            experts_max=max_expert,
            **kwargs,
        )
        htorch.core.mark_step()
        return final_hidden_states


# ==-------------------------------------------------------------------------==
# VLLM-HPU-EXT PATCH End
# ==-------------------------------------------------------------------------==
