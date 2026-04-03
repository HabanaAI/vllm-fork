# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

import os
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, SpeculativeConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc, MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.torch_gated_delta_rule import (
    torch_chunk_gated_delta_rule_opt, torch_recurrent_gated_delta_rule_opt)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (Qwen3_5Config,
                                                     Qwen3_5TextConfig)
from vllm.transformers_utils.configs.qwen3_5_moe import (Qwen3_5MoeConfig,
                                                         Qwen3_5MoeTextConfig)

from .interfaces import (HasInnerState, IsHybrid, MixtureOfExperts,
                         MultiModalEmbeddings, SupportsLoRA, SupportsPP,
                         _require_is_multimodal)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (Qwen3NextAttention, Qwen3NextDecoderLayer,
                         Qwen3NextGatedDeltaNet, Qwen3NextModel,
                         Qwen3NextSparseMoeBlock, QwenNextMixtureOfExperts,
                         _save_conv_state, _save_ssm_state)
from .qwen3_vl import (Qwen3_VisionTransformer,
                       Qwen3_VisionTransformerStaticShape,
                       Qwen3VLDummyInputsBuilder,
                       Qwen3VLForConditionalGeneration,
                       Qwen3VLMultiModalProcessor, Qwen3VLProcessingInfo)
from .utils import (AutoWeightsLoader, PPMissingLayer,
                    _merge_multimodal_embeddings, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

is_hpu = current_platform.is_hpu()
logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            speculative_config=speculative_config,
            prefix=prefix,
        )
        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.prefix = prefix

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        raise NotImplementedError(
            "Qwen3.5 Series don't need to fix query key value ordering")

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 has separate in_proj_b and in_proj_a weights in the
        # checkpoint, which are loaded into the fused in_proj_ba parameter
        # via stacked_params_mapping with shard_id 0 and 1 respectively.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def adapt_gdn_inputs(self, hidden_states, attn_metadata):

        assert hidden_states.ndim == 3, \
            f"unexpected hidden_states shape: {hidden_states.shape}"

        #  1) input projection
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)

        b = b.contiguous().float()
        a = a.contiguous().float()
        mixed_qkv = mixed_qkv.float()

        beta = b.sigmoid().to(hidden_states.dtype)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        prefill_part = None
        decode_part = None

        num_prefills = attn_metadata.num_prefills
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        # 2) prefill part, packed layout: [1, total_tokens, h]
        if num_prefills > 0 and num_prefill_tokens > 0:
            prefill_idx = attn_metadata.mamba_cache_prefill_indices

            assert mixed_qkv.shape[0] == 1
            assert num_prefill_tokens % num_prefills == 0

            prefill_seq_len = num_prefill_tokens // num_prefills
            prefill_mixed_qkv = mixed_qkv[:, :num_prefill_tokens, :].reshape(
                num_prefills, prefill_seq_len, -1)
            prefill_z = z[:, :num_prefill_tokens, :].reshape(
                num_prefills, prefill_seq_len, -1)
            prefill_beta = beta[:, :num_prefill_tokens, :].reshape(
                num_prefills, prefill_seq_len, -1)
            prefill_g = g[:, :num_prefill_tokens, :].reshape(
                num_prefills, prefill_seq_len, -1)

            prev_conv_state = torch.index_select(self.conv_state, 0,
                                                 prefill_idx)
            prev_ssm_state = torch.index_select(self.ssm_state, 0, prefill_idx)

            # first chunk -> 0
            context_lens = attn_metadata.context_lens_tensor[:num_prefills]
            is_first_chunk = (context_lens == 0)
            prev_conv_state.masked_fill_(is_first_chunk[:, None, None], 0)
            prev_ssm_state.masked_fill_(is_first_chunk[:, None, None, None], 0)

            prefill_part = {
                "mixed_qkv": prefill_mixed_qkv,
                "z": prefill_z,
                "beta": prefill_beta,
                "g": prefill_g,
                "bs": prefill_mixed_qkv.shape[0],
                "seq_len": prefill_seq_len,
                "cache_indices": prefill_idx,
                "prev_conv_state": prev_conv_state,
                "prev_ssm_state": prev_ssm_state,
                "is_first_chunk": is_first_chunk,
            }

        # 3) decode part
        if num_decode_tokens > 0:
            decode_idx = attn_metadata.mamba_cache_decode_indices

            decode_mixed_qkv = mixed_qkv[:, num_prefill_tokens:, :].reshape(
                num_decode_tokens, 1, -1)
            decode_z = z[:, num_prefill_tokens:, :].reshape(
                num_decode_tokens, 1, -1)
            decode_beta = beta[:, num_prefill_tokens:, :].reshape(
                num_decode_tokens, 1, -1)
            decode_g = g[:, num_prefill_tokens:, :].reshape(
                num_decode_tokens, 1, -1)

            prev_conv_state = torch.index_select(self.conv_state, 0,
                                                 decode_idx)
            prev_ssm_state = torch.index_select(self.ssm_state, 0, decode_idx)

            decode_part = {
                "mixed_qkv": decode_mixed_qkv,
                "z": decode_z,
                "beta": decode_beta,
                "g": decode_g,
                "bs": decode_mixed_qkv.shape[0],
                "seq_len": 1,
                "cache_indices": decode_idx,
                "prev_conv_state": prev_conv_state,
                "prev_ssm_state": prev_ssm_state,
            }

        return prefill_part, decode_part, z

    def forward_chunked_prefill(self, hidden_states: torch.Tensor,
                                attn_metadata: AttentionMetadata
                                # output: torch.Tensor,
                                ):

        prefill_part, decode_part, z = self.adapt_gdn_inputs(
            hidden_states, attn_metadata)

        if self.conv1d_weight is None:
            self.conv1d_weight = self.conv1d.weight.squeeze(1).transpose(
                0, 1).flatten().reshape(self.conv_kernel_size,
                                        self.conv_dim // self.tp_size).float()
            del self.conv1d.weight

        core_attn_out = None
        if prefill_part is not None:
            qkv_dim = prefill_part["mixed_qkv"].shape[-1]
            conv_state_indices = \
                attn_metadata.conv_state_indices % prefill_part["seq_len"]
            prefill_conv_state = torch.index_select(
                prefill_part["mixed_qkv"].reshape(-1, qkv_dim),
                dim=0,
                index=conv_state_indices).reshape(-1,
                                                  self.conv_kernel_size - 1,
                                                  qkv_dim)
            mixed_qkv_with_pad = torch.cat(
                [prefill_part["prev_conv_state"], prefill_part["mixed_qkv"]],
                dim=1)

            # update conv_state
            mixed_qkv_with_pad = _save_conv_state(
                mixed_qkv_with_pad, prefill_conv_state, self.conv_state,
                prefill_part["cache_indices"])
            for idx in range(self.conv_kernel_size):
                qkv_slice = mixed_qkv_with_pad[:, idx:(
                    idx + prefill_part["seq_len"]), :]
                conv1d_weight_slice = self.conv1d_weight[idx]
                qkv_conv = qkv_slice * conv1d_weight_slice
                if idx == 0:
                    prefill_mixed_qkv_non_spec = qkv_conv
                else:
                    prefill_mixed_qkv_non_spec.add_(qkv_conv)

            prefill_mixed_qkv_non_spec = F.silu(prefill_mixed_qkv_non_spec)

            query, key, value = torch.split(
                prefill_mixed_qkv_non_spec.to(hidden_states.dtype),
                [
                    self.key_dim // self.tp_size,
                    self.key_dim // self.tp_size,
                    self.value_dim // self.tp_size,
                ],
                dim=-1,
            )
            query_non_spec = query.reshape(query.shape[0], query.shape[1], -1,
                                           self.head_k_dim)
            key_non_spec = key.reshape(key.shape[0], key.shape[1], -1,
                                       self.head_k_dim)
            value_non_spec = value.reshape(value.shape[0], value.shape[1], -1,
                                           self.head_v_dim)

            if self.num_v_heads // self.num_k_heads > 1:
                query_non_spec = query_non_spec.repeat_interleave(
                    self.num_v_heads // self.num_k_heads, dim=2)
                key_non_spec = key_non_spec.repeat_interleave(
                    self.num_v_heads // self.num_k_heads, dim=2)
            ssm_indices = torch.remainder(attn_metadata.seq_lens_tensor - 1,
                                          self.chunked_prefill_size) + 1
            prefill_core_attn_out, last_recurrent_state = (
                torch_chunk_gated_delta_rule_opt(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g=prefill_part["g"],
                    beta=prefill_part["beta"],
                    eye_constant=self.eye_constant,
                    valid_seq_len=ssm_indices,
                    chunk_size=self.chunk_size,
                    initial_state=prefill_part["prev_ssm_state"],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                ))
            if core_attn_out is None:
                core_attn_out = prefill_core_attn_out.reshape(
                    -1, prefill_core_attn_out.shape[-1])
            # update ssm_state
            core_attn_out = _save_ssm_state(core_attn_out,
                                            last_recurrent_state,
                                            self.ssm_state,
                                            prefill_part["cache_indices"])
        if decode_part is not None:
            decode_mixed_qkv_non_spec, cur_conv_state = causal_conv1d_update(
                decode_part["mixed_qkv"],
                self.conv_state,
                self.conv1d_weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=decode_part["cache_indices"],
            )
            decode_mixed_qkv_non_spec = _save_conv_state(
                decode_mixed_qkv_non_spec, cur_conv_state, self.conv_state,
                decode_part["cache_indices"])

            query, key, value = torch.split(
                decode_mixed_qkv_non_spec.to(hidden_states.dtype),
                [
                    self.key_dim // self.tp_size,
                    self.key_dim // self.tp_size,
                    self.value_dim // self.tp_size,
                ],
                dim=-1,
            )
            query_non_spec = query.reshape(query.shape[0], query.shape[1], -1,
                                           self.head_k_dim)
            key_non_spec = key.reshape(key.shape[0], key.shape[1], -1,
                                       self.head_k_dim)
            value_non_spec = value.reshape(value.shape[0], value.shape[1], -1,
                                           self.head_v_dim)

            if self.num_v_heads // self.num_k_heads > 1:
                query_non_spec = query_non_spec.repeat_interleave(
                    self.num_v_heads // self.num_k_heads, dim=2)
                key_non_spec = key_non_spec.repeat_interleave(
                    self.num_v_heads // self.num_k_heads, dim=2)

            decode_core_attn_out, last_recurrent_state = (
                torch_recurrent_gated_delta_rule_opt(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g=decode_part["g"],
                    beta=decode_part["beta"],
                    recurrent_state=decode_part["prev_ssm_state"],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                ))
            if core_attn_out is None:
                core_attn_out = decode_core_attn_out.reshape(
                    -1, decode_core_attn_out.shape[-1])
            else:
                core_attn_out = torch.cat([
                    core_attn_out,
                    decode_core_attn_out.reshape(
                        -1, decode_core_attn_out.shape[-1])
                ],
                                          dim=0)
            core_attn_out = _save_ssm_state(core_attn_out,
                                            last_recurrent_state,
                                            self.ssm_state,
                                            decode_part["cache_indices"])

        # Output Projection
        z_shape_og = z.shape
        # reshape input data into 2D tensor
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z).to(hidden_states.dtype)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                              core_attn_out.shape[1], -1)

        output, _ = self.out_proj(core_attn_out)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        # output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)

        if attn_metadata.chunk_prefill_enabled:
            return self.forward_chunked_prefill(hidden_states, attn_metadata)

        conv_state = self.conv_state
        ssm_state = self.ssm_state

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)

        b = b.contiguous().float()
        a = a.contiguous().float()
        mixed_qkv = mixed_qkv.float()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        mamba_cache_prefill_indices = attn_metadata.mamba_cache_prefill_indices
        mamba_cache_decode_indices = attn_metadata.mamba_cache_decode_indices

        if self.conv1d_weight is None:
            self.conv1d_weight = self.conv1d.weight.squeeze(1).transpose(
                0, 1).flatten().reshape(self.conv_kernel_size,
                                        self.conv_dim // self.tp_size).float()
            del self.conv1d.weight

        if attn_metadata.is_prompt:
            bs, seq_len, qkv_dim = mixed_qkv.shape
            conv_state_indices = attn_metadata.conv_state_indices
            prefill_conv_state = torch.index_select(
                mixed_qkv.reshape(-1, qkv_dim),
                dim=0,
                index=conv_state_indices).reshape(bs, -1, qkv_dim)
            mixed_qkv = _save_conv_state(mixed_qkv, prefill_conv_state,
                                         conv_state,
                                         mamba_cache_prefill_indices)
            mixed_qkv_with_pad = F.pad(mixed_qkv,
                                       (0, 0, self.conv_kernel_size - 1, 0))
            for idx in range(self.conv_kernel_size):
                qkv_slice = mixed_qkv_with_pad[:, idx:(idx + seq_len), :]
                conv1d_weight_slice = self.conv1d_weight[idx]
                qkv_conv = qkv_slice * conv1d_weight_slice
                if idx == 0:
                    mixed_qkv_non_spec = qkv_conv
                else:
                    mixed_qkv_non_spec.add_(qkv_conv)

            mixed_qkv_non_spec = F.silu(mixed_qkv_non_spec)

        else:
            mixed_qkv_non_spec, cur_conv_state = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d_weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_decode_indices,
            )
            mixed_qkv_non_spec = _save_conv_state(
                mixed_qkv_non_spec,
                cur_conv_state,
                conv_state,
                mamba_cache_decode_indices,
            )

        query, key, value = torch.split(
            mixed_qkv_non_spec.to(hidden_states.dtype),
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query_non_spec = query.reshape(query.shape[0], query.shape[1], -1,
                                       self.head_k_dim)
        key_non_spec = key.reshape(key.shape[0], key.shape[1], -1,
                                   self.head_k_dim)
        value_non_spec = value.reshape(value.shape[0], value.shape[1], -1,
                                       self.head_v_dim)

        beta = b.sigmoid().to(hidden_states.dtype)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query_non_spec = query_non_spec.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2)
            key_non_spec = key_non_spec.repeat_interleave(self.num_v_heads //
                                                          self.num_k_heads,
                                                          dim=2)

        if attn_metadata.is_prompt:
            core_attn_out, last_recurrent_state = (
                torch_chunk_gated_delta_rule_opt(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g=g,
                    beta=beta,
                    eye_constant=self.eye_constant,
                    valid_seq_len=attn_metadata.seq_lens_tensor,
                    chunk_size=self.chunk_size,
                    inv_loop=self.inv_loop,
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                ))
            core_attn_out = _save_ssm_state(core_attn_out,
                                            last_recurrent_state, ssm_state,
                                            mamba_cache_prefill_indices)
        else:
            recurrent_state = torch.index_select(
                ssm_state,
                dim=0,
                index=mamba_cache_decode_indices,
            )
            core_attn_out, last_recurrent_state = (
                torch_recurrent_gated_delta_rule_opt(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g=g,
                    beta=beta,
                    recurrent_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                ))
            core_attn_out = _save_ssm_state(core_attn_out,
                                            last_recurrent_state, ssm_state,
                                            mamba_cache_decode_indices)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z).to(hidden_states.dtype)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                              core_attn_out.shape[1], -1)

        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                vllm_config,
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                # prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size,
                                              eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size,
                                                       eps=config.rms_norm_eps)

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                ), )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                ), )

        self.graph_break = os.environ.get("VLLM_MOE_GRAPH_BREAK",
                                          "false").lower() == "true"


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen3_5Model(Qwen3NextModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config)

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = \
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size)

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = True  #False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (self.config.num_experts if hasattr(
            self.config, "num_experts") else 0)
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:  # noqa: E501
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (name_mapped.endswith(".bias")
                                or name_mapped.endswith("_bias")
                            ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"  # noqa: E501
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
        nn.Module,
        HasInnerState,
        SupportsLoRA,
        SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        # cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        # if cache_config.mamba_cache_mode == "all":
        #     raise NotImplementedError(
        #         "Qwen3.5 currently does not support 'all' prefix caching, "
        #         "please use '--mamba-cache-mode=align' instead"
        #     )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration,
                                      IsHybrid):
    packed_modules_mapping = \
        Qwen3VLForConditionalGeneration.packed_modules_mapping | {
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
        }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        # self.use_data_parallel = \
        #     multimodal_config.mm_encoder_tp_mode == "data"
        # self.video_pruning_rate = multimodal_config.video_pruning_rate
        # self.is_multimodal_pruning_enabled = (
        #     multimodal_config.is_multimodal_pruning_enabled()
        # )

        # with self._mark_tower_model(vllm_config, {"image", "video"}):
        if is_hpu:
            qwen3_visionTransformer = Qwen3_VisionTransformerStaticShape
        else:
            qwen3_visionTransformer = Qwen3_VisionTransformer
        self.visual = qwen3_visionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=None,
            prefix=maybe_prefix(prefix, "visual"),
        )

        # with self._mark_language_model(vllm_config):
        self.language_model = Qwen3_5ForCausalLM(vllm_config=vllm_config,
                                                 prefix=maybe_prefix(
                                                     prefix, "language_model"))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        self.use_deepstack = False
        self.text_dim = config.text_config.hidden_size
        self.mm_offset_image = 0
        self.mm_offset_image_multiscale = 0
        self.mm_offset_video = 0
        self.mm_offset_video_multiscale = 0

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None
        seq_len = input_ids.shape[
            -1] if input_ids is not None else inputs_embeds.shape[-2]
        if positions.ndim == 1 and positions.shape[
                0] == seq_len * 3:  # recover flattened position
            positions = positions.reshape(3, -1)
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype)

    @classmethod
    def get_mamba_state_shape_from_config(
            cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (vllm_config.speculative_config.num_speculative_tokens
                    if vllm_config.speculative_config else 0)
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
            cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = \
            num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                    layer.mlp, Qwen3NextSparseMoeBlock):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration,
                                         Qwen3_5_MoeMixtureOfExperts):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        # self.use_data_parallel = \
        #     multimodal_config.mm_encoder_tp_mode == "data"
        # self.video_pruning_rate = multimodal_config.video_pruning_rate
        # self.is_multimodal_pruning_enabled = (
        #     multimodal_config.is_multimodal_pruning_enabled()
        # )

        # with self._mark_tower_model(vllm_config, {"image", "video"}):
        if is_hpu:
            qwen3_visionTransformer = Qwen3_VisionTransformerStaticShape
        else:
            qwen3_visionTransformer = Qwen3_VisionTransformer
        self.visual = qwen3_visionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=None,
            prefix=maybe_prefix(prefix, "visual"),
        )

        # with self._mark_language_model(vllm_config):
        self.language_model = Qwen3_5MoeForCausalLM(vllm_config=vllm_config,
                                                    prefix=maybe_prefix(
                                                        prefix,
                                                        "language_model"))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        self.use_deepstack = False
        self.text_dim = config.text_config.hidden_size

        # set MoE hyperparameters
        self.set_moe_parameters()
        self.mm_offset_image = 0
        self.mm_offset_image_multiscale = 0
        self.mm_offset_video = 0
        self.mm_offset_video_multiscale = 0
