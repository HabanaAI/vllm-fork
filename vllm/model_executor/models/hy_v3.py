# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# coding=utf-8
# Copyright 2026 The HY team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only HY model compatible with HuggingFace weights."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


def _cfg(cfg: PretrainedConfig, k: str, default: Any) -> Any:
    return getattr(cfg, k, default)


def _is_moe_layer(cfg: PretrainedConfig, layer_id: int) -> bool:
    return layer_id >= int(_cfg(cfg, "first_k_dense_replace", 0))


class HYV3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported hidden_act={hidden_act!r}, expected 'silu'.")
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HYV3Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        rope_scaling: Optional[dict[str, Any]] = None,
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        hs = int(config.hidden_size)
        total_heads = int(config.num_attention_heads)
        total_kv_heads = int(config.num_key_value_heads)
        head_dim = int(config.head_dim)

        tp = get_tensor_model_parallel_world_size()
        assert total_heads % tp == 0
        self.num_heads = total_heads // tp

        if total_kv_heads >= tp:
            assert total_kv_heads % tp == 0
        else:
            assert tp % total_kv_heads == 0
        self.num_kv_heads = max(1, total_kv_heads // tp)

        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hs,
            head_size=head_dim,
            total_num_heads=total_heads,
            total_num_kv_heads=total_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=total_heads * head_dim,
            output_size=hs,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=int(config.max_position_embeddings),
            base=float(config.rope_theta),
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.use_qk_norm = bool(_cfg(config, "qk_norm", False))
        if self.use_qk_norm:
            eps = float(_cfg(config, "rms_norm_eps", 1e-6))
            self.q_norm = RMSNorm(head_dim, eps=eps)
            self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q.view(-1, self.num_heads,
                                   self.head_dim)).view_as(q)
            k = self.k_norm(k.view(-1, self.num_kv_heads,
                                   self.head_dim)).view_as(k)

        q, k = self.rotary_emb(positions, q, k)

        out = self.attn(q, k, v)
        out, _ = self.o_proj(out)
        return out


class HYV3Router(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.gate(x)
        return logits


class HYV3MoEBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = -1,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.expert_hidden_dim = int(config.expert_hidden_dim)
        self.num_shared_experts = int(_cfg(config, "num_shared_experts", 0))

        router_scaling_factor = float(
            _cfg(config, "router_scaling_factor", 1.0))

        self.router = HYV3Router(
            self.hidden_size,
            self.num_experts,
            prefix=f"{prefix}.router",
        )

        self.expert_bias = nn.Parameter(torch.empty(self.num_experts))

        if self.num_shared_experts > 0:
            self.shared_mlp = HYV3MLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.expert_hidden_dim *
                self.num_shared_experts,
                hidden_act=str(config.hidden_act),
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp",
                reduce_results=False,
            )
        else:
            self.shared_mlp = None

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.expert_hidden_dim,
            reduce_results=False,
            scoring_func="sigmoid",
            renormalize=bool(_cfg(config, "route_norm", True)),
            e_score_correction_bias=self.expert_bias,
            router_scaling_factor=router_scaling_factor,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, orig_shape[-1])

        router_logits = self.router(x).to(torch.float32)

        out = self.experts(hidden_states=x, router_logits=router_logits)

        if self.shared_mlp is not None:
            out = out + self.shared_mlp(x)

        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)

        return out.view(orig_shape)


class HYV3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hs = int(config.hidden_size)
        eps = float(_cfg(config, "rms_norm_eps", 1e-6))
        self.layer_id = layer_id

        self.self_attn = HYV3Attention(
            config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            layer_id=layer_id,
        )
        self.input_layernorm = RMSNorm(hs, eps=eps)
        self.post_attention_layernorm = RMSNorm(hs, eps=eps)

        if not hasattr(config, "first_k_dense_replace"):
            raise ValueError(
                "first_k_dense_replace not exist, please check config")

        if _is_moe_layer(config, layer_id):
            self.is_moe = True
            self.block_type = "moe"
            self.mlp = HYV3MoEBlock(
                config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.is_moe = False
            self.block_type = "feedforward"
            self.mlp = HYV3MLP(
                hidden_size=hs,
                intermediate_size=int(config.intermediate_size),
                hidden_act=str(config.hidden_act),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=True,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class HYV3Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config

        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = int(config.vocab_size) + lora_vocab
        self.org_vocab_size = int(config.vocab_size)

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                int(config.hidden_size),
                org_num_embeddings=int(config.vocab_size),
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            int(config.num_hidden_layers),
            lambda prefix: HYV3DecoderLayer(
                config=config,
                layer_id=int(prefix.split(".")[-1]),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                int(config.hidden_size),
                eps=float(_cfg(config, "rms_norm_eps", 1e-6)),
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"],
                int(config.hidden_size),
            ))

        self.moe_layers: list[nn.Module] = []
        example_moe = None
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, HYV3DecoderLayer)
            if layer.block_type == "moe":
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.num_logical_experts = int(
            config.num_experts) if example_moe else None

        self.num_redundant_experts = 0
        self.num_physical_experts = self.num_logical_experts
        self.num_routed_experts = self.num_logical_experts

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=int(self.config.num_experts),
        )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        pass

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = (inputs_embeds if inputs_embeds is not None else
                             self.get_input_embeddings(input_ids))
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[i](positions,
                                                     hidden_states,
                                                     residual,
                                                     idx=i)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        raise NotImplementedError("Call HYV3ForCausalLM.load_weights instead.")


class HYV3ForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = HYV3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = int(config.vocab_size)
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                int(config.hidden_size),
                org_num_embeddings=int(config.vocab_size),
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                quant_config=quant_config,
            )
            if bool(_cfg(config, "tie_word_embeddings", False)):
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = float(_cfg(config, "logit_scale", 1.0))
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                int(config.vocab_size),
                logit_scale,
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """
        Weight mapping:
        ckpt key                                  → model param
        ─────────────────────────────────────────────────────────────────
        self_attn.{q,k,v}_proj.weight         → self_attn.qkv_proj 
        mlp.{gate,up}_proj.weight (dense)     → mlp.gate_up_proj  
        mlp.shared_mlp.{gate,up}_proj.weight  → mlp.shared_mlp.gate_up_proj 
        mlp.experts.<eid>.{gate,up,down}_proj → mlp.experts
        mlp.router.gate.weight                → mlp.router.gate.weight
        mlp.expert_bias                       → mlp.expert_bias 
        everything else                       → direct
        """

        params_dict: dict[str, torch.Tensor] = dict(self.named_parameters())

        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".mlp.gate_up_proj", ".mlp.gate_proj", 0),
            (".mlp.gate_up_proj", ".mlp.up_proj", 1),
            (".mlp.shared_mlp.gate_up_proj", ".mlp.shared_mlp.gate_proj", 0),
            (".mlp.shared_mlp.gate_up_proj", ".mlp.shared_mlp.up_proj", 1),
        ]

        expert_params_mapping = self.model.get_expert_mapping()

        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (bool(_cfg(self.config, "tie_word_embeddings", False))
                    and "lm_head.weight" in name):
                continue

            # ---- branch 1: stacked params --------------------------------
            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name_m = name.replace(weight_name, param_name)
                if name_m.endswith(".bias") and name_m not in params_dict:
                    continue
                if is_pp_missing_parameter(name_m, self):
                    continue
                param = params_dict[name_m]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_m)
                is_found = True
                break
            if is_found:
                continue

            # ---- branch 2: FusedMoE expert weights -----------------------
            expert_found = False
            for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
            ) in expert_params_mapping:
                if weight_name not in name:
                    continue
                name_m = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name_m, self):
                    expert_found = True
                    break
                if name_m not in params_dict:
                    continue
                param = params_dict[name_m]
                weight_loader = typing.cast(Callable[..., bool],
                                            param.weight_loader)
                weight_loader(
                    param,
                    loaded_weight,
                    name_m,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(name_m)
                expert_found = True
                break
            if expert_found:
                continue

            # ---- branch 3: direct load -----------------------------------
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
