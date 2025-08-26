# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn.functional as F
import os
from torch import nn
from transformers import Gemma3TextConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               SplitQKVParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)

is_hpu = current_platform.is_hpu()
if is_hpu:
    import habana_frameworks.torch as htorch

# split_size>128: fixed-length splits (each slice is split_size)
# split_size<128: fixed-num splits (split_size num of slices)
def get_split_size(seq_len, batch_size, orig_split_size):
    if orig_split_size<128:
        split_size = max((seq_len*batch_size)//orig_split_size, 1)
    else:
        split_size = orig_split_size
    return split_size

# Use the first override whenever possible
VLLM_MLP_SIZE_OVERRIDE = int(os.environ.get("VLLM_MLP_SIZE_OVERRIDE", "512"))
# Use the second override 
VLLM_MLP_SIZE_OVERRIDE_2 = int(os.environ.get("VLLM_MLP_SIZE_OVERRIDE_2", "384"))

class Gemma3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        do_split: bool = False,
        split_size: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`.")
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(1)
        if (seq_len*batch_size)%VLLM_MLP_SIZE_OVERRIDE==0:
            x = x.view(-1,VLLM_MLP_SIZE_OVERRIDE,self.hidden_size)
        elif (seq_len*batch_size)%VLLM_MLP_SIZE_OVERRIDE_2==0:
            x = x.view(-1,VLLM_MLP_SIZE_OVERRIDE_2,self.hidden_size)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        # Separate split for down is not implemented yet
        x, _ = self.down_proj(x)
        if ((seq_len*batch_size)%VLLM_MLP_SIZE_OVERRIDE==0) or ((seq_len*batch_size)%VLLM_MLP_SIZE_OVERRIDE_2==0):
            x = x.view(batch_size,seq_len,self.hidden_size)
        return x


class Gemma3Attention(nn.Module):

    def __init__(self,
                 config: Gemma3TextConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 attn_logits_soft_cap: Optional[float] = None,
                 prefix: str = "",
                 do_split: bool = False,
                 split_size: int = 2,
                 output_slice: bool = False) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.split_qkv = cache_config.split_qkv
        self.do_split = do_split
        self.split_size = split_size
        self.output_slice = output_slice

        if self.split_qkv:
            self.qkv_proj = SplitQKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=config.attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=config.attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # TODO(woosuk): Add reference to the original HF implementation.
        layer_idx = extract_layer_index(prefix)
        self.is_sliding = (getattr(
            config, "interleaved_sliding_window", None) is not None and bool(
                (layer_idx + 1) % config.sliding_window_pattern))
        # Initialize the rotary embedding.
        if self.is_sliding:
            # Local attention. Override the values in config.json.
            self.rope_theta = config.rope_local_base_freq
            self.rope_scaling = {"rope_type": "default"}
            self.sliding_window = config.interleaved_sliding_window
        else:
            # Global attention. Use the values in config.json.
            self.rope_theta = config.rope_theta
            self.rope_scaling = config.rope_scaling
            self.sliding_window = None
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
            rope_scaling=self.rope_scaling,
        )

        # Initialize the attention.
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              logits_soft_cap=attn_logits_soft_cap,
                              per_layer_sliding_window=self.sliding_window,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        attn_metadata = kwargs['attn_metadata']
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        split_size = get_split_size(seq_len, batch_size, self.split_size)
        do_split = self.do_split and attn_metadata.is_prompt

        if self.split_qkv:
            q, k, v, _ = self.qkv_proj(hidden_states)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for rotary embedding
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if do_split and (seq_len * batch_size) // split_size >= 2:
            # Split tensors by sequence
            q_chunks = torch.split(q, split_size, dim=1)
            k_chunks = torch.split(k, split_size, dim=1)
            v = v.view(batch_size, seq_len, -1)
            v_chunks = torch.split(v, split_size, dim=1)
            pos_chunks = torch.split(positions, split_size, dim=1)

            attn_output_chunks = []
            for q_i, k_i, v_i, pos_i in zip(q_chunks, k_chunks, v_chunks, pos_chunks):
                # Flatten for rotary_emb (assumes rotary_emb expects [B*T, heads, dim])
                q_i = q_i.reshape(-1, self.head_dim)
                k_i = k_i.reshape(-1, self.head_dim)
                pos_i_flat = pos_i.reshape(-1)

                q_i, k_i = self.rotary_emb(pos_i_flat, q_i, k_i)

                # Reshape back
                q_i = q_i.reshape(-1, self.num_heads, self.head_dim).contiguous()
                k_i = k_i.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()
                v_i = v_i.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()

                # Run attention
                attn_out_i = self.attn(q_i, k_i, v_i)
                attn_output_chunks.append(attn_out_i)

            # Combine all attention slices
            attn_output = torch.cat(attn_output_chunks, dim=1)
            attn_output = attn_output.view(1, -1, self.q_size)

            # Output projection
            attn_list = torch.split(attn_output, split_size, dim=1)
            output_list = [self.o_proj(slice_)[0] for slice_ in attn_list]

            if self.output_slice:
                return output_list
            else:
                output = torch.cat(output_list, dim=1)
                output = output.view(batch_size, seq_len, self.hidden_size)
                return output
        else:
            # No split path
            q = q.view(-1, self.head_dim)
            k = k.view(-1, self.head_dim)
            positions = positions.reshape(-1)

            q, k = self.rotary_emb(positions, q, k)

            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)

            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

        # NOTE(woosuk): Gemma3 uses bidirectional attention between image tokens
        # that correspond to the same image while using causal attention
        # otherwise. Current attention backends cannot handle this pattern, so
        # we temporarily use a naive attention implementation with mask tensors.

        # We intentionally keep the attention backend as-is and only override
        # `attn_output` with the naive implementation's output. This minimizes
        # changes to existing model runners and attention backends. The call to
        # `self.attn(q, k, v)` is only used to populate the KV cache - its
        # output is discarded and overwritten below. While this duplicates
        # computation, it maintains compatibility.
        # TODO(woosuk): Optimize by implementing custom attention kernels.
        attn_output = self.naive_attn_with_masks(q,
                                                 k,
                                                 v,
                                                 out=attn_output,
                                                 **kwargs)

    def naive_attn_with_masks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # NOTE(woosuk): As described in the comment above, this code is not
        # meant to be performant. It is only meant to be correct.
        q = q.view(-1, self.num_heads, self.head_dim)
        # Expand the key and value to handle GQA.
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        k = k.repeat_interleave(num_queries_per_kv, dim=-2)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        v = v.repeat_interleave(num_queries_per_kv, dim=-2)

        if self.is_sliding:
            attn_masks = kwargs["local_attn_masks"]
        else:
            attn_masks = kwargs["global_attn_masks"]

        seq_lens = kwargs["seq_lens"]
        start_idx = 0
        for seq_len, attn_mask in zip(seq_lens, attn_masks):
            end_idx = start_idx + seq_len
            query = q[start_idx:end_idx].unsqueeze(0)
            key = k[start_idx:end_idx].unsqueeze(0)
            value = v[start_idx:end_idx].unsqueeze(0)

            # Transpose.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                self.scaling,
            )
            output = output.transpose(1, 2).flatten(-2, -1)
            out[start_idx:end_idx] = output
            start_idx = end_idx
        return out


class Gemma3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma3TextConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        split_size = int(os.environ.get('VLLM_TP_SPLIT_SIZE_BY_SEQ', '1'))
        print("==========split_size: ", split_size) 
        output_slice = int(os.environ.get('OUTPUT_SLICE', '1')) == 1
        do_split = split_size > 1
        self.split_size = split_size
        self.do_split = do_split
        self.output_slice = output_slice and do_split
        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=None,
            prefix=f"{prefix}.self_attn",
            do_split=do_split,
            split_size=split_size,
            output_slice=output_slice,
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            do_split=do_split,
            split_size=split_size,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size,
                                                       eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get prompt bs from attn_metadata. The one from hidden_states may be inaccurate due to slicing
        attn_metadata = kwargs['attn_metadata']
        if attn_metadata.is_prompt:
            batch_size = attn_metadata.seq_lens_tensor.size(0)
        else:
            batch_size = 1
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            if type(hidden_states) is list:
                # TP parallel slice cross layers
                residual_list_output = []
                for hidden_states_ind, residual_ind in zip(hidden_states, residual):
                    hidden_states_ind, residual_ind = self.input_layernorm(hidden_states_ind, residual_ind)
                    residual_list_output.append(residual_ind)
                residual = torch.cat(residual_list_output, dim=1)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
        
        if (residual is not None) and type(hidden_states) is list:
            hidden_states_shape = residual.shape
            batch_size_fake, seq_len_fake, hidden_size = hidden_states_shape
            hidden_states = self.self_attn(positions=positions,
                                    hidden_states=hidden_states_ind,
                                    attn_metadata=attn_metadata)
        else:
            hidden_states_shape = hidden_states.shape
            batch_size_fake, seq_len_fake, hidden_size = hidden_states_shape
            hidden_states = self.self_attn(positions=positions,
                                    hidden_states=hidden_states,
                                    attn_metadata=attn_metadata)
        
        # Calculate real seq_len from product of inaccurate batch_size and seq_len
        seq_len = (batch_size_fake*seq_len_fake)//batch_size
        split_size = get_split_size(seq_len, batch_size, self.split_size)
        # only split for prefill
        do_split = self.do_split and attn_metadata.is_prompt

        # self_attn output a list of tensors to be processed sequential at layernorm and mlp
        if do_split and (seq_len*batch_size)//split_size>=2 and self.output_slice:
            # Slice residual
            residual = residual.view(1, -1, hidden_size)
            residual_list = torch.split(residual, split_size, 1)

            residual_list_output = []
            output_list = []
            # Sequentially process slices
            for hidden_state, residual in zip(hidden_states, residual_list):
                hidden_state, residual = self.post_attention_layernorm(hidden_state, residual)
                hidden_state, residual = self.pre_feedforward_layernorm(
                    hidden_state, residual)
                hidden_state = self.mlp(hidden_state)
                hidden_state = self.post_feedforward_layernorm(hidden_state)
                residual_list_output.append(residual)
                output_list.append(hidden_state)

            residual = residual_list_output
            hidden_states = output_list

        else:
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            hidden_states, residual = self.pre_feedforward_layernorm(
                hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)

        return hidden_states, residual

@support_torch_compile
class Gemma3Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma3DecoderLayer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer",
                             torch.tensor(normalizer),
                             persistent=False)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self.split_qkv = cache_config.split_qkv

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # NOTE(woosuk): Only apply the normalizer to the output of
        # vocab embedding. Don't apply it to the vision embedding.
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                **kwargs,
            )
        if type(hidden_states) is list:
            hidden_states = torch.cat(hidden_states, dim=1)
            residual = torch.cat(residual, dim=1)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        if not self.split_qkv:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]
        else:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("qkv_proj.q_proj", "q_proj", "q"),
                ("qkv_proj.k_proj", "k_proj", "k"),
                ("qkv_proj.v_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                if self.split_qkv and (shard_id == "q" or shard_id == "v"
                                       or shard_id == "k"):
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Gemma3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        del lora_config  # Unused.
        super().__init__()
        self.config = config
        # currently all existing Gemma models have `tie_word_embeddings` enabled
        assert config.tie_word_embeddings
        self.quant_config = quant_config
        self.model = Gemma3Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.embed_tokens, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
