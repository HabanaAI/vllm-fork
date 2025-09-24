# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm.attention import Attention, AttentionBackend, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CacheConfig, ModelConfig, SpeculativeConfig,
                         VllmConfig, get_current_vllm_config)
from vllm.distributed import (divide, get_ep_group, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
# yapf conflicts with isort for this block
# yapf: disable
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm)
from vllm.model_executor.layers.layernorm import RMSNormGated
# yapf: enable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    mamba_v2_sharded_weight_loader)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextConfig

from .interfaces import (HasInnerState, IsHybrid, MixtureOfExperts,
                         SupportsLoRA, SupportsPP)
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        head_dim = query.size(-1)
        inv_scale = head_dim**-0.5
        query = F.rms_norm(query, (head_dim, ), eps=1e-6) * inv_scale
        key = F.rms_norm(key, (head_dim, ), eps=1e-6) * inv_scale
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1]**0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size,
                                 chunk_size,
                                 dtype=torch.bool,
                                 device=query.device),
                      diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - 
                   g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((torch.matmul(k_beta.contiguous(),
                           key.transpose(-1, -2).contiguous())) *
             decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, sequence_length,
                                        k_head_dim, v_head_dim).to(value) if
                            initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size,
                                 chunk_size,
                                 dtype=torch.bool,
                                 device=query.device),
                      diagonal=1)

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * 
                decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp() +
            (k_i *
             (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                 -1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    else:
        last_recurrent_state = last_recurrent_state.to(initial_dtype)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                          core_attn_out.shape[1],
                                          -1,
                                          core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1,
                                            2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    recurrent_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        head_dim = query.size(-1)
        inv_scale = head_dim**-0.5
        query = F.rms_norm(query, (head_dim, ), eps=1e-6) * inv_scale
        key = F.rms_norm(key, (head_dim, ), eps=1e-6) * inv_scale
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1]**0.5)
    query = query * scale

    recurrent_state = recurrent_state.to(value)

    if num_heads > 1:
        core_attn_out = torch.zeros(batch_size, sequence_length, num_heads,
                                    v_head_dim).to(value)
        for i in range(num_heads):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            recurrent_state = recurrent_state * g_t
            kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            recurrent_state = recurrent_state + k_t.unsqueeze(
                -1) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (recurrent_state *
                                      q_t.unsqueeze(-1)).sum(dim=-2)
    else:
        q_t = query.squeeze(-2)
        k_t = key.squeeze(-2)
        v_t = value.squeeze(-2)
        g_t = g.squeeze(-1).exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta

        recurrent_state = recurrent_state * g_t
        kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        recurrent_state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2))
        core_attn_out = (recurrent_state *
                         q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(-2)

    if not output_final_state:
        recurrent_state = None
    else:
        recurrent_state = recurrent_state.to(initial_dtype)
    core_attn_out = core_attn_out.transpose(1,
                                            2).contiguous().to(initial_dtype)
    return core_attn_out, recurrent_state


class Qwen3NextSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        # Load balancing settings.
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = (self.n_logical_experts +
                                   self.n_redundant_experts)
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = (self.ep_rank *
                                      self.n_local_physical_experts)
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        self.experts = FusedMoE(num_experts=self.n_routed_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                reduce_results=False,
                                renormalize=config.norm_topk_prob,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts")

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=f"{prefix}.gate")

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=self.experts.must_reduce_shared_expert_outputs(
                ),
            )
        else:
            self.shared_expert = None
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size,
                                                  1,
                                                  bias=False)

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid gate quantization.
        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(
                    self.shared_expert_gate(hidden_states)) * shared_output

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states)

        return final_hidden_states.view(orig_shape)


class Qwen3NextGatedDeltaNet(nn.Module, MambaBase):

    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
        return GDNAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
            use_v1=True)

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Qwen3NextConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.fla_layer_idx = self.layer_idx // config.full_attention_interval *
            (config.full_attention_interval - 1) +
            self.layer_idx % config.full_attention_interval
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (self.speculative_config.num_speculative_tokens
                         if self.speculative_config else 0)

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.projection_size_ba = self.num_v_heads * 2
        self.in_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.projection_size_qkvz, self.projection_size_ba],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ], self.tp_size, self.tp_rank)
            })

        conv_state_shape = (vllm_config.scheduler_config.max_num_seqs * 2 + 1,
                            self.conv_kernel_size - 1,
                            divide(self.conv_dim, self.tp_size),
        )
        temporal_state_shape = (
            vllm_config.scheduler_config.max_num_seqs * 2 + 1,
            divide(self.num_v_heads, self.tp_size),
            self.head_k_dim, self.head_v_dim
        )

        self.conv_state = torch.empty(conv_state_shape,
                                      dtype=torch.float32,
                                      device=self.conv1d.weight.device)
        self.ssm_state = torch.empty(temporal_state_shape,
                                     dtype=torch.float32,
                                     device=self.conv1d.weight.device)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size), )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            ))

        set_weight_attrs(self.A_log,
                         {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            dtype=config.torch_dtype,
        )

        self.out_proj = RowParallelLinear(self.value_dim,
                                          self.hidden_size,
                                          bias=False,
                                          input_is_parallel=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_proj")

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz,
        mixed_ba,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (self.head_k_dim + self.head_k_dim +
             (self.head_v_dim + self.head_v_dim) * self.num_v_heads //
             self.num_k_heads),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz,
                                             split_arg_list_qkvz,
                                             dim=3)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=3)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, 'l (h d) -> 1 l h d', d=self.head_k_dim),
            (query, key))
        value = rearrange(value, 'l (h d) -> 1 l h d', d=self.head_v_dim)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        conv_state = self.conv_state
        ssm_state = self.ssm_state

        projected_states, _ = self.in_proj(hidden_states)
        projected_states = projected_states.float()
        projected_states_qkvz, projected_states_ba = torch.split(
            projected_states,
            [
                self.projection_size_qkvz // self.tp_size,
                self.projection_size_ba // self.tp_size
            ],
            dim=-1,
        )
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) \
            for x in (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        mamba_cache_prefill_indices = attn_metadata.mamba_cache_prefill_indices
        mamba_cache_decode_indices = attn_metadata.mamba_cache_decode_indices

        if attn_metadata.is_prompt:
            bs, seq_len, qkv_dim = mixed_qkv.shape
            conv_state_indices = attn_metadata.conv_state_indices
            prefill_conv_state = torch.index_select(
                mixed_qkv.reshape(-1, qkv_dim),
                dim=0,
                index=conv_state_indices
            ).reshape(bs, -1, qkv_dim)
            conv_state.index_copy_(
                dim=0,
                index=mamba_cache_prefill_indices,
                source=prefill_conv_state
            )

            mixed_qkv_non_spec = F.conv1d(
                mixed_qkv.transpose(1, 2).contiguous(),
                self.conv1d.weight.contiguous().float(),
                bias=None,
                padding=self.conv_kernel_size - 1,
                groups=(self.conv_dim//self.tp_size)
            )
            mixed_qkv_non_spec = F.silu(mixed_qkv_non_spec)
            mixed_qkv_non_spec = \
                mixed_qkv_non_spec[:, :, :seq_len].transpose(1, 2)

        else:
            mixed_qkv_non_spec, cur_conv_state = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.float(),
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_decode_indices,
            )
            conv_state.index_copy_(0,
                                   mamba_cache_decode_indices,
                                   cur_conv_state)

        query, key, value = torch.split(
            mixed_qkv_non_spec,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query_non_spec = query.reshape(query.shape[0],
                                       query.shape[1],
                                       -1,
                                       self.head_k_dim)
        key_non_spec = key.reshape(key.shape[0],
                                   key.shape[1],
                                   -1,
                                   self.head_k_dim)
        value_non_spec = value.reshape(value.shape[0],
                                       value.shape[1],
                                       -1,
                                       self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query_non_spec = query_non_spec.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key_non_spec = key_non_spec.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if attn_metadata.is_prompt:
            core_attn_out, last_recurrent_state = (
                torch_chunk_gated_delta_rule(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                ))
            ssm_state.index_copy_(dim=0,
                                  index=mamba_cache_prefill_indices,
                                  source=last_recurrent_state)
        else:
            recurrent_state = torch.index_select(
                ssm_state,
                dim=0,
                index=mamba_cache_decode_indices,
            )
            core_attn_out, last_recurrent_state = (
                torch_recurrent_gated_delta_rule(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g=g,
                    beta=beta,
                    recurrent_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                ))
            ssm_state.index_copy_(
                dim=0,
                index=mamba_cache_decode_indices,
                source=last_recurrent_state,
            )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z).to(hidden_states.dtype)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                              core_attn_out.shape[1],
                                              -1)

        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3NextAttention(nn.Module):

    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None)
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            partial_rotary_factor=config.partial_rotary_factor,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config":
                self.dual_chunk_attention_config,
            } if self.dual_chunk_attention_config else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        bs, seq, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        q = q.reshape(bs, seq, -1)
        k = k.reshape(bs, seq, -1)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)

        return output


class Qwen3NextDecoderLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Qwen3NextConfig,
        layer_type: str,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
                vllm_config,
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f'{prefix}.linear_attn')
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f'{prefix}.self_attn',
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (self.layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (self.layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3NextSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    self.config.hidden_size,
                    dtype=config.torch_dtype,
                ), )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    self.config.hidden_size,
                    dtype=config.torch_dtype,
                ), )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        if self.layer_type == "linear_attention":
            self_attention_output = self.linear_attn(
                hidden_states=hidden_states, )
        elif self.layer_type == "full_attention":
            self_attention_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert len(hidden_states.shape) == len(
                    self.ffn_layer_scale.shape
                ), f'shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}'  # noqa: E501
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        speculative_config = vllm_config.speculative_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.norm = Qwen3NextRMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj", "in_proj_qkvz", 0),
            ("in_proj", "in_proj_ba", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
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
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3NextForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP,
                           MixtureOfExperts, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"],
        "in_proj": ["in_proj_qkvz", "in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Qwen3Next currently does not support prefix caching"
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers: list[FusedMoE] = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3NextDecoderLayer)
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = (num_physical_experts -
                                      self.num_logical_experts)
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)

        return hidden_states

    def _get_mamba_cache_shape(
            self) -> tuple[tuple[int, int], tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = (self.config.mamba_num_heads *
                       self.config.hidden_size_per_head)
        conv_state_shape = (
            hidden_size // world_size,
            self.config.mamba_d_conv - 1,
        )
        temporal_state_shape = (
            hidden_size // world_size,
            self.config.mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

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
        hf_config = vllm_config.model_config.hf_config
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
            use_v1=True)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


