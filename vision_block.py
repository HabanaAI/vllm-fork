import argparse
import habana_frameworks.torch as htorch
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import logging

from functools import partial
from typing import (Callable, Iterable, Optional, Set, Tuple)
from einops import rearrange, repeat
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from vllm.model_executor import set_random_seed
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import _Backend
import pytest
import habana_frameworks.torch.core as htcore

logger = logging.getLogger("vllm")
logger.setLevel(logging.ERROR)
logger.propagate = False

is_hpu = True


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1),
                         "... d two -> ... (d two)",
                         two=2)


def apply_rotary_emb_torch(x: torch.Tensor,
                           cos: torch.Tensor,
                           sin: torch.Tensor,
                           interleaved: bool = False) -> torch.Tensor:
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos +
            rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]
        ],
        dim=-1,
    )


def apply_rotary_pos_emb_vision(t: torch.Tensor,
                                freqs: torch.Tensor,
                                use_flash_attn=False) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    apply_rotary_emb = apply_rotary_emb_torch
    if use_flash_attn:
        from flash_attn.layers.rotary import apply_rotary_emb
    output = apply_rotary_emb(t_, cos, sin).type_as(t)
    return output


### COPY AND PASTE from qwen2_5_vl.py


class Qwen2_5_VisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.gate_proj = ReplicatedLinear(in_features,
                                          hidden_features,
                                          bias=bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.gate_proj")
        self.up_proj = ReplicatedLinear(in_features,
                                        hidden_features,
                                        bias=bias,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.up_proj")
        self.down_proj = ReplicatedLinear(hidden_features,
                                          in_features,
                                          bias=bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.down_proj")
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        x_gate, _ = self.gate_proj(x)
        x_gate = self.act_fn(x_gate)
        x_up, _ = self.up_proj(x)
        x_down, _ = self.down_proj(x_gate * x_up)
        return x_down


class Qwen2_5_VisionAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        # self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        # self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.tp_size = 1
        self.tp_rank = 0
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size)

        self.qkv = ReplicatedLinear(input_size=embed_dim,
                                    output_size=3 * projection_size,
                                    quant_config=quant_config,
                                    prefix=f"{prefix}.qkv")
        self.proj = ReplicatedLinear(input_size=projection_size,
                                     output_size=embed_dim,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.proj")

        # Detect attention implementation.
        self.attn_backend: _Backend = _Backend.TORCH_SDPA

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            qkv = tensor_model_parallel_all_gather(qkv)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(dist_utils.split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (seq_len, bs, self.num_attention_heads_per_partition,
                     self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        if rotary_pos_emb is not None:
            use_flash_attn = self.attn_backend == _Backend.FLASH_ATTN
            q = apply_rotary_pos_emb_vision(q,
                                            rotary_pos_emb,
                                            use_flash_attn=use_flash_attn)
            k = apply_rotary_pos_emb_vision(k,
                                            rotary_pos_emb,
                                            use_flash_attn=use_flash_attn)

        if self.attn_backend == _Backend.FLASH_ATTN:
            # from vllm_flash_attn.flash_attn_interface import (
            #   flash_attn_varlen_func)
            from flash_attn import flash_attn_varlen_func

            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            output = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen,
                                            dropout_p=0,
                                            causal=False)

            context_layer = rearrange(output,
                                      "(b s) ... -> b s ...",
                                      b=batch_size)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            # Execute attention entry by entry for speed & less VRAM.
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d")
                                 for x in [q_i, k_i, v_i])
                if is_hpu:
                    from habana_frameworks.torch.hpex.kernels import FusedSDPA
                    output_i = FusedSDPA.apply(q_i, k_i, v_i, None, 0.0)
                else:
                    output_i = F.scaled_dot_product_attention(q_i,
                                                              k_i,
                                                              v_i,
                                                              dropout_p=0.0)
                output_i = rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask

            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                       kv_seqlen=None)

            context_layer = xops.memory_efficient_attention_forward(
                q, k, v, attn_bias=attn_bias, p=0, scale=None)
        context_layer = rearrange(context_layer,
                                  "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(embed_dim=dim,
                                            num_heads=num_heads,
                                            projection_size=dim,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.attn")
        self.mlp = Qwen2_5_VisionMLP(dim,
                                     mlp_hidden_dim,
                                     act_fn=act_fn,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        #print(f"VisionBlock : x[{x.shape}], cu_seqlens[{cu_seqlens.shape}, rotary_pos_emb{[rotary_pos_emb.shape]}]")
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              hidden_size,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.ModuleList([
            ReplicatedLinear(self.hidden_size,
                             self.hidden_size,
                             bias=True,
                             quant_config=quant_config,
                             prefix=f"{prefix}.mlp.0"),
            nn.GELU(),
            ReplicatedLinear(self.hidden_size,
                             d_model,
                             bias=True,
                             quant_config=quant_config,
                             prefix=f"{prefix}.mlp.2"),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2_5_VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (self.theta**(torch.arange(
                0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
                                                / self.dim))
            seq = torch.arange(seqlen,
                               device=self.inv_freq.device,
                               dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        # args for get_window_index
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen2_5_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(depth)
        ])
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        #TODO:  h//2 w//2 -> DOESN"T WORK in PT_COMPILE_ONLY_MODE(warmup)
        #Shape or reshape() need to be predetermined, not depends on the tensor value itself.
        # w, h, t
        #grid_thw: 1024x1024 -> [[1,74,74]] -> reshape [37,2,37,2] -> flatten
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)  #[74,74]
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)  #[74,74]
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(
                0, 2, 1,
                3).flatten()  #[37,2,37,2] => [37,2,2,37] => flatten [5476]
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            #TODO: This arange, reshape based on the value of tensor can't work in WARMUP, or with TENSOR_CACHE
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
            index_padded = index_padded.reshape(grid_t, num_windows_h,
                                                vit_merger_window_size,
                                                num_windows_w,
                                                vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
                vit_merger_window_size)
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(
                0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        #print(f"VisionTransformer: x-[{x.shape}, grid_thw-[{grid_thw}]")
        # patchify
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        if is_hpu:
            # NOTE: unique_consecutive is a dynamic operation
            # we are using `remove_duplicates_cpu` instead
            def remove_duplicates_cpu(a):
                return [
                    a[i] for i in range(len(a)) if i == 0 or a[i - 1] != a[i]
                ]

            cu_window_seqlens = remove_duplicates_cpu(cu_window_seqlens)
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens,
                device=hidden_states.device,
                dtype=grid_thw.dtype
                if torch.jit.is_tracing() else torch.int32)

        else:
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens,
                device=hidden_states.device,
                dtype=grid_thw.dtype
                if torch.jit.is_tracing() else torch.int32)
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)

        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        hidden_states = hidden_states.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens_now,
                                rotary_pos_emb=rotary_pos_emb)

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def gen_multi_image(hs, ws):
    pxl = []
    grid = []
    for h, w in zip(hs, ws):
        pixel_values, image_grid_thw = generate_image(h, w)
        pxl += [pixel_values]
        grid += [image_grid_thw]
    return torch.cat(pxl, dim=0), torch.cat(grid, dim=0)


def gen_batch_image(hs, ws):
    pxl = []
    grid = []
    for h, w in zip(hs, ws):
        pixel_values, image_grid_thw = generate_image(h, w)
        pxl += [pixel_values.unsqueeze(0)]
        grid += [image_grid_thw.unsqueeze(0)]
    return torch.cat(pxl, dim=0), torch.cat(grid, dim=0)


def generate_image(h, w):
    #import PIL
    from vllm.assets.image import ImageAsset
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # image to pixel values
    image_processor = processor.image_processor

    image = ImageAsset("stop_sign").pil_image
    image = image.resize((w, h))

    preprocess_result = image_processor \
        .preprocess(images=image, return_tensors="pt") \
        .data
    pixel_values = preprocess_result["pixel_values"]
    image_grid_thw = preprocess_result["image_grid_thw"]

    print("pixel_values :", {pixel_values.shape})
    print("image_grid_thw :", {image_grid_thw})

    return pixel_values, image_grid_thw


def get_model():
    vision_config = Qwen2_5_VLVisionConfig(hidden_size=1280,
                                           out_hidden_size=2048,
                                           in_chans=3,
                                           spatial_patch_size=14,
                                           tokens_per_second=2)
    vllmvis = Qwen2_5_VisionTransformer(
        vision_config,
        norm_eps=1e-6,
    )
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    hfmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct")  #, device_map="auto")
    hfvis = hfmodel.visual
    vllmvis.load_state_dict(hfvis.state_dict())

    return vllmvis


def init_device():
    device = torch.device("hpu")
    torch.hpu.set_device(device)
    set_random_seed(0)
    return device


class Setup:

    def __init__(self):
        self.device = init_device()
        self.visual = get_model()
        self.visual.to(self.device)


# to load teh model only once for multiple tests
@pytest.fixture(scope="session")
def setup_fn():
    print('setup')
    yield Setup()
    print('teardown')


def test_main(setup_fn):
    #parser = argparse.ArgumentParser(description="Process get_window_index from image")
    #parser.add_argument('--width', type=int, default=112, help='Width of the image')
    #parser.add_argument('--height', type=int, default=112, help='Height of the image')

    #args = parser.parse_args()

    #h, w = args.height, args.width
    h = 112
    w = 112

    visual = setup_fn.visual
    device = setup_fn.device

    print(f"[h, w]: {h, w}")
    pixel_values, grid_thw = generate_image(h, w)
    print(f"pixel shape: {pixel_values.shape} grid_thw: {grid_thw}")

    pixel_values = pixel_values.to(device)
    grid_thw = grid_thw.to(device)

    image_embeds = visual(pixel_values, grid_thw=grid_thw)
    assert image_embeds.sum().item() == -791.01904296875


'''
Same as test_main, but image repeated twice
Because we pass teh same image side-by-side, we get same answer from model placed-side-by-side
'''


def test_2_sidebyside(setup_fn):

    h = 112
    w = 112

    visual = setup_fn.visual
    device = setup_fn.device

    print(f"[h, w]: {h, w}")
    pixel_values, grid_thw = generate_image(h, w)
    pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
    grid_thw = torch.cat([grid_thw, grid_thw], dim=0)
    print(f"pixel shape: {pixel_values.shape} grid_thw: {grid_thw}")

    pixel_values = pixel_values.to(device)
    grid_thw = grid_thw.to(device)

    image_embeds = visual(pixel_values, grid_thw=grid_thw)
    assert image_embeds.sum().item() == (-1582.038330078125)
    assert image_embeds[:16, :].sum().item() == (-791.0191040039062)
    assert image_embeds[16:, :].sum().item() == (-791.0191040039062)


'''
Same as test_2_sidebyside, but using util function
'''


def test_2_sidebyside_util(setup_fn):

    h = 112
    w = 112

    visual = setup_fn.visual
    device = setup_fn.device

    print(f"[h, w]: {h, w}")
    pixel_values, grid_thw = gen_multi_image([h, h], [w, w])
    print(f"pixel shape: {pixel_values.shape} grid_thw: {grid_thw}")

    pixel_values = pixel_values.to(device)
    grid_thw = grid_thw.to(device)

    image_embeds = visual(pixel_values, grid_thw=grid_thw)
    assert image_embeds.sum().item() == (-1582.038330078125)
    assert image_embeds[:16, :].sum().item() == (-791.0191040039062)
    assert image_embeds[16:, :].sum().item() == (-791.0191040039062)


'''
test slice
'''


def test_slice():

    class SliceModule(nn.Module):

        def __init__(self):
            super(SliceModule, self).__init__()

        def forward(self, x, slices):
            outputs = []
            for start, end in slices:
                #sliced_tensor = x[start:end]
                sliced_tensor = torch.index_select(
                    x, 0, torch.arange(start, end, device=x.device))
                outputs.append(sliced_tensor.sum().unsqueeze(0))
            return torch.cat(outputs, dim=0)

    model = SliceModule()
    x = torch.arange(10)
    slices = torch.tensor([[0, 5], [5, 10]])
    print(model(x, slices))

    model = model.to('hpu')
    x = x.to('hpu')
    slices = slices.to('hpu')
    print(model(x, slices))
    print(model(x, torch.tensor([[0, 3], [3, 10]], device='hpu')))

    model = htorch.hpu.wrap_in_hpu_graph(model)
    print(model(x, slices))
    print(model(x, torch.tensor([[0, 2], [2, 10]], device='hpu')))
    '''
    This test shows  x[start:end] or index_select with arange will not work with hpu graphs
    '''


'''
 x = x[rearrange]
 x = torch.gather(x, 0, rearrange)
 both work with hpu graph
'''


def test_rearrange():
    import habana_frameworks.torch as ht

    class RearrangeSliceModule(nn.Module):

        def __init__(self):
            super(RearrangeSliceModule, self).__init__()

        def forward(self, x, rearrange):
            #breakpoint()
            x = x[rearrange]
            #x = torch.gather(x, 0, rearrange)
            outputs = [x[:5].sum().unsqueeze(0), x[5:].sum().unsqueeze(0)]
            return torch.cat(outputs, dim=0)

    model = RearrangeSliceModule()
    x = torch.arange(10) * 10
    rearrange = torch.tensor([9, 8, 7, 6, 1, 2, 3, 4, 0, 5])
    print(model(x, rearrange))

    model = model.to('hpu')
    x = x.to('hpu')
    rearrange = rearrange.to('hpu')
    print(model(x, rearrange))
    print(model(x, torch.tensor([0, 1, 2, 3, 9, 4, 5, 6, 7, 8], device='hpu')))
    model = htorch.hpu.wrap_in_hpu_graph(model)

    print(model(x, rearrange))
    print(model(x, torch.tensor([0, 1, 2, 3, 9, 4, 5, 6, 7, 8], device='hpu')))
    y = x / 10
    print(y)
    print(model(y, torch.tensor([0, 1, 2, 3, 9, 4, 5, 6, 7, 8], device='hpu')))
    print(model(y, torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='hpu')))


def test_block():

    def create_block_diagonal_attention_mask(indices):
        max_index = indices[-1]
        attention_mask = torch.zeros((max_index, max_index))

        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            attention_mask[start:end, start:end] = 1

        return attention_mask

    slices = torch.tensor([0, 5, 7])

    res = create_block_diagonal_attention_mask(slices)
    print(res)

    def create_block_diagonal_attention_mask_outerprod(indices):
        maxsize = indices[-1]
        range_to_max_for_each_img = torch.arange(maxsize).unsqueeze(0).repeat(
            indices.shape[0] - 1, 1)
        yy = range_to_max_for_each_img < indices[1:].unsqueeze(1)
        zz = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
        xx = torch.logical_and(yy, zz)
        # can reduce sum externally or as batchmatmul
        # res = torch.sum(torch.einsum('bi,bj->bij', xx, xx), dim=0)
        res = torch.einsum('bi,bj->ij', xx.float(), xx.float())
        return res

    res = create_block_diagonal_attention_mask_outerprod(
        torch.tensor([0, 5, 7, 12]))
    print(res)


def test_block_static_indices():

    def create_block_diagonal_attention_mask_outerprod(indices):
        maxsize = indices[-1]
        range_to_max_for_each_img = torch.arange(maxsize).unsqueeze(0).repeat(
            indices.shape[0] - 1, 1)
        yy = range_to_max_for_each_img < indices[1:].unsqueeze(1)
        zz = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
        xx = torch.logical_and(yy, zz)
        # can reduce sum externally or as batchmatmul
        # res = torch.sum(torch.einsum('bi,bj->bij', xx, xx), dim=0)
        res = torch.einsum('bi,bj->ij', xx.float(), xx.float())
        return res

    def expand_to_max(indices, max_num_images):
        return torch.nn.functional.pad(indices,
                                       (0, max_num_images - indices.shape[0]),
                                       value=indices[-1])

    indices = torch.tensor([0, 5, 7, 12])
    indices = expand_to_max(indices, 10)

    res = create_block_diagonal_attention_mask_outerprod(indices)
    print(res)

    indices = torch.tensor([0, 5])
    indices = expand_to_max(indices, 10)

    res = create_block_diagonal_attention_mask_outerprod(indices)
    print(res)


def test_block_static_indices_hpugraph():

    def expand_to_max(indices, max_num_images):
        return torch.nn.functional.pad(indices,
                                       (0, max_num_images - indices.shape[0]),
                                       value=indices[-1])

    class CreateAttnMask(nn.Module):

        def __init__(self):
            super(CreateAttnMask, self).__init__()

        def forward(self, indices):
            maxsize = indices[-1]
            range_to_max_for_each_img = torch.arange(
                maxsize, device=indices.device).unsqueeze(0).repeat(
                    indices.shape[0] - 1, 1)
            yy = range_to_max_for_each_img < indices[1:].unsqueeze(1)
            zz = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
            xx = torch.logical_and(yy, zz)
            # can reduce sum externally or as batchmatmul
            # res = torch.sum(torch.einsum('bi,bj->bij', xx, xx), dim=0)
            res = torch.einsum('bi,bj->ij', xx.float(), xx.float())
            return res

    model = htorch.hpu.wrap_in_hpu_graph(CreateAttnMask().to('hpu'))

    # as long as the size of attn mask is same (7 here)
    # and max num images is same (10 here)
    # we should be fine

    indices = torch.tensor([0, 5, 7], device='hpu')
    indices = expand_to_max(indices, 10)
    res = model(indices)
    print(res)

    indices = torch.tensor([0, 7], device='hpu')
    indices = expand_to_max(indices, 10)
    res = model(indices)
    print(res)

    indices = torch.tensor([0, 2, 4, 7], device='hpu')
    indices = expand_to_max(indices, 10)
    res = model(indices)
    print(res)


def test_fsdpa():
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    dim = 1600
    q1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
    k1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
    v1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
    fullatt_block_attn_mask = torch.ones([dim, dim], device='hpu').bool()
    fused_out = FusedSDPA.apply(q1, k1, v1, fullatt_block_attn_mask, 0.0)
    print(fused_out.sum())



from habana_frameworks.torch.hpex.kernels import FusedSDPA
import habana_frameworks.torch.hpu as ht

def expand_to_max(indices, max_num_images):
    return torch.nn.functional.pad(indices,
                                   (0, max_num_images - indices.shape[0]),
                                   value=indices[-1])

def create_block_diagonal_attention_mask_outerprod(indices):
    maxsize = indices[
        -1]  # TODO using -1 here causes crashes.. using hardcoded 9, since for now I assume max num images = 10 ... maybe not revert to -1 when fusedsdpa with attn is resolved
    #print(indices.shape)
    #print(indices)
    #print(maxsize)

    range_to_max_for_each_img = torch.arange(
        maxsize,
        device=indices.device).unsqueeze(0).repeat(indices.shape[0] - 1, 1)
    yy = range_to_max_for_each_img < indices[1:].unsqueeze(1)
    zz = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
    xx = torch.logical_and(yy, zz)
    # can reduce sum externally or as batchmatmul
    res = torch.sum(torch.einsum('bi,bj->bij', xx, xx), dim=0)
    #breakpoint()
    #res = torch.einsum('bi,bj->ij', xx.float(), xx.float())
    return res.bool()

def test_fsdpa_shapechange():


    class FSDPAAttnMask(nn.Module):

        def __init__(self):
            super(FSDPAAttnMask, self).__init__()
            self.qkv = nn.Linear(1280, 1280 * 3)
            self.use_FusedSDPA = True

        def forward(self, x, cu_seqlens):
            new_shape = [x.shape[0], x.shape[1], 16, 80]
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=2)
            #breakpoint()
            q, k, v = (x.view(*new_shape) for x in (q, k, v)
                       )  # [seqlen, batchsz, num_head, head_dim]

            q1, k1, v1 = (rearrange(x, "s b ... -> b s ...").contiguous()
                          for x in (q, k, v))

            q1, k1, v1 = (rearrange(x, "b s h d -> b h s d")
                          for x in [q1, k1, v1])

            fullatt_block_attn_mask = create_block_diagonal_attention_mask_outerprod(
                cu_seqlens)

            (batch_size, n_heads, seq_len_N_t, head_dim_qk) = q1.shape
            (batch_size, n_heads, seq_len_N_s, head_dim_qk) = k1.shape

            mask_shape = (batch_size, 1, seq_len_N_t, seq_len_N_s)

            print(mask_shape)

            attn_mask = fullatt_block_attn_mask.reshape(
                batch_size, 1, seq_len_N_t, seq_len_N_s, -1)[:, :, :, :, 0]
            assert attn_mask.shape == mask_shape

            if self.use_FusedSDPA:
                fused_out = FusedSDPA.apply(q1, k1, v1, attn_mask, 0.0)
            else:
                fused_out = torch.nn.functional.scaled_dot_product_attention(
                    q1, k1, v1, attn_mask, 0.0)

            #q1, k1, v1 = (rearrange(x, "b s h d -> b h s d") for x in [q, k, v]) ##################### <<<<<
            # q1/k1/v1 shapes: 1600 16 1 80
            # fused_out = FusedSDPA.apply(q1, k1, v1, fullatt_block_attn_mask.bfloat16(), 0.0) #

            #breakpoint()
            '''
            N=1600, Hq=H=
            '''
            #fused_out = torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, fullatt_block_attn_mask, 0.0) #
            #torch.nn.functional.scaled_dot_product_attention(q, k, v, fullatt_block_attn_mask, 0.0)
            #torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, None, 0.0)
            #print(fused_out)

            context_layer = rearrange(fused_out, "b h s d -> b s h d ")
            return context_layer

    set_random_seed(0)
    dim = 1600
    device = 'hpu'
    cu_seqlens = expand_to_max(torch.tensor([0, dim], device=device), 8)
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    #breakpoint()
    model = FSDPAAttnMask().to(torch.bfloat16).to(device)
    y = model(x, cu_seqlens)
    print("fused", y.sum())

    model.use_FusedSDPA = False
    y = model(x, cu_seqlens)
    print("sdpa", y.sum())

    dim = 3200
    cu_seqlens = expand_to_max(torch.tensor([0, dim], device=device), 8)
    x = torch.randn([dim, 1, 1280], device=device)
    model.use_FusedSDPA = True
    y = model(x, cu_seqlens)
    print("fused", y.sum())

    model.use_FusedSDPA = False
    y = model(x, cu_seqlens)
    print("sdpa", y.sum())
    '''
    LOG_LEVEL_GC=2 ENABLE_CONSOLE=true pytest -s -v vision_block.py -k test_fsdpa_shapechange
    
    [2025-03-30 06:47:16.296] [COMPLEXGUID_LIB] [error] 'tpckernel.sdpa_recomp_core_fwd' op Broadcast of shape [1, 10] not possible at index 1. Dimension 10 incompatible with output shape dimension 1800Broadcast of shape [1, 10] not possible at index 1. Dimension 10 incompatible with output shape dimension 1800
[06:47:16.297048][SYN_RECIPE    ][error][tid:1F6C4] compileGraph: Can not compile graph
[06:47:16.297059][SYN_RECIPE    ][error][tid:1F6C4] addRecipeHandleAndCompileGraph: Can not compile
[06:47:16.297077][SYN_API       ][error][tid:1F6C4] compileGraph: Failed to add recipe handle into Recipe-Singleton status 26[synFail]
[06:47:16.297089][SYN_API       ][error][tid:1F6C4] synGraphCompile: SYN_API_CALL failed status: 26[synFail]. graphHandle 0x60dfefefb308
FAILED

    '''


class FSDPAAttnMaskOneshot(nn.Module):
    def __init__(self, use_fused, with_mask):
        super(FSDPAAttnMaskOneshot, self).__init__()
        self.qkv = nn.Linear(1280, 1280 * 3)
        self.use_FusedSDPA = use_fused
        self.with_mask = with_mask

    def forward(self, x, cu_seqlens):
        new_shape = [x.shape[0], x.shape[1], 16, 80]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)
        #breakpoint()
        q, k, v = (x.view(*new_shape) for x in (q, k, v)
                   )  # [seqlen, batchsz, num_head, head_dim]

        q1, k1, v1 = (rearrange(x, "s b ... -> b s ...").contiguous()
                      for x in (q, k, v))

        q1, k1, v1 = (rearrange(x, "b s h d -> b h s d")
                      for x in [q1, k1, v1])

        if self.with_mask:
            fullatt_block_attn_mask = create_block_diagonal_attention_mask_outerprod(
                cu_seqlens)

            (batch_size, n_heads, seq_len_N_t, head_dim_qk) = q1.shape
            (batch_size, n_heads, seq_len_N_s, head_dim_qk) = k1.shape

            mask_shape = (batch_size, 1, seq_len_N_t, seq_len_N_s)

            print(mask_shape)

            attn_mask = fullatt_block_attn_mask.reshape(
                batch_size, 1, seq_len_N_t, seq_len_N_s, -1)[:, :, :, :, 0]
            assert attn_mask.shape == mask_shape
        else:
            attn_mask = None

        if self.use_FusedSDPA:
            fused_out = FusedSDPA.apply(q1, k1, v1, attn_mask, 0.0)
        else:
            fused_out = torch.nn.functional.scaled_dot_product_attention(
                q1, k1, v1, fullatt_block_attn_mask, 0.0)

        context_layer = rearrange(fused_out, "b h s d -> b s h d ")
        return context_layer


def test_fsdpa_compare_fused_unfused():
    set_random_seed(0)
    dim = 1600
    device = 'hpu'
    cu_seqlens = expand_to_max(torch.tensor([0, dim//2, dim//2 + dim//4, dim], device=device), 8)
    assert cu_seqlens[-1].item() == dim
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    model = FSDPAAttnMaskOneshot(True, True).to(torch.bfloat16).to(device)
    weights = model.state_dict()
    y_fused = model(x, cu_seqlens).to('cpu')

    device = 'cpu'
    model = FSDPAAttnMaskOneshot(False, True)
    model.load_state_dict(weights)
    model = model.bfloat16().to(device)
    y_unfused = model(x, cu_seqlens)
    assert torch.allclose(y_fused, y_unfused, atol = 0.01)


class AttentionLongSequence:
    @staticmethod
    def forward(q, k, v, mask, q_block_size, step_after=None):
        """
        Support long sequence at prompt phase
        """
        q_len = q.size(-2)
        q_tiles = (q_len // q_block_size) if (q_len % q_block_size == 0) else math.ceil(q_len / q_block_size)
        q_padding = q_tiles * q_block_size - q_len
        q = F.pad(q, (0, 0, 0, q_padding), "constant", 0)
        if mask is not None:
            mask = F.pad(mask, (0, 0, 0, q_padding), "constant", -10000.0)
        attn_output = torch.zeros_like(q)

        for i in range(q_tiles):
            s, e = i * q_block_size, (i + 1) * q_block_size
            row_q = q[:, :, s:e, :]
            row_mask = mask[:, :, s:e, :]
            attn_output[:, :, s:e, :] = FusedSDPA.apply(row_q, k, v, row_mask, 0.0, False, None)
            if not None:
                if i % step_after == 0:
                    htcore.mark_step()

        if q_padding != 0:
            attn_output = attn_output[:, :, :-q_padding, :]

        return attn_output

class FSDPAAttnMaskChunked(nn.Module):
    def __init__(self, block_size):
        super(FSDPAAttnMaskChunked, self).__init__()
        self.qkv = nn.Linear(1280, 1280 * 3)
        self.block_size = block_size

    def forward(self, x, cu_seqlens):
        new_shape = [x.shape[0], x.shape[1], 16, 80]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        q, k, v = (x.view(*new_shape) for x in (q, k, v)
                   )  # [seqlen, batchsz, num_head, head_dim]

        q1, k1, v1 = (rearrange(x, "s b ... -> b s ...").contiguous()
                      for x in (q, k, v))

        q1, k1, v1 = (rearrange(x, "b s h d -> b h s d")
                      for x in [q1, k1, v1])

        fullatt_block_attn_mask = create_block_diagonal_attention_mask_outerprod(
            cu_seqlens)

        (batch_size, n_heads, seq_len_N_t, head_dim_qk) = q1.shape
        (batch_size, n_heads, seq_len_N_s, head_dim_qk) = k1.shape

        mask_shape = (batch_size, 1, seq_len_N_t, seq_len_N_s)

        print(mask_shape)

        attn_mask = fullatt_block_attn_mask.reshape(
            batch_size, 1, seq_len_N_t, seq_len_N_s, -1)[:, :, :, :, 0]
        assert attn_mask.shape == mask_shape

        assert seq_len_N_t == seq_len_N_s
        assert seq_len_N_t % self.block_size == 0

        fused_out = AttentionLongSequence.forward(q1, k1, v1, attn_mask, self.block_size)

        context_layer = rearrange(fused_out, "b h s d -> b s h d ")
        return context_layer



def test_fsdpa_chunked():
    set_random_seed(0)
    dim = 3072
    device = 'cpu'
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    cu_seqlens = expand_to_max(torch.tensor([0, dim//2, dim//2 + dim//4, dim], device=device), 8)
    assert cu_seqlens[-1].item() == dim
    model = FSDPAAttnMaskOneshot(False, True).to(torch.bfloat16).to(device)
    weights = model.state_dict()
    y_unfused_oneshot = model(x, cu_seqlens)
    
    device = 'hpu'
    model = FSDPAAttnMaskChunked(1024)
    model.load_state_dict(weights)
    model = model.to(torch.bfloat16).to(device)
    y_chunked = model(x.to(device), cu_seqlens.to(device))
    
    assert torch.allclose(y_chunked.to('cpu'), y_unfused_oneshot.to('cpu'), atol = 0.01)

def test_fsdpa_long_chunked():
    dim = 25600
    device='hpu'
    q1 = torch.randn([1, 16, dim, 80], device=device).bfloat16()
    k1 = torch.randn([1, 16, dim, 80], device=device).bfloat16()
    v1 = torch.randn([1, 16, dim, 80], device=device).bfloat16()
    attn_mask = torch.randn([1,1,dim,dim]) > 0.5
    fused_out = AttentionLongSequence.forward(q1, k1, v1, attn_mask, 64)
    print('fwd recorded')
    print(fused_out.sum())
    print('test done')


def test_fsdpa_long_chunked1():
    set_random_seed(0)
    dim = 25600
    device = 'cpu'
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    cu_seqlens = expand_to_max(torch.tensor([0, dim//2, dim//2 + dim//4, dim], device=device), 8)
    assert cu_seqlens[-1].item() == dim
    model = FSDPAAttnMaskOneshot(False, True).to(torch.bfloat16).to(device)
    weights = model.state_dict()
    y_unfused_oneshot = model(x, cu_seqlens)
    
    device = 'hpu'
    model = FSDPAAttnMaskChunked(1024)
    model.load_state_dict(weights)
    model = model.to(torch.bfloat16).to(device)
    y_chunked = model(x.to(device), cu_seqlens.to(device))
    
    assert torch.allclose(y_chunked.to('cpu'), y_unfused_oneshot.to('cpu'), atol = 0.01)
    # test failed: Synapse detected a device critical error that requires a restart.


class Slicer(nn.Module):
    def __init__(self):
        super(Slicer, self).__init__()
    def forward(self, x, slices):
        res = 1
        for i in range(1, len(slices)):
            start_idx = slices[i - 1]
            end_idx = slices[i]
            x_i = x[start_idx:end_idx]
            res *= x_i.sum()
        return res
            
# this test shows slice doesnt work with hpu graph as expected.
def test_sliceinhpugraph():
    model = Slicer().to('hpu')
    # tensor([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100], device='hpu:0')
    x = torch.arange(1, 11, device='hpu') * 10
    slices1 = torch.tensor([0, 5, 10], device='hpu')
    slices2 = torch.tensor([0, 2, 10], device='hpu')

    y1 = model(x, slices1) # 60000
    y2 = model(x, slices2) # 15600
    print(y1)
    print(y2)

    model = htorch.hpu.wrap_in_hpu_graph(model)

    y1 = model(x, slices1) # 60000
    y2 = model(x, slices2) # 60000 # incorrect
    print(y1)
    print(y2)


'''
input x will; always be aligned to 4
first we will chunk x in pieces of 4, run each piece thru self.lin and concat them back
Then depending on slices, we will slice out output of linear and normalize it
Finally we will concat it back again and return
'''
class SimulateBatchedFullAttn(nn.Module):
    def __init__(self):
        super(SimulateBatchedFullAttn, self).__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.arange(16).float().reshape([4, 4]))
        
    def forward(self, x, slices):
        x = torch.concat([self.lin(x[i*4:(i+1)*4]) for i in range(x.shape[0]//4)])
        lst = []
        for i in range(1, len(slices)):
            start_idx = slices[i - 1]
            end_idx = slices[i]
            x_i = x[start_idx:end_idx]
            lst.append(x_i/x_i.abs().sum())
        return torch.concat(lst)


class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
    def forward(self, x):
        return x/x.abs().sum(dim=-1, keepdim=True)


import copy

class AssignmentPerBucket:
    def __init__(self, key):
        self.max_bs = key[0]
        self.max_seq_len = key[1]
        self.assignment = [[]]
    def curr_slot_occupancy(self):
        return len(self.assignment[-1])
    def add(self, seq):
        if self.curr_slot_occupancy() == self.max_bs:
            self.assignment.append([])
        self.assignment[-1].append(seq)
    def __repr__(self):
        return str(self._canonical())
    def _canonical(self):
        return [i for i in self.assignment if len(i) > 0]
    def num_batches(self):
        return len(self._canonical())
    def __eq__(self, other):
        self._canonical() == other._canonical()
        

def get_timed_cost_fn(buckets):
  def compute_time(assignment):
      return sum(assignment[k].num_batches()*buckets[k] for k in assignment)
  return compute_time


def padding_cost(assignment):
    cost = 0
    for k in assignment:
        bs, max_len = k
        batches = assignment[k]._canonical()
        
        for b_idx, batch in enumerate(batches):
            cost += sum([max_len - i for i in batch])
            if b_idx == len(batches) - 1:
                cost += max_len * (bs - len(batch))

    return cost


def assign_images_to_batches(buckets, image_sizes, cost_fn):
    '''
    image_sizes_yet_to_be_assigned: a list of remaining images to assign to batches
    curr_batches: dict
      Key: tuple of len 2, first is bs, second is seq len
      Value: List of assignments
      For a key (2, 500) assignment could be something like: [[300, 200], [500, 400], [200]]. This means, first batch (bs=2) processes image of shape (300, 200), second one (500, 400) and the last one is still incomplete but it has 1 entry with image of size 200
      Clearly all numbers in the assignments must be <= the key[1]
    '''
    def helper(image_sizes_yet_to_be_assigned, curr_batches):
        if len(image_sizes_yet_to_be_assigned) == 0:
            return curr_batches

        min_time = 10000000000000
        best_assignment = None
        # Write it better by incorporating pruning. if ur current assignment is gonna be over best time till now, dont explore that branch
        for i_idx, img_size in enumerate(image_sizes_yet_to_be_assigned):
            for bkt in curr_batches:
                if img_size <= bkt[1]:
                    new_curr_batches = copy.deepcopy(curr_batches)
                    new_curr_batches[bkt].add(img_size)
                    new_image_sizes_yet_to_be_assigned = image_sizes_yet_to_be_assigned[:i_idx] + image_sizes_yet_to_be_assigned[i_idx+1:] # leave out i_idx
                    assignment_for_this_branch = helper(new_image_sizes_yet_to_be_assigned, new_curr_batches)
                    time_for_this_branch = cost_fn(assignment_for_this_branch) # do i need to compute time? doesnt the fn already return it?
                    if (time_for_this_branch <= min_time):
                        min_time = time_for_this_branch
                        best_assignment = assignment_for_this_branch
        return best_assignment
    x = helper(image_sizes, {k: AssignmentPerBucket(k) for k in buckets})
    return (x, cost_fn(x))


def test_assign_imgs_to_batches():
    buckets = {(1, 1000): 2, (4, 500): 2, (2,600): 1.2, (2, 500): 1}
    image_sizes = [320, 640, 400, 600, 800]
    x = assign_images_to_batches(buckets, image_sizes, get_timed_cost_fn(buckets))
    
    golden = ({(1, 1000): [[800], [640]], (4, 500): [], (2, 600): [[600]], (2, 500): [[400, 320]]}, 6.2)
    x_vals = {k:x[0][k]._canonical() for k in x[0]}
    assert golden == (x_vals, x[1]) 
    
    x1 = assign_images_to_batches(buckets, image_sizes, padding_cost)
    golden1 = 1240
    assert golden1 == x1[1] 



class SimulateBatchedFullAttn2(nn.Module):
    def __init__(self, batched, buckets):
        super(SimulateBatchedFullAttn2, self).__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.arange(16).float().reshape([4, 4]))
        self.norm = Norm()
        self.buckets = buckets  # this cannot be in the ctor in real case
        self.batched = batched
        
    def forward(self, x, slices):
        lst0 = [self.lin(x[i*4:(i+1)*4]) for i in range(x.shape[0]//4)]
        htcore.mark_step()
        x = torch.concat(lst0)
        lst1 = [x[slices[i - 1]:slices[i]] for i in range(1, len(slices))]
        htcore.mark_step()
        if self.batched:
            # slices always starts from 0, and assign_images_to_batches accepts inputs without the initial 0, hence the [1:] slicing
            img_sizes_cpu = list(torch.diff(slices.to('cpu').unique()).numpy())
            assignment_batches, _ = assign_images_to_batches(self.buckets, img_sizes_cpu, padding_cost) # how to keep track of a growing number of buckets, maybe we see a large image and we recompile during runtime?
            # lst1 has an ordering of images, say [320, 640, 400, 600, 800, 800]
            # assignment_batches is something like: {(1, 1000): [[800], [640], [800]], (4, 500): [], (2, 600): [[600]], (2, 500): [[400, 320]]}
            # we need a way to pull together images from lst1 into batches and them place them back into an output list again
            num_images = len(img_sizes_cpu)
            processed_already = [False] * num_images
            assignment = []
            for bkt in assignment_batches:
                for batch in assignment_batches[bkt]._canonical():
                    prep_batch = []
                    for seq_len in batch:
                        add_this_image = None
                        for image_idx, img_size in enumerate(img_sizes_cpu): # must be a better way to do this, gonna brute force it for now
                            if img_size == seq_len and not processed_already[image_idx]:
                                #print(f'setting {image_idx} to true')
                                processed_already[image_idx] = True
                                add_this_image = image_idx
                                break
                        prep_batch += [add_this_image]
                    assignment += [(prep_batch, bkt)]
            #assert sum(processed_already) == num_images
            #breakpoint()
            '''
            For example, for slices = tensor([ 0,  4,  8, 16, 16]), buckets = [(1, 16), (2, 4)]
            img_sizes_cpu = [4, 4, 8] (first 2 images of size 4, last one of size 8)
            assignment_batches = {(1, 16): [[8]], (2, 4): [[4, 4]]}
            ... the img of size 8 gets processed in bucket with bs=1, max len = 16, while the 2 images with sizes 4 fit in bs=2, maxsize=4 bucket
            we might get assignment = [([2], (1, 16)), ([0, 1], (2, 4))]
            ... img 2 (of size 8) go to (1,16), images 0 and 1 goes to bucket (2,4)
            '''
            #list_to_be_concat = [None] * num_images
            ret_tensor = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            for img_indices, (bs, max_len) in assignment:
                #if len(img_indices) != bs:
                #    breakpoint()
                #    print()
                pad_for_incomplete_batch = [] if len(img_indices) == bs else torch.zeros([bs-len(img_indices), max_len])
                gathered_batch = [torch.nn.functional.pad(lst1[i], (0, max_len-img_sizes_cpu[i])) for i in img_indices]
                gathered_batch = torch.concat([i.unsqueeze(0) for i in gathered_batch])
                if len(img_indices) != bs:
                    pad_empty_slots_in_batch = torch.zeros([bs-len(img_indices), max_len], device=gathered_batch.device, dtype=x.dtype)
                    gathered_batch = torch.concat([gathered_batch, pad_empty_slots_in_batch])
                htcore.mark_step()
                norm_out = self.norm(gathered_batch)
                #breakpoint()
                htcore.mark_step()
                # split the batch, and put back the outputs in the original location where it was taken from
                #for b_idx, orig_location in zip(range(norm_out.shape[0]), img_indices):
                #    list_to_be_concat[orig_location] = norm_out[b_idx][:img_sizes_cpu[orig_location]]
                for b_idx, orig_location in zip(range(norm_out.shape[0]), img_indices):
                    ret_tensor[slices[orig_location] : slices[orig_location+1]] = norm_out[b_idx][:img_sizes_cpu[orig_location]]
                #breakpoint()
                #print()
            #lst2 = list_to_be_concat
            htcore.mark_step()
            return ret_tensor
        else:
            lst2 = [self.norm(slc) for slc in lst1]
            htcore.mark_step()
            return torch.concat(lst2)
            
def test_simulate_window_and_full():
    model = SimulateBatchedFullAttn().bfloat16()
    x = torch.arange(16).bfloat16()
    slices = [torch.tensor([0, 16]), # 1 img
              torch.tensor([0, 4, 16]), torch.tensor([0, 8, 16]), torch.tensor([0, 12, 16]), # 2 images
              torch.tensor([0, 4, 8, 16]), torch.tensor([0, 8, 12, 16]), torch.tensor([0, 4, 12, 16]), # 3 images
              torch.tensor([0, 4, 8, 12, 16])] # 4 images
    cpu_res = [model(x, slc) for slc in slices]
    print(cpu_res)

    from habana_frameworks.torch.hpu.metrics import metric_global
    model = htorch.hpu.wrap_in_hpu_graph(model.to('hpu'))
    x_h = x.to('hpu')
    slices_h = [i.to('hpu') for i in slices]
    for slc in slices_h:
        htcore.mark_step(sync=True)
        gc_metric = metric_global("graph_compilation")
        print(gc_metric.stats())
        y = model(x_h, slc)
        print(y)
        htcore.mark_step(sync=True)
        gc_metric = metric_global("graph_compilation")
        print(gc_metric.stats())
        print('------------')
    '''
    1 img: recompiles. correct. expected
    2 imgs: recompiles for 1st 2 (why the second?). correct only for the first one
    3 imgs: recompiles for 1st 2 (why the second?). correct only for the first one
    4 imgs: recompiles. correct. expected
    '''
    
    
    model = SimulateBatchedFullAttn2(False, None).bfloat16()
    cpu_res_2 = [model(x, slc) for slc in slices]
    for i, j in zip(cpu_res, cpu_res_2):
        assert torch.allclose(i, j, atol=0.00001)
    # this proves SimulateBatchedFullAttn and SimulateBatchedFullAttn2(False) (unbatched) are the same

    model = SimulateBatchedFullAttn2(True, [(1, 16), (2, 4)]).bfloat16()
    cpu_res_3 = [model(x, expand_to_max(slc, 5)) for slc in slices]
    for idx, (i, j) in enumerate(zip(cpu_res, cpu_res_3)):
        assert torch.allclose(i, j, atol=0.00001)
    # This proves that the SimulateBatchedFullAttn and SimulateBatchedFullAttn2(True) (batched) are the same on CPU

    model = model.to('hpu')
    hpu_res = [model(x_h, expand_to_max(slc.to('hpu'), 5)) for slc in slices]
    for idx, (i, j) in enumerate(zip(cpu_res, hpu_res)):
        assert torch.allclose(i, j.to('cpu'), atol=0.005) # needs a bit loose tolerance for cpu vs hpu comparision
    # This proves that the SimulateBatchedFullAttn on CPU and SimulateBatchedFullAttn2(True) (batched) on HPU are the same


    model.lin = htorch.hpu.wrap_in_hpu_graph(model.lin)
    model.norm = htorch.hpu.wrap_in_hpu_graph(model.norm)

    hpu_graphs_res = [model(x_h, expand_to_max(slc.to('hpu'), 5)) for slc in slices]
    for idx, (i, j) in enumerate(zip(cpu_res, hpu_graphs_res)):
        assert torch.allclose(i, j.to('cpu'), atol=0.005) # needs a bit loose tolerance for cpu vs hpu comparision
    # This makes sure SimulateBatchedFullAttn2(True) works ok on hpu with hpugraphs


def test_crossoverpoint():
    def foo():
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        import time
        dims = [1600, 3200, 4800, 6400]
        fused_sdpa_times = {}
        for dim in dims:
            q1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
            k1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
            v1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
            fullatt_block_attn_mask = torch.ones([1, 1, dim, dim], device='hpu').bool()
            htcore.mark_step(sync=True)
            t0 = time.time()
            fused_out = FusedSDPA.apply(q1, k1, v1, fullatt_block_attn_mask, 0.0)
            print(fused_out.sum())
            t1 = time.time() - t0
            fused_sdpa_times[dim] = t1
            print(f'Done fused {dim}')
        htcore.mark_step(sync=True)

        # could have done in the last loop, but just to be safe doing it separately
        longprompt_mixtralstyle = {}
        for step in [25, 50, 75]:
            for block in [64, 128, 256]:
                for dim in dims:
                    if dim%block != 0:
                        continue
                    q1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
                    k1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
                    v1 = torch.rand([1, 16, dim, 80], device='hpu').bfloat16()
                    fullatt_block_attn_mask = torch.ones([1, 1, dim, dim], device='hpu').bool()
                    htcore.mark_step(sync=True)
                    t0 = time.time()
                    fused_out = AttentionLongSequence.forward(q1, k1, v1, fullatt_block_attn_mask, block, step)
                    print(fused_out.sum())
                    t1 = time.time() - t0
                    longprompt_mixtralstyle[(dim, step, block)] = t1
                    print(f'Done AttentionLongSequence {dim} step {step} block {block}')
        return fused_sdpa_times, longprompt_mixtralstyle
    foo() # warming up
    foo() # warming up
    fused_sdpa_times, longprompt_mixtralstyle = foo()
    print(fused_sdpa_times)
    print(longprompt_mixtralstyle)
    '''
    {1600: 0.0016040802001953125,
    3200: 0.0029897689819335938,
    4800: 0.005352973937988281,
    6400: 0.0066585540771484375}


    {(1600, 25, 64): 0.00471806526184082, <<<
    (1600, 50, 64): 0.004710674285888672 <<<
    (1600, 75, 64): 0.004685878753662109 <<<

    (3200, 25, 128): 0.005719423294067383 <<<
    (3200, 50, 64): 0.009069204330444336
    (3200, 50, 128): 0.005730867385864258 <<<
    (3200, 75, 64): 0.009095430374145508
    (3200, 25, 64): 0.008173942565917969,
    (3200, 75, 128): 0.005720376968383789 <<<

    (4800, 25, 64): 0.011794090270996094,  <<<
    (4800, 50, 64): 0.011925697326660156
    (4800, 75, 64): 0.014850854873657227

    (6400, 25, 64): 0.014657258987426758,
    (6400, 25, 128): 0.009445905685424805,
    (6400, 25, 256): 0.00830841064453125,   <<<
    (6400, 50, 64): 0.015633106231689453
    (6400, 50, 128): 0.011644601821899414
    (6400, 50, 256): 0.008297920227050781  <<<
    (6400, 75, 64): 0.015521526336669922
    (6400, 75, 128): 0.011669635772705078
    (6400, 75, 256): 0.008321523666381836}  <<<


    '''


#if __name__ == "__main__":
#    main()
