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
    vllmvis =  Qwen2_5_VisionTransformer(
        vision_config,
        norm_eps=1e-6,
    )
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    hfmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")#, device_map="auto")
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
    h = 112; w = 112

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

    h = 112; w = 112

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
    assert image_embeds[:16,:].sum().item() == (-791.0191040039062)
    assert image_embeds[16:,:].sum().item() == (-791.0191040039062)



'''
Same as test_2_sidebyside, but using util function
'''
def test_2_sidebyside_util(setup_fn):

    h = 112; w = 112

    visual = setup_fn.visual
    device = setup_fn.device

    print(f"[h, w]: {h, w}")
    pixel_values, grid_thw = gen_multi_image([h,h], [w,w])
    print(f"pixel shape: {pixel_values.shape} grid_thw: {grid_thw}")

    pixel_values = pixel_values.to(device)
    grid_thw = grid_thw.to(device)


    image_embeds = visual(pixel_values, grid_thw=grid_thw)
    assert image_embeds.sum().item() == (-1582.038330078125)
    assert image_embeds[:16,:].sum().item() == (-791.0191040039062)
    assert image_embeds[16:,:].sum().item() == (-791.0191040039062)


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
                sliced_tensor = torch.index_select(x, 0, torch.arange(start, end, device=x.device))
                outputs.append(sliced_tensor.sum().unsqueeze(0))            
            return torch.cat(outputs, dim=0)

    model = SliceModule()
    x = torch.arange(10)
    slices = torch.tensor([[0,5],[5,10]])
    print(model(x,slices))
    
    model = model.to('hpu')
    x = x.to('hpu')
    slices = slices.to('hpu')
    print(model(x,slices))
    print(model(x,torch.tensor([[0,3],[3,10]], device='hpu')))
    
    model = htorch.hpu.wrap_in_hpu_graph(model)
    print(model(x,slices))
    print(model(x,torch.tensor([[0,2],[2,10]], device='hpu')))
    
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
    x = torch.arange(10)*10
    rearrange = torch.tensor([9,8,7,6,1,2,3,4,0,5])
    print(model(x,rearrange))

    model = model.to('hpu')
    x = x.to('hpu')
    rearrange = rearrange.to('hpu')
    print(model(x,rearrange))
    print(model(x,torch.tensor([0,1,2,3,9,4,5,6,7,8], device='hpu')))
    model = htorch.hpu.wrap_in_hpu_graph(model)

    print(model(x,rearrange))
    print(model(x,torch.tensor([0,1,2,3,9,4,5,6,7,8], device='hpu')))
    y = x/10
    print(y)
    print(model(y,torch.tensor([0,1,2,3,9,4,5,6,7,8], device='hpu')))
    print(model(y,torch.tensor([0,0,0,0,0,0,0,0,0,0], device='hpu')))



def test_block():
    def create_block_diagonal_attention_mask(indices):
      max_index = indices[-1]
      attention_mask = torch.zeros((max_index, max_index))

      for i in range(len(indices)-1):
          start = indices[i]
          end = indices[i + 1]
          attention_mask[start:end, start:end] = 1

      return attention_mask 

    slices = torch.tensor([0,5,7])

    res = create_block_diagonal_attention_mask(slices)
    print(res)

    def create_block_diagonal_attention_mask_outerprod(indices):
        maxsize = indices[-1]
        range_to_max_for_each_img = torch.arange(maxsize).unsqueeze(0).repeat(indices.shape[0]-1,1)
        yy = range_to_max_for_each_img < indices[1:].unsqueeze(1)
        zz = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
        xx = torch.logical_and(yy, zz)
        # can reduce sum externally or as batchmatmul
        # res = torch.sum(torch.einsum('bi,bj->bij', xx, xx), dim=0)
        res = torch.einsum('bi,bj->ij', xx.float(), xx.float())
        return res
   
    res = create_block_diagonal_attention_mask_outerprod(torch.tensor([0,5,7,12]))
    print(res)


    
def test_block_static_indices():

    def create_block_diagonal_attention_mask_outerprod(indices):
        maxsize = indices[-1]
        range_to_max_for_each_img = torch.arange(maxsize).unsqueeze(0).repeat(indices.shape[0]-1,1)
        yy = range_to_max_for_each_img < indices[1:].unsqueeze(1)
        zz = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
        xx = torch.logical_and(yy, zz)
        # can reduce sum externally or as batchmatmul
        # res = torch.sum(torch.einsum('bi,bj->bij', xx, xx), dim=0)
        res = torch.einsum('bi,bj->ij', xx.float(), xx.float())
        return res

    def expand_to_max(indices, max_num_images):
        return torch.nn.functional.pad(indices, (0, max_num_images-indices.shape[0]), value=indices[-1])

    indices = torch.tensor([0,5,7,12])
    indices = expand_to_max(indices, 10)

    res = create_block_diagonal_attention_mask_outerprod(indices)
    print(res)

    indices = torch.tensor([0,5])
    indices = expand_to_max(indices, 10)

    res = create_block_diagonal_attention_mask_outerprod(indices)
    print(res)
    
def test_block_static_indices_hpugraph():
    def expand_to_max(indices, max_num_images):
        return torch.nn.functional.pad(indices, (0, max_num_images-indices.shape[0]), value=indices[-1])

    class CreateAttnMask(nn.Module):
        def __init__(self):
            super(CreateAttnMask, self).__init__()

        def forward(self, indices):
            maxsize = indices[-1]
            range_to_max_for_each_img = torch.arange(maxsize, device=indices.device).unsqueeze(0).repeat(indices.shape[0]-1,1)
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

    indices = torch.tensor([0,5,7], device='hpu')
    indices = expand_to_max(indices, 10)
    res = model(indices)
    print(res)

    indices = torch.tensor([0,7], device='hpu')
    indices = expand_to_max(indices, 10)
    res = model(indices)
    print(res)


    indices = torch.tensor([0,2,4,7], device='hpu')
    indices = expand_to_max(indices, 10)
    res = model(indices)
    print(res)


    
    
    
      
   


#if __name__ == "__main__":
#    main()
