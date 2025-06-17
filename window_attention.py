# SPDX-License-Identifier: Apache-2.0
#import habana_frameworks.torch as htorch
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# import habana_frameworks.torch.core as htcore
# from habana_frameworks.torch.hpex.kernels import FusedSDPA

from einops import rearrange

# logger = logging.getLogger("vllm")
# logger.setLevel(logging.ERROR)
# logger.propagate = False

is_hpu = True


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query,
                                 key,
                                 value,
                                 attn_mask=None,
                                 dropout_p=0.0,
                                 is_causal=False,
                                 scale=None,
                                 enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    print("before softmax", attn_weight)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    print("after softmax", attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def pad_input_with_block_of_64(x, cu_seqlens):
    new_shape = [i for i in x.shape]
    new_shape[0] = (len(cu_seqlens) - 1) * 64
    padded_x = torch.zeros(*new_shape, device=x.device).bfloat16()
    seq_lens_deltas = cu_seqlens[1:] - cu_seqlens[:-1]
    padded_pointer = 0
    x_pointer = 0
    for i, delta in enumerate(seq_lens_deltas):
        padded_x[padded_pointer:padded_pointer + delta] = \
            x[x_pointer:x_pointer + delta]
        padded_pointer += 64
        x_pointer = cu_seqlens[i + 1]
    return padded_x


class FSDPAWindownAttnMask(nn.Module):

    def __init__(self, use_fused):
        super(FSDPAWindownAttnMask, self).__init__()
        self.qkv = nn.Linear(1280, 1280 * 3)
        self.use_FusedSDPA = use_fused
        self.softmax_mode = 'None'
        self.proj = nn.Linear(1280, 4 * 1280)

    def forward(self, x, cu_seqlens, attn_mask=None):
        new_shape = (x.shape[0], x.shape[1], 16, 80)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))

        # Window Attention computation
        outputs = []
        attn_mask_i = None
        for i in range(1, len(cu_seqlens)):
            start_idx = cu_seqlens[i - 1]
            end_idx = cu_seqlens[i]
            q_i = q[:, start_idx:end_idx]
            k_i = k[:, start_idx:end_idx]
            v_i = v[:, start_idx:end_idx]

            q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d")
                             for x in [q_i, k_i, v_i])
            print(
                f"index: {start_idx.item()} : {end_idx.item()} delta {end_idx.item() - start_idx.item()}"
            )
            print("qi, ki, vi shapes:", q_i.shape, k_i.shape, v_i.shape)

            if attn_mask is not None:
                attn_mask_i = attn_mask[i - 1, :, :]
                print("attn_mask", attn_mask_i.shape)

            if self.use_FusedSDPA:
                output_i = FusedSDPA.apply(q_i, k_i, v_i, attn_mask_i, 0.0)
            else:
                output_i = F.scaled_dot_product_attention(q_i,
                                                          k_i,
                                                          v_i,
                                                          attn_mask_i,
                                                          dropout_p=0.0)

            output_i = rearrange(output_i, "b h s d -> b s h d ")
            outputs.append(output_i)

        context_layer = torch.cat(outputs, dim=1)

        context_layer = rearrange(context_layer,
                                  "b s h d -> s b (h d)").contiguous()
        output = self.proj(context_layer)
        return output


def test_pad_input_with_block_of_64(device="cpu"):
    # cu_seqlens has blocks of 64, nothing to change
    dim = 256
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    cu_seqlens = [i for i in range(0, dim + 1, 64)]
    cu_seqlens = torch.tensor(cu_seqlens)
    padded_x = pad_input_with_block_of_64(x=x, cu_seqlens=cu_seqlens)
    assert x.shape == padded_x.shape
    assert torch.allclose(x, padded_x)

    # cu_seqlens needs to be padded to 256
    dim = 240
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    cu_seqlens = [0, 64, 128, 192, 240]
    cu_seqlens = torch.tensor(cu_seqlens)
    padded_x = pad_input_with_block_of_64(x=x, cu_seqlens=cu_seqlens)
    assert padded_x.shape[0] == 256
    assert torch.allclose(padded_x[:240, :], x[:240, :])
    assert torch.allclose(padded_x[241:, :],
                          torch.zeros_like(padded_x[241:, :]))


def test_pad_input_with_block_of_64_with_example():
    dim = 484
    device = "cpu"
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    cu_seqlens = [0, 64, 128, 176, 240, 304, 352, 400, 448, 484]
    cu_seqlens = torch.tensor(cu_seqlens)

    padded_x = pad_input_with_block_of_64(x=x, cu_seqlens=cu_seqlens)

    # same parts
    assert torch.allclose(x[0:176, :, :], padded_x[0:176])
    assert torch.allclose(x[176:304, :, :], padded_x[192:320])
    assert torch.allclose(x[304:352, :, :], padded_x[320:(320 + 48)])
    assert torch.allclose(x[352:400, :, :], padded_x[384:(384 + 48)])
    assert torch.allclose(x[400:448, :, :], padded_x[448:(448 + 48)])
    assert torch.allclose(x[448:484, :, :], padded_x[512:(512 + 36)])

    # expected zero sections
    padded_section = padded_x[176:176 + (64 - 48)]
    assert torch.allclose(padded_section, torch.zeros_like(padded_section))
    padded_section = padded_x[(320 + 48):(320 + 48) + (64 - 48)]
    assert torch.allclose(padded_section, torch.zeros_like(padded_section))
    padded_section = padded_x[(384 + 48):(384 + 48) + (64 - 48)]
    assert torch.allclose(padded_section, torch.zeros_like(padded_section))
    padded_section = padded_x[(448 + 48):(448 + 48) + (64 - 48)]
    assert torch.allclose(padded_section, torch.zeros_like(padded_section))
    padded_section = padded_x[(512 + 36):(512 + 48) + (64 - 36)]
    assert torch.allclose(padded_section, torch.zeros_like(padded_section))


def test_window_attention_mask_of_ones(use_fused=False):
    dim = 192
    device = 'cpu'
    atol = 0.001
    cu_seqlens = [0, 64, 128, 192]
    n = 64
    cu_seqlens = torch.tensor(cu_seqlens)

    assert cu_seqlens[-1].item() == dim
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    # no mask
    model = FSDPAWindownAttnMask(use_fused=use_fused).to(
        torch.bfloat16).to(device)
    weights = model.state_dict()
    y = model(x, cu_seqlens).to('cpu')

    # with mask and pad
    model = FSDPAWindownAttnMask(use_fused=use_fused).to(
        torch.bfloat16).to(device)
    model.load_state_dict(weights)
    model = model.bfloat16().to(device)

    attn_mask = torch.ones(len(cu_seqlens), n, n)
    if use_fused:
        attn_mask.to(torch.bfloat16)
    y_with_mask = model(x, cu_seqlens, attn_mask=attn_mask).to('cpu')

    assert torch.allclose(y, y_with_mask, atol=atol)


def test_window_attention_mask_with_gaps(use_fused=False):
    dim = 240
    device = 'cpu'
    atol = 0.001
    n = 64
    cu_seqlens = [0, 64, 128, 192, 240]
    cu_seqlens = torch.tensor(cu_seqlens)

    assert cu_seqlens[-1].item() == dim
    x = torch.randn([dim, 1, 1280], device=device).bfloat16()
    padded_x = pad_input_with_block_of_64(x, cu_seqlens)

    # no mask being called here
    model = FSDPAWindownAttnMask(use_fused=use_fused).to(
        torch.bfloat16).to(device)
    weights = model.state_dict()
    y = model(padded_x, cu_seqlens).to('cpu')

    # using mask
    model = FSDPAWindownAttnMask(use_fused=use_fused).to(
        torch.bfloat16).to(device)
    model.load_state_dict(weights)
    model = model.to(device)
    new_cu_seqlens = [i for i in range(0, len(cu_seqlens) * 64, 64)]
    new_cu_seqlens = torch.tensor(new_cu_seqlens)
    assert len(new_cu_seqlens) == len(cu_seqlens)
    assert padded_x.shape[0] == new_cu_seqlens[-1]

    attn_mask = torch.ones(len(new_cu_seqlens), n, n)
    vector_mask = torch.ones(64)
    vector_mask[48:] = 0
    attn_mask[-1, :, :] = torch.outer(vector_mask, vector_mask)
    attn_mask = attn_mask.bool()

    if use_fused:
        attn_mask.to(torch.bfloat16)
    y_with_mask = model(padded_x, new_cu_seqlens,
                        attn_mask=attn_mask).to('cpu')

    assert torch.allclose(y[:192], y_with_mask[:192], atol=atol)
    assert torch.allclose(y[192:], y_with_mask[192:192 + 48], atol=atol)


def test_scaled_dot_product_attention_masks_one():
    q_i = torch.rand(1, 16, 64, 80)
    k_i = torch.rand(1, 16, 64, 80)
    v_i = torch.rand(1, 16, 64, 80)
    attn_mask_i = torch.ones(64, 64).bool()
    output_no_mask = F.scaled_dot_product_attention(q_i,
                                                    k_i,
                                                    v_i,
                                                    dropout_p=0.0)

    output_mask = F.scaled_dot_product_attention(q_i,
                                                 k_i,
                                                 v_i,
                                                 attn_mask_i,
                                                 dropout_p=0.0)

    assert torch.allclose(output_no_mask, output_mask, atol=0.001)


def test_scaled_dot_product_attention():
    n = 64
    q_i = torch.rand(1, 2, n, 3)
    k_i = torch.rand(1, 2, n, 3)
    v_i = torch.rand(1, 2, n, 3)
    debug = True

    q_i[:, :, -1, :] = float("-inf")
    k_i[:, :, -1, :] = float("-inf")

    attn_mask_i = torch.ones(n, n)
    attn_mask_i[-1, :] = 0
    attn_mask_i[:, -1] = 0

    attn_mask_i = attn_mask_i.bool()

    if debug:
        print(" no mask ")
        output_no_mask = scaled_dot_product_attention(q_i,
                                                      k_i,
                                                      v_i,
                                                      dropout_p=0.0)

        print(" using mask ")
        output_mask = scaled_dot_product_attention(q_i,
                                                   k_i,
                                                   v_i,
                                                   attn_mask_i,
                                                   dropout_p=0.0)

    output_no_mask = F.scaled_dot_product_attention(q_i,
                                                    k_i,
                                                    v_i,
                                                    dropout_p=0.0)
    output_mask = F.scaled_dot_product_attention(q_i,
                                                 k_i,
                                                 v_i,
                                                 attn_mask_i,
                                                 dropout_p=0.0)

    clean_output_no_mask = output_no_mask.ravel()[output_no_mask.ravel() > 0]
    clean_output_mask = output_mask.ravel()[output_mask.ravel() > 0]
    assert torch.allclose(clean_output_no_mask, clean_output_mask)


# test_pad_input_with_block_of_64(device="cpu")
# test_window_attention_mask_of_ones(use_fused=False)

test_scaled_dot_product_attention()

#test_window_attention_mask_with_gaps(use_fused=False)
# test_window_attention_cpu_not_implemented()

# [h, w]: (308, 308)
# pixel_values : {torch.Size([484, 1176])}
# image_grid_thw : {tensor([[ 1, 22, 22]])}
# pixel shape: torch.Size([484, 1176]) grid_thw: tensor([[ 1, 22, 22]])
# window attention
# forward: x torch.Size([484, 1, 1280])
# cu_seqlens:  [  0,  64, 128, 176, 240, 304, 352, 400, 448, 484]
# q k v shapes:
# torch.Size([484, 1, 16, 80])
# torch.Size([484, 1, 16, 80])
# torch.Size([484, 1, 16, 80])
