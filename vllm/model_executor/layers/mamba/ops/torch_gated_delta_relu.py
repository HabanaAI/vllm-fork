# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/models/qwen3_next/modeling_qwen3_next.py

import torch
import torch.nn.functional as F


def torch_chunk_gated_delta_rule_opt(
    query,
    key,
    value,
    g,
    beta,
    eye_constant,
    valid_seq_len=None,
    chunk_size=64,
    inv_loop=12,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    ssm_dtype = g.dtype
    if use_qk_l2norm_in_kernel:
        head_dim = query.size(-1)
        inv_scale = head_dim**-0.5
        query = F.rms_norm(query, (head_dim, ), eps=1e-6) * inv_scale
        key = F.rms_norm(key, (head_dim, ), eps=1e-6) * inv_scale
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    if valid_seq_len is None:
        valid_seq_len = torch.full((batch_size, ),
                                   sequence_length,
                                   dtype=torch.long,
                                   device=key.device)
    else:
        valid_seq_len = valid_seq_len.to(device=key.device, dtype=torch.long)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    tot_len = sequence_length + pad_size

    token_idx = torch.arange(tot_len, device=key.device).view(1, 1, tot_len)
    valid_mask = (token_idx < valid_seq_len.view(batch_size, 1,
                                                 1)).to(value.dtype)

    query = query * valid_mask.unsqueeze(-1)
    key = key * valid_mask.unsqueeze(-1)
    value = value * valid_mask.unsqueeze(-1)
    beta = beta * valid_mask
    g = g * valid_mask

    valid_chunk_cnt = torch.div(valid_seq_len + chunk_size - 1,
                                chunk_size,
                                rounding_mode='floor')

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
    valid_mask = valid_mask.reshape(batch_size, 1, -1, chunk_size)

    chunk_idx = torch.arange(tot_len // chunk_size,
                             device=key.device).view(1, 1, -1)
    chunk_valid = (chunk_idx < valid_chunk_cnt.view(batch_size, 1,
                                                    1)).to(value.dtype)
    chunk_valid_state = chunk_valid.unsqueeze(-1).unsqueeze(-1)

    mask = torch.ones(chunk_size,
                      chunk_size,
                      dtype=value.dtype,
                      device=value.device).tril(-1)

    # chunk decay
    g = g.cumsum(dim=-1)
    g_exp = g.exp().to(value.dtype)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).exp().to(
        value.dtype)).tril()

    attn = torch.matmul(k_beta,
                        key.transpose(-1, -2).contiguous()) * \
           decay_mask * mask + eye_constant
    inv_attn = torch.zeros_like(attn) + eye_constant
    for _ in range(inv_loop):
        prod = torch.matmul(attn, inv_attn)
        err = prod * mask
        update = torch.matmul(inv_attn, err)
        inv_attn.sub_(update)
    attn = inv_attn

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_exp.unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim,
                                        v_head_dim).to(value) if initial_state
                            is None else initial_state.to(value))
    mask = torch.tril(torch.ones(chunk_size,
                                 chunk_size,
                                 dtype=value.dtype,
                                 device=value.device),
                      diagonal=0)
    attn = torch.matmul(query,
                        key.transpose(-1, -2).contiguous()) * decay_mask * mask
    qg = query * g_exp[..., None]
    delta_g_exp = (g[:, :, :, -1, None] - g).exp()[..., None].to(value.dtype)
    k_term = key * delta_g_exp

    num_chunks = tot_len // chunk_size
    k_eye = torch.eye(k_head_dim, dtype=value.dtype, device=value.device)
    k_eye = k_eye.view(1, 1, 1, k_head_dim, k_head_dim)

    alpha = g_exp[:, :, :, -1, None, None]
    B = k_term.transpose(-1, -2).contiguous()
    K = k_cumdecay
    V = value
    Q = qg
    A = attn

    M = alpha * k_eye - torch.matmul(B, K)
    N = torch.matmul(B, V)
    C = Q - torch.matmul(A, K)
    core_attn_out = torch.matmul(A, V)

    M = M * chunk_valid_state + k_eye * (1 - chunk_valid_state)
    N = N * chunk_valid_state
    C = C * valid_mask.unsqueeze(-1)
    core_attn_out = core_attn_out * valid_mask.unsqueeze(-1)

    # for each chunk
    for i in range(num_chunks):
        core_attn_out[:, :,
                      i].add_(torch.matmul(C[:, :, i], last_recurrent_state))
        last_recurrent_state = torch.matmul(M[:, :, i],
                                            last_recurrent_state) + N[:, :, i]

    if not output_final_state:
        last_recurrent_state = None
    else:
        last_recurrent_state = last_recurrent_state.to(ssm_dtype)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                          core_attn_out.shape[1], -1,
                                          core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule_opt(
    query,
    key,
    value,
    g,
    beta,
    recurrent_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    ssm_dtype = g.dtype
    if use_qk_l2norm_in_kernel:
        head_dim = query.size(-1)
        inv_scale = head_dim**-0.5
        query = F.rms_norm(query, (head_dim, ), eps=1e-6) * inv_scale
        key = F.rms_norm(key, (head_dim, ), eps=1e-6) * inv_scale
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)
    ]

    scale = 1 / (query.shape[-1]**0.5)
    query = query * scale

    recurrent_state = recurrent_state.to(value.dtype)

    q_t = query.squeeze(-2)
    k_t = key.squeeze(-2)
    v_t = value.squeeze(-2)
    g_t = g.squeeze(-1).exp().to(value.dtype).unsqueeze(-1).unsqueeze(-1)

    recurrent_state = recurrent_state * g_t
    kv_mem = torch.matmul(k_t.unsqueeze(-2), recurrent_state).squeeze(-2)
    delta = (v_t - kv_mem) * beta
    recurrent_state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2))
    core_attn_out = torch.matmul(q_t.unsqueeze(-2), recurrent_state)

    if not output_final_state:
        recurrent_state = None
    else:
        recurrent_state = recurrent_state.to(ssm_dtype)
    core_attn_out = core_attn_out.transpose(1, 2)
    return core_attn_out, recurrent_state


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    eye_constant,
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

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    tot_len = sequence_length + pad_size
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
                                 device=value.device),
                      diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    g_exp = g.exp()
    decay_mask = ((g.unsqueeze(-1) -
                   g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((torch.matmul(k_beta.contiguous(),
                           key.transpose(-1, -2).contiguous())) *
             decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].contiguous()
        sub = attn[..., :i, :]
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)[..., :i]
    attn = attn + eye_constant
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_exp.unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim,
                                        v_head_dim).to(value) if initial_state
                            is None else initial_state.to(value))
    core_attn_out = torch.zeros_like(value)
    mask = torch.tril(torch.ones(chunk_size,
                                 chunk_size,
                                 dtype=torch.bool,
                                 device=value.device),
                      diagonal=0)
    mask = mask.view(1, 1, 1, chunk_size, chunk_size)
    attn = (query @ key.transpose(-1, -2)) * decay_mask * mask
    qg = query * g_exp[..., None]
    delta_g_exp = (g[:, :, :, -1, None] - g).exp()[..., None]
    k_term = (key * delta_g_exp)

    # for each chunk
    for i in range(0, tot_len // chunk_size):
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = value[:, :, i] - v_prime
        attn_inter = qg[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn[:, :, i] @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_exp[:, :, i, -1, None, None] +
            k_term[:, :, i].transpose(-1, -2) @ v_new)

    if not output_final_state:
        last_recurrent_state = None
    else:
        last_recurrent_state = last_recurrent_state.to(initial_dtype)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                          core_attn_out.shape[1], -1,
                                          core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype)
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

    scale = 1 / (query.shape[-1]**0.5)
    query = query * scale

    recurrent_state = recurrent_state.to(value)

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
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype)
    return core_attn_out, recurrent_state
