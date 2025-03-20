import math
import pytest
import torch
import random
from collections import deque
from itertools import accumulate
from vllm.attention.backends.torch_paged_attention import (
    nearset_multiple,
    calculate_q_chunk_indices,
    convert_right_pad_to_left_pad,
    compute_tokens_per_block,
    compute_block_id_for_token_idx,
    compute_per_token_offset_in_block,
    gather_kv_tokens,
    gather_q_slice_from_tensor,
    paged_attention_var_len,
)


# Device under test. One of "cpu", "cuda", "hpu"
DEVICE_TYPE = "cuda"

if DEVICE_TYPE == "hpu":
    import habana_frameworks
    import habana_frameworks.torch.core as htcore

DEVICE = torch.device(DEVICE_TYPE)


def _get_qkv_tensors(batch_size, q_sizes, kv_sizes, nh, d, use_int=False):
    def _get_tensor(sizes):
        max_size = max(sizes)
        if use_int:
            t = torch.arange(1, max_size + 1).float().view(1,-1,1,1).repeat(batch_size, 1, nh, d)
        else:
            t = torch.rand((batch_size, max_size, nh, d), dtype=torch.float)
        mask = torch.arange(max_size).unsqueeze(0) < torch.tensor(sizes).unsqueeze(1)
        return t * mask.view(batch_size, max_size, 1, 1)

    q = _get_tensor(q_sizes) if q_sizes is not None else None
    k = _get_tensor(kv_sizes) if kv_sizes is not None else None
    v = _get_tensor(kv_sizes) if kv_sizes is not None else None
    return q, k, v


def _copy_kv_to_blocks(num_blocks, block_size, k, v, kv_sizes, max_blocks_per_sample, rand_blocks=False):
    assert k.shape == v.shape
    batch_size, _, nh, d = k.shape
    k_cache = torch.zeros((num_blocks, block_size, nh, d), dtype=torch.float)
    v_cache = torch.zeros((num_blocks, block_size, nh, d), dtype=torch.float)
    blocks_pool = deque(range(1, num_blocks+1))
    if rand_blocks:
        random.shuffle(blocks_pool)
    block_table = torch.zeros((batch_size, max_blocks_per_sample), dtype=torch.long)
    for sample_idx, n_tokens in enumerate(kv_sizes):
        n_blocks = int(math.ceil(n_tokens / block_size))
        assert n_blocks <= max_blocks_per_sample
        for j in range(n_blocks):
            start_token = j * block_size
            end_token = min(start_token + block_size, n_tokens)
            tokens_in_block = end_token - start_token
            block_no = blocks_pool.popleft()
            k_cache[block_no, :tokens_in_block, :, :] = k[sample_idx, start_token:end_token, :, :]
            v_cache[block_no, :tokens_in_block, :, :] = v[sample_idx, start_token:end_token, :, :]
            block_table[sample_idx, j] = block_no
    return block_table, k_cache, v_cache


def _create_q_1d(q, q_sizes, nh, d):
    n_total_tokens = sum(q_sizes)
    q_1d = torch.zeros((n_total_tokens, nh, d), dtype=torch.float)
    offset = 0
    for sample_idx, n_tokens in enumerate(q_sizes):
        q_1d[offset:offset+n_tokens, :, :] = q[sample_idx, :n_tokens, :, :]
        offset += n_tokens
    return q_1d


def build_pa_inputs(
        batch, nh, d, q_sizes, kv_sizes, use_int, num_blocks, block_size, max_blocks_per_sample, rand_blocks
):
    """ Returns block_table, k_cache, v_cache, q_1d """
    q, k, v = _get_qkv_tensors(batch, q_sizes, kv_sizes, nh, d, use_int=use_int)
    block_table, k_cache, v_cache = _copy_kv_to_blocks(
        num_blocks,
        block_size,
        k,
        v,
        kv_sizes,
        max_blocks_per_sample,
        rand_blocks
    ) if kv_sizes is not None else (None, None, None)
    q_1d = _create_q_1d(q, q_sizes, nh, d) if q_sizes is not None else None
    return block_table, k_cache, v_cache, q_1d


@pytest.mark.parametrize("max_blocks, n_blocks_used", [
    (5, [1]),
    (5, [1, 2]),
    (5, [1, 2, 3])
])
def test_pa_convert_right_pad_to_left_pad(max_blocks, n_blocks_used):
    batch = len(n_blocks_used)
    right_padded_block_table = torch.zeros((batch, max_blocks), dtype=torch.long)
    left_padded_block_table = torch.zeros((batch, max_blocks), dtype=torch.long)
    for i in range(batch):
        right_padded_block_table[i, :n_blocks_used[i]] = torch.arange(1, n_blocks_used[i] + 1)
        left_padded_block_table[i, max_blocks - n_blocks_used[i]:] = torch.arange(1, n_blocks_used[i] + 1)
    seq_used = torch.tensor(n_blocks_used, dtype=torch.long)
    res = convert_right_pad_to_left_pad(
        right_padded_block_table.to(DEVICE),
        seq_used.to(DEVICE),
        block_size=1)
    res = res.to('cpu')
    assert res.eq(left_padded_block_table).all().item()


@pytest.mark.parametrize("seq, block_size, max_blocks", [
    ([15, 7, 20], 5, 6),
    ([1, 8, 19], 4, 6),
])
def test_pa_compute_tokens_per_block(seq, block_size, max_blocks):
    batch = len(seq)
    t_seq = torch.tensor(seq, dtype=torch.long)
    expected = torch.zeros((batch, max_blocks), dtype=torch.long)
    for i in range(batch):
        for j in range(max_blocks-1, -1, -1):
            if seq[i] == 0:
                count = 0
            elif seq[i] % block_size > 0:
                count = seq[i] % block_size
            else:
                count = block_size
            seq[i] -= count
            expected[i][j] = count
    res = compute_tokens_per_block(t_seq.to(DEVICE), block_size, max_blocks)
    res = res.to('cpu')
    assert torch.equal(res, expected)


@pytest.mark.parametrize("m, seq, block_size, max_blocks", [
    (9, [9, 8, 2],  3, 4),
    (9, [9, 8, 2],  3, 3),
    (9, [9, 9, 9],  3, 3),
])
def test_pa_compute_block_id_for_token_idx(m, seq, block_size, max_blocks):
    batch = len(seq)
    t_seq = torch.tensor(seq, dtype=torch.long)
    t_tokens_per_block = compute_tokens_per_block(t_seq, block_size, max_blocks)
    t_block_table = torch.zeros((batch, max_blocks), dtype=torch.long)
    for i in range(batch):
        t_block_table[i, :] = torch.arange(i * max_blocks, (i+1) * max_blocks)
    t_block_table = convert_right_pad_to_left_pad(t_block_table, t_seq, block_size)
    res = compute_block_id_for_token_idx(
        m,
        t_block_table.to(DEVICE),
        t_tokens_per_block.to(DEVICE)
    )
    expected = torch.full((batch, m), -1, dtype=torch.long)
    for i in range(batch):
        block_index = max_blocks - 1
        end_token_index = m
        while seq[i] > 0:
            remainder = seq[i] % block_size
            n_tokens_in_block = remainder if remainder > 0 else block_size
            start_token_index = end_token_index - n_tokens_in_block
            expected[i, start_token_index: end_token_index] = t_block_table[i, block_index]
            seq[i] -= n_tokens_in_block
            block_index -= 1
            end_token_index = start_token_index
    res = res.to('cpu')
    assert torch.equal(res, expected)


def test_pa_gather_kv_tokens():
    batch = 3
    nh = 1
    d = 1
    num_blocks = 10
    block_size = 4
    max_blocks_per_sample = 3
    seq_used_k = torch.tensor([2, 7, 12], dtype=torch.long)

    block_table, k_cache, v_cache, q_1d = build_pa_inputs(
        batch=batch,
        nh=nh,
        d=d,
        q_sizes=None,
        kv_sizes=seq_used_k.tolist(),
        use_int=True,
        num_blocks=num_blocks,
        block_size=block_size,
        max_blocks_per_sample=max_blocks_per_sample,
        rand_blocks=False
    )

    chunk_start = 3
    chunk_size = 7

    # prepare for gather_kv_tokens
    m = seq_used_k.max()
    block_table = convert_right_pad_to_left_pad(block_table, seq_used_k, block_size)
    m_padded = nearset_multiple(m, chunk_size)
    tokens_per_block = compute_tokens_per_block(seq_used_k, block_size, max_blocks_per_sample)
    block_id_for_token_idx = compute_block_id_for_token_idx(m_padded, block_table, tokens_per_block)
    offset_in_block_for_token_idx = compute_per_token_offset_in_block(block_id_for_token_idx)

    # activate
    k_chunk, v_chunk, pad_mask = gather_kv_tokens(
        k_cache.to(DEVICE),
        v_cache.to(DEVICE),
        block_id_for_token_idx.to(DEVICE),
        offset_in_block_for_token_idx.to(DEVICE),
        chunk_start,
        chunk_size
    )
    assert k_chunk.shape == (batch, chunk_size, nh, d)
    assert v_chunk.shape == (batch, chunk_size, nh, d)

    # expected
    expected_vals = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 2, 3],
        [2, 3, 4, 5, 6, 7, 8],
    ]
    expected = torch.tensor(expected_vals, dtype=torch.float).view(batch, chunk_size, nh, d)
    assert expected.eq(k_chunk.to('cpu')).all().item()


@pytest.mark.parametrize("seqlen, chunk_size", [
    ([3, 7, 11], 3),
    ([3, 7, 11], 4),
    ([3, 7, 11], 5),
])
def test_pa_gather_q_tokens(seqlen, chunk_size):
    # create ref tensors
    batch = len(seqlen)
    q_list = []
    start_val = 1
    for i in range(batch):
        end_val = start_val + seqlen[i]
        q_list.append(torch.arange(start_val, end_val, dtype=torch.long))
        start_val = end_val

    # left-pad q_list tensors to max_q_size rounded to nearset chunk size
    max_seqlen = max(seqlen)
    max_seqlen = nearset_multiple(max_seqlen, chunk_size)
    q_list_padded = [q.clone() for q in q_list]
    for i in range(batch):
        q_list_padded[i] = torch.nn.functional.pad(q_list_padded[i], (max_seqlen-seqlen[i], 0)).unsqueeze(0)
    q_ref = torch.cat(q_list_padded, dim=0)
    q_ref = q_ref.view(batch, max_seqlen, 1, 1)  # (batch, max_seqlen, nh, d)

    # prepare required tensors: q_1d, cu_seqlen
    q_1d = torch.cat(q_list, dim=0)
    q_1d = q_1d.view(-1, 1, 1)  # (total_len, nh, d)
    cu_seqlen = list(accumulate(seqlen))
    cu_seqlen = torch.tensor(cu_seqlen, dtype=torch.long)

    # compare to ref for all chunks
    for chunk_start in range(0, max_seqlen, chunk_size):
        chunk_i_idx, chunk_i_pad_mask = calculate_q_chunk_indices(
            cu_seqlen.to(DEVICE),
            max_seqlen=max_seqlen,
            chunk_start=chunk_start,
            chunk_size=chunk_size)
        chunk = gather_q_slice_from_tensor(q_1d.to(DEVICE), chunk_i_idx, chunk_i_pad_mask, pad_val=0.)
        chunk_ref = q_ref[:, chunk_start:chunk_start+chunk_size]
        assert chunk.to('cpu').eq(chunk_ref).all().item()


def pytorch_attention_ref(q, k, v, causal=False, q_sizes=None, kv_sizes=None):
    b, n, h, d = q.shape
    _, m, h, _ = k.shape

    assert k.shape == (b, m, h, d)
    assert m == n or (q_sizes is not None and kv_sizes is not None)

    # for prefill we assume that q and kv num tokens for each sample are the same
    # for decode/chunked-prefill, num q tokens can be smaller than kv - need to handle
    if q_sizes is not None:
        assert b == len(q_sizes)
        assert b == len(kv_sizes)

        # for each q sample, copy tokens from start to end of tensor
        q_padded = torch.zeros_like(k)
        for i in range(b):
            q_size = q_sizes[i]
            q_padded[i][-q_size:] = q[i][:q_size]
        q = q_padded

        # same for kv tensors
        k_padded = torch.zeros_like(k)
        v_padded = torch.zeros_like(v)
        for i in range(b):
            kv_size = kv_sizes[i]
            k_padded[i][-kv_size:] = k[i][:kv_size]
            v_padded[i][-kv_size:] = v[i][:kv_size]
        k, v = k_padded, v_padded

    q = q.transpose(1, 2)    # (b, h, n, d)
    k = k.transpose(1, 2)    # (b, h, n, d)
    v = v.transpose(1, 2)    # (b, h, n, d)

    attn_scores = torch.einsum('bhnd,bhmd->bhnm', q, k) / d ** 0.5

    if q_sizes is not None:
        # mask attention scores with padded k tokens
        for i in range(b):
            kv_size = kv_sizes[i]
            attn_scores[i, :, :, :-kv_size] = float("-inf")

    if causal:
        mask = torch.triu(torch.ones(m, m, device=q.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_probs = torch.softmax(attn_scores, dim=-1)
    res = torch.einsum('bhnm,bhmd->bhnd', attn_probs, v)

    res = res.transpose(1, 2)

    if q_sizes is not None:
        # for each q sample, copy back tokens from end to start of tensor
        res_final = torch.zeros((b, n, h, d), dtype=q.dtype, device=q.device)
        for i in range(b):
            q_size = q_sizes[i]
            res_final[i][:q_size] = res[i][-q_size:]
        res = res_final
    else:
        res = res[:, -n:]

    return res


@pytest.mark.parametrize("q_sizes, kv_sizes, max_chunk_size, max_block_size", [
    ([3, 7], [3, 7], 8, 8),
    ([1, 6], [3, 7], 8, 8),
    ([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], 6, 6)
])
def test_pa_exhaustive(q_sizes, kv_sizes, max_chunk_size, max_block_size):
    # Input configuration:
    assert len(q_sizes) == len(kv_sizes)
    batch = len(q_sizes)
    nh, d = 2, 3
    use_int = True

    # attention configuration
    causal = True

    # blocks configuration
    num_blocks = 16
    max_blocks_per_sample = 16
    rand_blocks = False

    # create input tensors
    q, k, v = _get_qkv_tensors(batch, q_sizes, kv_sizes, nh, d, use_int=use_int)

    # run reference and convert result to 1d representation
    out_ref = pytorch_attention_ref(q, k, v, causal, q_sizes, kv_sizes)  # (b, nh, n, d)
    out_ref_1d = _create_q_1d(out_ref, q_sizes, nh, d)                   # (n_q_tokens, nh, d)

    # create required inputs for paged attention
    # note that kv cache is dependent on block_size, so created later
    q_1d = _create_q_1d(q, q_sizes, nh, d)
    q_sizes = torch.tensor(q_sizes, dtype=torch.long)
    kv_sizes = torch.tensor(kv_sizes, dtype=torch.long)
    max_seqlen_q = max(q_sizes)
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.long), q_sizes.cumsum(dim=0)])
    max_seqlen_k = max(kv_sizes).item()

    # test for all chunk sizes up till max_chunk_size and for all KV block sizes
    for block_size in range(1, max_block_size+1):
        # create kv cache using current block size
        block_table, k_cache, v_cache = _copy_kv_to_blocks(
            num_blocks,
            block_size,
            k,
            v,
            kv_sizes,
            max_blocks_per_sample,
            rand_blocks
        )
        for chunk_size_row in range(1, max_chunk_size+1):
            for chunk_size_col in range(1, max_chunk_size+1):
                print(f'Test for {q_sizes=} {kv_sizes=} {block_size=} {chunk_size_row=} {chunk_size_col=}')
                out = paged_attention_var_len(
                    q=q_1d.to(DEVICE),
                    k=k_cache.to(DEVICE),
                    v=v_cache.to(DEVICE),
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_q=cu_seqlens_q.to(DEVICE),
                    max_seqlen_k=max_seqlen_k,
                    cu_seqlens_k=None,
                    seqused_k=kv_sizes.to(DEVICE),
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=causal,
                    window_size=None,
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=False,
                    block_table=block_table.to(DEVICE),
                    return_softmax_lse=False,
                    out=None,
                    fa_version=2,
                    chunk_size_q=chunk_size_row,
                    chunk_size_k=chunk_size_col,
                )

                # run reference, align representations and compare
                same = torch.allclose(out.to('cpu'), out_ref_1d)
                assert same, f'Failed for {q_sizes=} {kv_sizes=} {block_size=} {chunk_size_row=} {chunk_size_col=}'
                print(f'Passed for {q_sizes=} {kv_sizes=} {block_size=} {chunk_size_row=} {chunk_size_col=}')


def test_pa_single():
    # -------------------------------------------------------------
    # TEST input:
    # input q, k, v
    batch, nh, d = 3, 2, 4
    q_sizes = [1, 6, 4]
    kv_sizes = [3, 7, 9]
    use_int = True

    # attention configuration
    causal = True

    # blocks configuration
    num_blocks = 10
    block_size = 6
    max_blocks_per_sample = 4
    rand_blocks = False

    # chunks configuration
    chunk_size_row = 5
    chunk_size_col = 5
    # -------------------------------------------------------------

    q, k, v = _get_qkv_tensors(batch, q_sizes, kv_sizes, nh, d, use_int=use_int)
    block_table, k_cache, v_cache = _copy_kv_to_blocks(
        num_blocks,
        block_size,
        k,
        v,
        kv_sizes,
        max_blocks_per_sample,
        rand_blocks
    )
    q_1d = _create_q_1d(q, q_sizes, nh, d)

    max_seqlen_q = max(q_sizes)
    t_q_sizes = torch.tensor(q_sizes, dtype=torch.long)
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.long), t_q_sizes.cumsum(dim=0)])
    max_seqlen_k = max(kv_sizes)
    seqused_k = torch.tensor(kv_sizes, dtype=torch.long)

    out = paged_attention_var_len(
        q=q_1d.to(DEVICE),
        k=k_cache.to(DEVICE),
        v=v_cache.to(DEVICE),
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q.to(DEVICE),
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=None,
        seqused_k=seqused_k.to(DEVICE),
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=None,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=block_table.to(DEVICE),
        return_softmax_lse=False,
        out=None,
        fa_version=2,
        chunk_size_q=chunk_size_row,
        chunk_size_k=chunk_size_col,
    )

    # run reference, align representations and compare
    out_ref = pytorch_attention_ref(q, k, v, causal, q_sizes, kv_sizes)  # (b, nh, n, d)
    out_ref_1d = _create_q_1d(out_ref, q_sizes, nh, d)                   # (n_q_tokens, nh, d)
    same = torch.allclose(out.to('cpu'), out_ref_1d)
    assert same
    print(f'test_pa_single: Passed')


if __name__ == '__main__':
    test_pa_single()
