import math
import torch
from bisect import bisect_left
from typing import Optional, List

try:
    import habana_frameworks
    import habana_frameworks.torch.core as htcore
    USE_HPU = True
except:
    USE_HPU = False


PAD_BLOCK_ID = -1
DO_PAD_BATCH = True
DO_PAD_MAX_SEQLEN_K = True
DO_PAD_MAX_SEQLEN_Q = True
DO_AVOID_FULLY_PADDED_CHUNKS = True

# Default bins configuration
BATCH_BINS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256]
MAX_SEQLEN_K_BINS = [128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]
MAX_SEQLEN_Q_BINS = [1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]
NON_MASKED_BATCH_BINS = sorted(list(set([0, 1, 2, 3, 4, 5, 6, 7, 8] + BATCH_BINS)))


def mark_step():
    if USE_HPU:
        htcore.mark_step()


def nearset_multiple(value, divisor):
    return int(math.ceil(value / divisor)) * divisor


def bucketize_tensor_on_batch_dim(t, padded_batch):
    """ Pad (prepend) tensor on batch dim

        Padding is *prepended* to batch dim.

        Prepending is mandatory (appending not allowed).
        This is due to write_back_q_slice_to_tensor() that writes from slice to q-shaped tensor.
        The padded elements have index=-1 which causes a write to last q-shape token.
        When using prepending the "real" values will overwrite the padded values - Ok.
        If we were to use appending, the padding would have over-write the real values - Bad.
    """
    batch = t.shape[0]
    pad = [0] * (2 * len(t.shape))
    pad[-2] = padded_batch - batch
    t_padded = torch.nn.functional.pad(t, pad)
    return t_padded


def get_bucketed_batch(batch):
    """ Force static shapes by padding batch to nearest batch bin """
    padded_batch = batch
    if DO_PAD_BATCH:
        if batch > BATCH_BINS[-1]:
            return nearset_multiple(batch, 64)

        i = bisect_left(BATCH_BINS, batch)
        padded_batch = BATCH_BINS[i]
    return padded_batch


def get_bucketed_max_seqlen_k(max_seqlen_k):
    """ Force static shapes by padding max_seqlen_k to nearest bin.
        This controls the size of the K dim of the attention matrix (aka m).
    """
    padded_max_seqlen_k = max_seqlen_k
    if DO_PAD_MAX_SEQLEN_K:
        if max_seqlen_k > MAX_SEQLEN_K_BINS[-1]:
            return nearset_multiple(max_seqlen_k, 1024)
        i = bisect_left(MAX_SEQLEN_K_BINS, max_seqlen_k)
        padded_max_seqlen_k = MAX_SEQLEN_K_BINS[i]
    return padded_max_seqlen_k


def get_bucketed_max_seqlen_q(max_seqlen_q):
    """ Force static shapes by padding max_seqlen_q to nearest bin.
        This controls the size of the Q dim of the attention matrix (aka n).
    """
    padded_max_seqlen_q = max_seqlen_q
    if DO_PAD_MAX_SEQLEN_Q:
        if max_seqlen_q > MAX_SEQLEN_Q_BINS[-1]:
            return nearset_multiple(max_seqlen_q, 1024)
        i = bisect_left(MAX_SEQLEN_Q_BINS, max_seqlen_q)
        padded_max_seqlen_q = MAX_SEQLEN_Q_BINS[i]
    return padded_max_seqlen_q


def get_bucketed_n_non_masked(n_non_masked):
    """ When calculating attention for a specific chunk, we only calculate for samples that are not fully masked.
        In order to avoid dynamic shapes, we force static shapes by padding n_non_masked to nearest bin.
    """
    n_padded_non_masked = n_non_masked
    if DO_AVOID_FULLY_PADDED_CHUNKS:
        if n_non_masked > NON_MASKED_BATCH_BINS[-1]:
            return nearset_multiple(n_non_masked, 256)
        i = bisect_left(NON_MASKED_BATCH_BINS, n_non_masked)
        n_padded_non_masked = NON_MASKED_BATCH_BINS[i]
    return n_padded_non_masked


def bucketize_non_masked_tensor(n_non_masked, non_masked):
    """ Given non_masked 1D tensor, bucketize it by flipping false to true to get
        n_non_masked_bucketed true values. This will ensure that the rest of the graph
        will have static shapes with batch = n_non_masked_bucketed.
    """
    b_padded = non_masked.shape[0]
    n_non_masked_bucketed = get_bucketed_n_non_masked(n_non_masked)
    sorting_indices = torch.argsort(non_masked.to(torch.int), descending=True)
    mask = torch.arange(b_padded, device=non_masked.device) < n_non_masked_bucketed
    non_masked_bucketed = torch.zeros(b_padded, dtype=torch.bool, device=non_masked.device)
    non_masked_bucketed[sorting_indices] = mask
    return non_masked_bucketed


def convert_right_pad_to_left_pad(blocks: torch.Tensor, seq_used: torch.Tensor, block_size: int):
    """ Converts blocks with shape (b, n_blocks) from right padding to left padding along dim=1.
        E.g. given [[8,0,0,0], [6,7,0,0], [5,3,4,0]] returns: [[0,0,0,8], [0,0,6,7], [0,5,3,4]]
        The function uses static shapes and avoid breaking the execution graph, e.g. avoid .item()
    """
    _, n = blocks.shape
    n_used_blocks = torch.ceil(seq_used / block_size).long()
    valid_mask = torch.arange(n, device=blocks.device).unsqueeze(0) < n_used_blocks.unsqueeze(1)
    n_zeros = (n - n_used_blocks).unsqueeze(1)
    indices = torch.arange(n, dtype=torch.long, device=blocks.device).unsqueeze(0) - n_zeros
    pad_mask = valid_mask.flip((1,))
    indices *= pad_mask
    res = torch.gather(blocks, 1, indices)
    res *= pad_mask
    return res


def extract_blocks(tokens_to_block, max_blocks):
    """ Given a 2D tensor which maps block-id of token indexes,
        Returns block_ids: a 2D *left-padded* tensor which includes the unique block-ids
        Note that padded block is marked with PAD_BLOCK_ID

        Example:
        t = torch.tensor([
            [1, 1, 2, 2, 2, 2, 3, 3],
            [0, 0, 5, 5, 5, 5, 5, 6],
            [0, 0, 0, 0, 0, 7, 7, 7]
        ])
        max_blocks = 3
        extract_blocks(t, max_blocks)
        tensor([
            [ 1,  2, 3],
            [-1,  5, 6],
            [-1, -1, 7]
        ])
    """
    batch, m = tokens_to_block.shape

    # Identify block-id transitions
    transitions = torch.zeros_like(tokens_to_block, dtype=torch.bool, device=tokens_to_block.device)
    transitions[:, 1:] = tokens_to_block[:, 1:] != tokens_to_block[:, :-1]
    transitions[:, 0] = True  # First element is always a block start

    # Assign running group indices per row
    # Where for each row, the highest group id should be equal to max_groups-1
    group_ids = torch.cumsum(transitions, dim=1) - 1
    max_group_num = group_ids.max(dim=1, keepdim=True)[0]  # Shape: (batch, 1)
    group_ids = group_ids + (max_blocks - 1 - max_group_num)

    # Scatter block id values based on group ids
    result = torch.full((batch, max_blocks), fill_value=PAD_BLOCK_ID, dtype=tokens_to_block.dtype, device=tokens_to_block.device)
    result.scatter_(1, group_ids, tokens_to_block)
    return result


def get_num_tokens_at_last_block(tokens_to_block):
    """ Given a 2D tensor which maps block-id of token indexes,
        Return number of tokens in right-most block.

        Example:
        t = torch.tensor([
            [1, 1, 2, 2, 2, 2, 3, 3],    # 2 tokens at last block=3
            [0, 0, 5, 5, 5, 6, 6, 6],    # 3 tokens at last block=6
        ])
        get_num_tokens_at_last_block(t)
        tensor([2, 3])
    """
    last_block = tokens_to_block[:, -1].unsqueeze(1)
    return torch.sum(tokens_to_block.eq(last_block), dim=1)


def compute_tokens_per_block(seq: torch.Tensor, block_size: int, max_blocks: int) -> torch.Tensor:
    """ Given used tokens per sample, compute number of tokens used per block.
        This assumes that the used tokens are *right aligned*.
        The returned tokens per block is for up to a maximum of max_blocks.

        Example:
        seq = torch.tensor([15, 7, 20], dtype=torch.long)
        block_size = 5
        max_blocks = 6
        expected = torch.tensor([
            [0, 0, 0, 5, 5, 5],
            [0, 0, 0, 0, 5, 2],
            [0, 0, 5, 5, 5, 5]
        ], dtype=torch.long)
        result = compute_tokens_per_block(seq, block_size, max_blocks)
        assert torch.equal(result, expected)
    """
    batch = seq.shape[0]
    seq = seq.unsqueeze(1)  # (batch, 1)

    # Compute total of full blocks needed per row and is last block full
    num_blocks = (seq / block_size).ceil().int()
    remainder = seq % block_size
    is_last_block_full = remainder == 0

    # Create mask for where to put full block_size values
    indices = torch.arange(max_blocks, device=seq.device).expand(batch, -1)
    full_block_mask = indices >= (max_blocks - num_blocks)

    # Set number of tokens for full blocks and last block
    ret = torch.zeros((batch, max_blocks), dtype=torch.long, device=seq.device)
    ret[full_block_mask] = block_size
    is_last_block_partial = (~is_last_block_full)
    ret[:, -1:] = (is_last_block_partial * remainder + is_last_block_full * block_size)
    return ret


def compute_block_id_for_token_idx(m, block_table, tokens_per_block):
    """ Creates token-index to block-id mapping
        Output shape is (batch, m)

        Example:
        m = 16
        block_table = torch.tensor([
            [0, 0, 0, 1],
            [0, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        tokens_per_block = torch.tensor([
            [0, 0, 4, 3],
            [0, 4, 4, 2],
            [4, 4, 4, 4]
        ])
        out = compute_block_id_for_token_idx(m, block_table, tokens_per_block)
        out
        tensor([[-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1,  2,  2,  2, 2, 3, 3, 3, 3, 4, 4],
                [ 5,  5,  5,  5,  6,  6,  6,  6,  7, 7, 7, 7, 8, 8, 8, 8]])
    """
    batch, max_blocks = block_table.shape

    # Create index offsets for each token
    token_offsets = torch.arange(m, device=block_table.device).expand(batch, m)

    # token_counts_except_last_block is the cumulative number of tokens per block except the last block
    # calculated in 2 steps:
    # 1. token_counts_except_last_block: calculate the cumulative number of tokens from next block till last block
    # 2. m - token_counts_except_last_block: cumulative number of tokens in current block
    token_counts_except_last_block = tokens_per_block[:, 1:].flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    token_counts_except_last_block = m - token_counts_except_last_block

    # we know that the number of cumulative tokens on last block is m
    # therefore, token_counts is a concat of: token_counts_except_last_block + [m]
    token_counts_last_block = torch.full((batch, 1), fill_value=m, device=tokens_per_block.device,
                                         dtype=tokens_per_block.dtype)
    token_counts = torch.cat([token_counts_except_last_block, token_counts_last_block], dim=1)

    # Create a mask to determine which block each token belongs to. shape=(batch, m, max_blocks)
    mask = token_offsets.unsqueeze(-1) < token_counts.unsqueeze(1)

    # Find the first occurrence in mask where it's True (indicating the block)
    indices = torch.argmax(mask.int(), dim=2)

    # Use out as an index to gather block ids from block_table
    out = torch.gather(block_table, 1, indices)

    # Set block_id=PAD_BLOCK_ID to padding tokens
    start_valid_tokens = m - tokens_per_block.sum(dim=1)
    pad_mask = token_offsets < start_valid_tokens.unsqueeze(1)
    out[pad_mask] = PAD_BLOCK_ID

    return out


def compute_per_token_offset_in_block(block_id_for_token_idx):
    """ Compute the offset of each token in the block it belongs to.

        Example:
        Given block_id_for_token_idx:  ([[-1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
                                         [-1,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3]])
        Return offsets:                ([[ 0,  1,  2,  3,  4,  5,  6,  0,  1,  2,  3,  4],
                                         [ 0,  0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  4]])
    """
    # Create a mask indicating the start of a contiguous group
    # The first column is always the start, and a new start occurs when the value changes
    device = block_id_for_token_idx.device
    group_start_mask = torch.cat([
        torch.ones(block_id_for_token_idx.shape[0], 1, dtype=torch.bool, device=device),
        block_id_for_token_idx[:, 1:] != block_id_for_token_idx[:, :-1]
    ], dim=1)

    # Create a tensor of column indices for each row
    col_indices = torch.arange(block_id_for_token_idx.shape[1], device=device).expand_as(block_id_for_token_idx)

    # For each row, mark group start positions with their column index; for others, use -1.
    # (Using -1 ensures that the cummax will ignore these until a real start is found)
    group_start_indices = torch.where(group_start_mask, col_indices, -torch.ones_like(col_indices))

    # Propagate the most recent group start along each row
    latest_group_start, _ = torch.cummax(group_start_indices, dim=1)

    # The offset for each element is its column index minus the column index of the group start
    offsets = col_indices - latest_group_start
    return offsets


def gather_kv_tokens(k, v, block_id_for_token_idx, offset_in_block_for_token_idx, chunk_start, chunk_size):
    """
    Gathers token chunks from a block-structured tensor.
    Gathers 'chunk_size' kv tokens starting from 'chunk_start'.

    Args:
        k (torch.Tensor): Tensor of shape (n_blocks, block_size, nh, d) containing k token data.
        v (torch.Tensor): Tensor of shape (n_blocks, block_size, nh, d) containing v token data.
        block_id_for_token_idx (torch.Tensor): Tensor of shape (batch, max_tokens) containing block id for token index.
        offset_in_block_for_token_idx (torch.Tensor): Tensor of shape (batch, max_tokens) containing offset within block.
        chunk_start (int): Specifies the starting index of the chunk to extract.
        chunk_size (int): Number of tokens to extract per sample.

    Returns:
        torch.Tensor: Tensor of shape (batch, chunk_size, nh, d) containing gathered tokens.

    Example:

        # inputs:
        n_blocks, block_size, nh, d = 10, 4, 1, 1
        k_cache = torch.arange(n_blocks * block_size).reshape(n_blocks, block_size, 1, 1).expand(n_blocks, block_size, nh, d)
        v_cache = torch.arange(n_blocks * block_size).reshape(n_blocks, block_size, 1, 1).expand(n_blocks, block_size, nh, d)
        seq_used_k = torch.tensor([
            2,
            7,
            12
        ])
        block_table = torch.tensor([
            [1, 0, 0],
            [2, 4, 0],
            [5, 7, 9],
        ])
        batch, max_blocks_per_sample = block_table.shape
        chunk_start = 3
        chunk_size = 7

        # prepare for gather_kv_tokens
        assert seq_used_k.shape[0] == batch
        m = seq_used_k.max()
        block_table = convert_right_pad_to_left_pad(block_table)
        m_padded = nearset_multiple(m, chunk_size)
        tokens_per_block = compute_tokens_per_block(seq_used_k, block_size, max_blocks_per_sample)
        t_block_id_for_token_idx = compute_block_id_for_token_idx(m_padded, block_table, tokens_per_block)
        t_offset_in_block_for_token_idx = compute_per_token_offset_in_block(t_block_id_for_token_idx)

        # activate:
        k_chunk, v_chunk, pad_mask = gather_kv_tokens(
            k_cache, v_cache, t_block_id_for_token_idx, t_offset_in_block_for_token_idx, chunk_start, chunk_size)
        assert k_chunk.shape == (batch, chunk_size, nh, d)
        assert v_chunk.shape == (batch, chunk_size, nh, d)
        for i in range(batch):
            sample = k_chunk[i].tolist()
            print(sample)

        # Output
        # [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]]
        # [[[0]], [[0]], [[0]], [[0]], [[8]], [[9]], [[10]]]
        # [[[21]], [[22]], [[23]], [[28]], [[29]], [[30]], [[31]]]
    """
    n_blocks, block_size, nh, d = k.shape
    batch = block_id_for_token_idx.shape[0]

    # slice request chunk from block_id_for_token_idx
    # slice also the offsets of each token in its block
    sliced_block_ids = block_id_for_token_idx[:, chunk_start:chunk_start + chunk_size]
    sliced_offset_in_block = offset_in_block_for_token_idx[:, chunk_start:chunk_start + chunk_size]

    # get unique block ids that we need to collect tokens from
    # maximum blocks to gather is incremented by 1 in order to account for a scenario where we use
    # only part of the tokens from the first and last block
    max_blocks_to_gather = int(math.ceil(chunk_size / block_size)) + 1
    blocks_to_gather = extract_blocks(sliced_block_ids, max_blocks_to_gather)

    # force padded blocks to read from block-id=0
    # this is not mandatory since we mask the tokens of the padded blocks.
    # however, this can be handy for debug
    blocks_to_gather = torch.where(blocks_to_gather == PAD_BLOCK_ID, 0, blocks_to_gather)

    # since we gather *full* blocks, we might gather extra elements from most left and right blocks.
    # therefore, we need to discard those elements. for that, we calculate the number of extra left and right
    # elements that were gathered. this is done per row.
    right_most_offset_in_block = sliced_offset_in_block[:, -1] % block_size
    n_extra_right_tokens = block_size - (right_most_offset_in_block + 1)
    n_gathered_tokens = blocks_to_gather.shape[1] * block_size
    n_extra_left_tokens = n_gathered_tokens - chunk_size - n_extra_right_tokens
    # keep_indices are running chunk indices shifted by the number of extra left tokens
    keep_indices = torch.arange(chunk_size, device=k.device).unsqueeze(0).repeat(batch, 1)
    keep_indices = keep_indices + n_extra_left_tokens.unsqueeze(1)
    keep_indices = keep_indices.view(batch, chunk_size, 1, 1).repeat(1, 1, nh, d)
    pad_mask = (sliced_block_ids == PAD_BLOCK_ID)

    # gather blocks_to_gather into a continuous tensor, remove extra tokens and zero pad tokens
    def extract_tokens_from_gathered_blocks(block_cache):
        # Gather tokens from the required blocks
        tokens = block_cache[blocks_to_gather]   # Shape: (batch, n_blocks_to_gather, block_size,  nh, d)
        tokens = tokens.view(batch, -1, nh, d)   # shape: (batch, n_blocks_to_gather * block_size, nh, d)
        # Remove extra tokens from left and right block
        tokens = tokens.gather(1, keep_indices)
        # Mask pad tokens
        tokens.masked_fill_(pad_mask.view(batch, chunk_size, 1, 1), 0)
        return tokens

    # extract k and v tokens from required gathered blocks
    k_tokens = extract_tokens_from_gathered_blocks(k)
    v_tokens = extract_tokens_from_gathered_blocks(v)
    return k_tokens, v_tokens, pad_mask


def calculate_q_chunk_indices(cu_seqlen: torch.Tensor, max_seqlen: int, chunk_start: int, chunk_size: int):
    """ Return q chunk indices and pad mask for the required chunk

        The indices of the chunk are returned in 2d (batch, chunk_size).
        The tokens are left-padded (i.e. right-aligned).

        This method is expected to be called once per q chunk and then the resulting indices and mask should be
        used to gather q, out, sm_l and sm_m.

        Example:
        chunk_size = 4
        cu_seqlen = torch.tensor([5,16])
        max_seqlen = 12   # 11 padded to chunk_size=4

        for chunk_start in range(0, max_seqlen, chunk_size):
            indices, pad_mask = calculate_q_chunk_indices(cu_seqlen, max_seqlen, chunk_start, chunk_size)

        Returned indices and masks:
        chunk_start=0:  indices = [[-7, -6, -5, -4], [4,   5,  6,  7]]  pad_mask = [[1, 1, 1, 1], [1, 0, 0, 0]]
        chunk_start=4:  indices = [[-3, -2, -1,  0], [8,   9, 10, 11]]  pad_mask = [[1, 1, 1, 0], [0, 0, 0, 0]]
        chunk_start=8:  indices = [[ 1,  2,  3,  4], [12, 13, 14, 15]]  pad_mask = [[0, 0, 0, 0], [0, 0, 0, 0]]

        Given q = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ]
        Sliced q for the returned indices (after applying pad_mask):
        chunk_start=0:  q = [[0, 0, 0, 0], [0,   6,  7,  8]]
        chunk_start=4:  q = [[0, 0, 0, 1], [9,  10, 11, 12]]
        chunk_start=8:  q = [[2, 3, 4, 5], [13, 14, 15, 16]]
    """
    batch = cu_seqlen.shape[0]
    cu_seqlen = cu_seqlen.unsqueeze(1)
    chunk_end = chunk_start + chunk_size
    chunk_indices = torch.arange(chunk_start, chunk_end, device=cu_seqlen.device).unsqueeze(0).repeat(batch, 1)
    q_indices = cu_seqlen - (max_seqlen - chunk_indices)
    cu_seqlen = torch.nn.functional.pad(cu_seqlen, (0, 0, 1, 0))
    q_pad_mask = (q_indices < cu_seqlen[:-1])
    # Note:
    # The q_indices might include negative numbers which will get tokens from end of q and then, we mask them.
    # An alternative is to mask the indices to (e.g.) 0 before the gather and then gather from 0-offset.
    # In this case, we either need to mask again the 0-index tokens (that may have value != 0) or make sure that
    # the 0-index is always 0.
    # In any case for now, set the negative indices to 0.
    q_indices.masked_fill_(q_pad_mask, 0)
    return q_indices, q_pad_mask   # (batch, chunk_size), (batch, chunk_size)


def gather_q_slice_from_tensor(t: torch.Tensor, indices: torch.Tensor, pad_mask: Optional[torch.Tensor], pad_val: float):
    """ Return a slice from t based on the indices and mask calculated by calculate_q_chunk_indices()
        The function first gathers the chunk, and then masks the padded indices to 'pad_val'
    """
    t_chunk = t[indices]
    n_broadcast_dims = len(t_chunk.shape) - len(pad_mask.shape)
    pad_mask = pad_mask.view([*pad_mask.shape] + [1] * n_broadcast_dims)
    t_chunk.masked_fill_(pad_mask, pad_val)
    return t_chunk


def calculate_q_slice_write_back_indices(
        q_n_tokens: int,
        t_indices: torch.Tensor,
        slice_select_mask: torch.Tensor
):
    """ Calculate the write-back indices to a q-like tensor.

        The valid_indices include last_index as pad values.
        However, when writing the last k chunk, the pads will override the real value of the last index
        Therefore, we work around this by re-writing the value of the last index
    """
    last_index = q_n_tokens - 1
    valid_indices = torch.where(slice_select_mask, t_indices, last_index)
    is_last_index_in_slice = t_indices[-1, -1].eq(last_index).logical_and(slice_select_mask[-1, -1])
    return valid_indices, is_last_index_in_slice


def write_back_q_slice_to_tensor(
        t: torch.Tensor,
        t_slice: torch.Tensor,
        q_indices: torch.Tensor,
        is_last_index_in_slice: torch.Tensor
):
    """ Write back t_slice into t[q_indices].
        For non-selected indices, we gather into the 'last_index' of t.
        The actual last_index is written last (timeline wise) which ensures correctness.
    """
    last_index = t.shape[0] - 1
    t_at_last_index = t[last_index]
    t[q_indices] = t_slice
    t[last_index] = torch.where(is_last_index_in_slice, t_slice[-1, -1], t_at_last_index)


def get_qk_chunk_pad_mask(q_pad_mask, kv_pad_mask):
    q_chunk_size = q_pad_mask.shape[1]
    kv_chunk_size = kv_pad_mask.shape[1]
    q_pad = q_pad_mask.view(-1, q_chunk_size, 1).repeat(1, 1, kv_chunk_size)
    kv_pad = kv_pad_mask.view(-1, 1, kv_chunk_size).repeat(1, q_chunk_size, 1)
    pad = torch.logical_or(q_pad, kv_pad)
    return pad  # (batch, q_chunk_size, kv_chunk_size)


def get_qk_chunk_causal_mask(i, i_end, j, j_end, device):
    row_indexes = torch.arange(i, i_end, device=device)[:, None]
    col_indexes = torch.arange(j, j_end, device=device)[None, :]
    causal_mask = row_indexes < col_indexes
    return causal_mask  # (q_chunk_size, kv_chunk_size)


def safe_exp_sub(t1, t2):
    """ Return torch.exp(t1 - t2). In case t1[loc] = t2[loc] = -inf, Return 0 """
    return torch.where(torch.maximum(t1, t2).eq(float("-inf")), float(0), torch.exp(t1 - t2))


"""
Copied interface from vllm/vllm/vllm_flash_attn/flash_attn_interface.py
"""
def paged_attention_var_len(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k=None,
        seqused_k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size: Optional[List[int]] = None,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
        *,
        return_softmax_lse=False,
        out=None,
        fa_version: int = 2,
        chunk_size_q=128,
        chunk_size_k=128,
):
    # Verify supported features
    assert k.shape == v.shape
    assert seqused_k is not None
    assert cu_seqlens_k is None
    assert block_table is not None
    assert dropout_p == 0.0
    assert softcap == 0.0
    assert window_size is None or (len(window_size) == 2 and window_size[0] == -1 and window_size[1] == -1)
    assert alibi_slopes is None
    assert deterministic is False
    assert return_attn_probs is False
    assert return_softmax_lse is False

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # collect required dim sizes
    # n_q_tokens : total number of q tokens from all samples squeezed into 1d
    # n          : maximum number of q tokens for a single sample
    # m          : maximum number of k tokens for a single sample
    # b          : batch size. deduced from the shape of cu_seqlens_q that lists offsets of q tokens in 1d q
    #              e.g. if cu_seqlens_q = [0, 4, 6] => b=2 with 1st query len=4 and 2nd query len=2
    n_q_tokens, n_heads, head_dim = q.shape
    n, m = max_seqlen_q, max_seqlen_k
    b, max_blocks_per_sample = block_table.shape
    block_size = k.shape[1]

    assert seqused_k.shape[0] == b
    assert cu_seqlens_q.shape[0] == b + 1   # cu_seqlens_q is prepended with 0

    # Step 1: Bucketing
    # First we handle bucketing via padding.
    # This is done at the entry of the function to enable the rest of the function
    # to use static/pre-compiled recipies.

    # Calculate bucketing of batch dim
    b_padded = get_bucketed_batch(b)

    # Calculate bucketing of m, n dimensions
    # bucketize m (i.e. max_seqlen_k) and then make it divisible by k chunk size
    # bucketize n (i.e. max_seqlen_q) and then make it divisible by q chunk size
    m_padded = nearset_multiple(get_bucketed_max_seqlen_k(m), chunk_size_k)
    n_padded = nearset_multiple(get_bucketed_max_seqlen_q(n), chunk_size_q)

    # Remove cu_seqlens_q[0] which is always 0. This will make it of shape (batch, )
    cu_seqlens_q = cu_seqlens_q[1:]

    # Bucketize tensors
    seqused_k = bucketize_tensor_on_batch_dim(seqused_k, b_padded)
    cu_seqlens_q = bucketize_tensor_on_batch_dim(cu_seqlens_q, b_padded)
    block_table = bucketize_tensor_on_batch_dim(block_table, b_padded)

    # -----------------------------------------
    # From HERE, all tensors should be bucketed
    # -----------------------------------------

    # step 2: convert block_table from right padded to left padded (right aligned)
    block_table = convert_right_pad_to_left_pad(block_table, seqused_k, block_size)

    # step 3: compute tokens per block where blocks are right aligned
    tokens_per_block = compute_tokens_per_block(seqused_k, block_size, max_blocks_per_sample)

    # step 4: compute block_id_per_token_idx: block_id_for_token_idx[0...m_padded]
    #         then, compute for each token the offset within its block
    block_id_for_token_idx = compute_block_id_for_token_idx(m_padded, block_table, tokens_per_block)
    offset_in_block_for_token_idx = compute_per_token_offset_in_block(block_id_for_token_idx)

    # step 5: allocate final output and online softmax partial tensors
    # TODO: consider using transpose on sm_l, sm_m
    out = torch.zeros_like(q) if out is None else out.zero_()
    sm_l = torch.zeros((n_q_tokens, n_heads, 1), dtype=q.dtype, device=q.device)
    sm_m = torch.full((n_q_tokens, n_heads, 1), float('-inf'), dtype=q.dtype, device=q.device)

    # Online softmax Outer loop over KV blocks
    for j in range(0, m_padded, chunk_size_k):
        k_j_full, v_j_full, kv_j_pad_mask = gather_kv_tokens(
            k, v, block_id_for_token_idx, offset_in_block_for_token_idx, chunk_start=j, chunk_size=chunk_size_k)
        k_j_full = k_j_full.transpose(1, 2)  # (b, nh, chunk_m, d)
        v_j_full = v_j_full.transpose(1, 2)  # (b, nh, chunk_m, d)

        j_end = j + chunk_size_k
        k_j_scaled_full = k_j_full * softmax_scale
        for i in range(0, n_padded, chunk_size_q):  # Inner loop over Q blocks
            # calculate indices and mask for current chunk for q-shaped tensors
            chunk_i_idx, chunk_i_pad_mask = calculate_q_chunk_indices(
                cu_seqlens_q, max_seqlen=n_padded, chunk_start=i, chunk_size=chunk_size_q)

            # for causality, we normalize i to j to simulate an m x m matrix
            normalized_i = i + m_padded - n_padded
            normalized_i_end = normalized_i + chunk_size_q

            if causal and normalized_i_end <= j:
                # i <= j for the lower left element in the current block
                # therefore we can mask out (ignore) the whole block
                # normalized_i_end "transforms" the scenario into regular n x n causal attention
                continue

            # calculate mask per sample
            # for non-causal: mask out q and k pad tokens in the current chunk
            # for causal: in addition, mask out based on causality
            pad_mask = get_qk_chunk_pad_mask(chunk_i_pad_mask, kv_j_pad_mask)
            if causal:
                causal_mask = get_qk_chunk_causal_mask(normalized_i, normalized_i_end, j, j_end, q.device)
                pad_mask = pad_mask.logical_or(causal_mask.unsqueeze(0))

            k_j_scaled = k_j_scaled_full
            v_j = v_j_full

            if DO_AVOID_FULLY_PADDED_CHUNKS:
                # fully_masked are samples that are entirely masked-out (no need to process the chunk)
                # non_masked are the rest of the samples (it is required to process the chunk)
                fully_masked = torch.all(pad_mask.view(b_padded, -1), dim=-1)   # (b, 1)
                non_masked = ~fully_masked

                # --------------------------------------------------------------------------------
                # the number of non-masked samples is dynamic, therefore force a graph break here!
                # --------------------------------------------------------------------------------
                n_non_masked = non_masked.sum().item()
                mark_step()

                # nothing samples to process. skip entirely this chunk.
                if n_non_masked == 0:
                    continue

                # get the bucketed number of the non-masked samples
                # then, bucketize non_masked accordingly
                non_masked_bucketed = bucketize_non_masked_tensor(n_non_masked, non_masked)

                # modify masks to "gather" non-masked samples (after bucketing)
                pad_mask = pad_mask[non_masked_bucketed]
                chunk_i_idx = chunk_i_idx[non_masked_bucketed]
                chunk_i_pad_mask = chunk_i_pad_mask[non_masked_bucketed]

                # get k, v slices for non-masked samples
                k_j_scaled = k_j_scaled_full[non_masked_bucketed]
                v_j = v_j_full[non_masked_bucketed]

            # gather q slice after applying pad mask
            q_i = gather_q_slice_from_tensor(q, chunk_i_idx, chunk_i_pad_mask, pad_val=0.)
            q_i = q_i.transpose(1, 2)   # (b, nh, chunk_n, d)

            # perform bmm(q, k.T) and apply calculated mask (note that h dim is transposed)
            s_ij = torch.einsum('bhnd,bhmd->bhnm', q_i, k_j_scaled)    # (b, nh, chunk_n, chunk_m)
            s_ij.masked_fill_(pad_mask.unsqueeze(1), float('-inf'))

            # get chunks
            o_i = gather_q_slice_from_tensor(out, chunk_i_idx, chunk_i_pad_mask, pad_val=0.)
            l_i = gather_q_slice_from_tensor(sm_l, chunk_i_idx, chunk_i_pad_mask, pad_val=0.)
            m_i = gather_q_slice_from_tensor(sm_m, chunk_i_idx, chunk_i_pad_mask, pad_val=float('-inf'))

            # transpose
            o_i = o_i.transpose(1, 2)    # (b, nh, chunk_n, d)
            l_i = l_i.transpose(1, 2)    # (b, nh, chunk_n, 1)
            m_i = m_i.transpose(1, 2)    # (b, nh, chunk_n, 1)

            m_ij = torch.max(s_ij, dim=-1, keepdim=True)[0]   # (b, nh, chunk_n, 1)
            p_ij = safe_exp_sub(s_ij, m_ij)                   # (b, nh, chunk_n, chunk_m)
            l_ij = p_ij.sum(dim=-1, keepdim=True)             # (b, nh, chunk_n, 1)

            m_i_new = torch.maximum(m_i, m_ij)
            e_past = safe_exp_sub(m_i, m_i_new)
            e_new = safe_exp_sub(m_ij, m_i_new)
            l_i = e_past * l_i
            l_i_new = l_i + e_new * l_ij
            o_i_new = l_i * o_i + e_new * torch.einsum('bhnm,bhmd->bhnd', p_ij, v_j)
            o_i_new = torch.where(l_i_new.eq(0), float(0), o_i_new / l_i_new)

            # calculate the indices in q-like tensor to write back the current slice
            chunk_i_select_mask = ~chunk_i_pad_mask
            q_indices,  is_last_index_in_slice = calculate_q_slice_write_back_indices(
                q_n_tokens=out.shape[0], t_indices=chunk_i_idx, slice_select_mask=chunk_i_select_mask
            )

            # write back chunks into place
            write_back_q_slice_to_tensor(out, o_i_new.transpose(1, 2), q_indices,  is_last_index_in_slice)
            write_back_q_slice_to_tensor(sm_l, l_i_new.transpose(1, 2), q_indices,  is_last_index_in_slice)
            write_back_q_slice_to_tensor(sm_m, m_i_new.transpose(1, 2), q_indices,  is_last_index_in_slice)

            # --------------------------------------------------------------------------------
            # end of dynamic graph
            # --------------------------------------------------------------------------------
            mark_step() if DO_AVOID_FULLY_PADDED_CHUNKS else None

    return out
