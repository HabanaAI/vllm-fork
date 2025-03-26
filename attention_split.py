import torch
from einops import rearrange
import habana_frameworks.torch
import sys
import time


def attention_split(attn_type):
    batch_size = 1
    seq_len = 6400
    attn_heads = 16
    head_dim = 80
    device = "hpu"

    q = torch.randn(batch_size, seq_len, attn_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, attn_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, attn_heads, head_dim, device=device)

    if attn_type == "split":
        # start , end , step
        cu_seq_lens = torch.arange(0, seq_len + 1, 64, dtype=torch.int32, device=device)
    else:
        cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    outputs = []
    t1 = time.time()
    for i in range(1, len(cu_seq_lens)):
        start_idx = cu_seq_lens[i - 1].item()
        end_idx = cu_seq_lens[i].item()
        q_i = q[:, start_idx:end_idx]
        k_i = k[:, start_idx:end_idx]
        v_i = v[:, start_idx:end_idx]
        q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i])

        if device == "hpu":
            from habana_frameworks.torch.hpex.kernels import FusedSDPA

            output_i = FusedSDPA.apply(q_i, k_i, v_i, None, 0.0)
        else:
            output_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, dropout_p=0.0
            )

        output_i = rearrange(output_i, "b h s d -> b s h d ")
        outputs.append(output_i)

    t2 = time.time()

    context_layer = torch.cat(outputs, dim=1)
    # print(context_layer)
    print(f"attn_type: {attn_type}: Time Taken = {(t2 - t1) * 1000:.3f} ms")


# # Run the test
def main():
    attn_type = sys.argv[1] if len(sys.argv) > 1 else ""
    attention_split(attn_type)


if __name__ == "__main__":
    main()
