import torch
from einops import rearrange
import habana_frameworks.torch
import sys
import time
from habana_frameworks.torch.hpu.metrics import metric_global
import argparse
import habana_frameworks.torch.core as htcore
import contextlib
torch.manual_seed(0)

def profile_ctx():
        activities = [
            torch.profiler.ProfilerActivity.HPU,
            torch.profiler.ProfilerActivity.CPU,
        ]
        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=100,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("log_dir"),
        )

def get_slice(t, slice_type, start_idx, end_idx, rearrange_outside):
    if rearrange_outside:
        if slice_type == 'original':
            return t[:, :, start_idx:end_idx]
        elif slice_type == 'index_select':
            return torch.index_select(t, 2, torch.arange(start_idx, end_idx, device=t.device, dtype=torch.int32))
        else:
            assert False
    else:
        if slice_type == 'original':
            return t[:, start_idx:end_idx]
        elif slice_type == 'index_select':
            return torch.index_select(t, 1, torch.arange(start_idx, end_idx, device=t.device, dtype=torch.int32))
        else:
            assert False

def attention_split(attn_type, index_type, warmup, islist, rearrange_outside):
    batch_size = 1
    seq_len = 6400
    attn_heads = 16
    head_dim = 80
    device = "hpu"

    q = torch.randn(batch_size, seq_len, attn_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, attn_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, attn_heads, head_dim, device=device)

    if attn_type == "split_uniform":
        # start , end , step
        end_pt = 1601 if warmup else seq_len + 1
        cu_seq_lens = torch.arange(0, seq_len + 1, 64, dtype=torch.int32, device=device)
    elif attn_type == "split_diffres":
        if warmup:
            cu_seq_lens = torch.tensor([   0,   64,  128,  192,  256, 272, 288, 304, 320], device=device)
        else:
            cu_seq_lens = torch.tensor(list(range(0,5185,64)) + list(range(5200,6401,16)), device=device)
    else:
        cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    if islist:
        cu_seq_lens = list(cu_seq_lens.cpu().numpy())
    htcore.mark_step(sync=True) # make sure the tensors are created and ready on device
    gc_metric = metric_global("graph_compilation")
    print(f"-------- graph_compilation before start: ", gc_metric.stats()[0][1])
    outputs = []
    t1 = time.time()
    if rearrange_outside:
        q, k, v = (rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
    for i in range(1, len(cu_seq_lens)):
        start_idx = cu_seq_lens[i - 1]#.item()
        end_idx = cu_seq_lens[i]#.item()

        #print(f"-------- graph_compilation {i}: ", gc_metric.stats()[0][1])
        q_i = get_slice(q, index_type, start_idx, end_idx, rearrange_outside)
        k_i = get_slice(k, index_type, start_idx, end_idx, rearrange_outside)
        v_i = get_slice(v, index_type, start_idx, end_idx, rearrange_outside)
        #print(f"-------- graph_compilation {i}: ", gc_metric.stats()[0][1], q_i.shape)
        if not rearrange_outside:
            q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i])

        if device == "hpu":
            from habana_frameworks.torch.hpex.kernels import FusedSDPA

            output_i = FusedSDPA.apply(q_i, k_i, v_i, None, 0.0)
        else:
            output_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, dropout_p=0.0
            )

        if not rearrange_outside:
            output_i = rearrange(output_i, "b h s d -> b s h d ")
        outputs.append(output_i)

    context_layer = torch.cat(outputs, dim=2 if rearrange_outside else 1)
    if rearrange_outside:
        context_layer = rearrange(context_layer, "b h s d -> b s h d ")
    x = context_layer.sum().item()
    print('Final sum', x)
    t2 = time.time()
    if not warmup:
        golden = {"split_diffres": -2587.10302734375, "split_uniform": -2511.75732421875, "nosplit": -2134.07568359375}
        assert x == golden[attn_type], x # for seed set to 0
    # print(context_layer)
    print(f"attn_type: {attn_type}: Time Taken = {(t2 - t1) * 1000:.3f} ms")
    print(f'cu_seq_lens = {cu_seq_lens}')
    gc_metric = metric_global("graph_compilation")
    print(f"-------- graph_compilation at end: ", gc_metric.stats()[0][1])


# # Run the test
def main():
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("-a", "--attn_type", type=str, help="options for cu_seq_lens", choices=['split_uniform', 'split_diffres', 'nosplit'])
    parser.add_argument("-i", "--index_type", type=str, help="options for indexing", choices=['original', 'index_select', 'reshape'])
    parser.add_argument("-p", "--profile", action='store_true', help='Profile')
    parser.add_argument("-w", "--warmup_exact", action='store_true', help='warmup with exact shapes as actual run')
    parser.add_argument("-l", "--list", action='store_true', help='cu_seq_lens is a list, else its a tensor by default')
    parser.add_argument("-r", "--rearrange_outside", action='store_true', help='hoist rearrange out of loop')
    args = parser.parse_args()
    if args.attn_type == 'split_diffres' and args.index_type == 'reshape':
        assert False, 'These cant go together. Use split_uniform with reshape'
        # TODO implement reshape based slicing
    ctx = profile_ctx if args.profile else contextlib.nullcontext
    attention_split(args.attn_type, args.index_type, (not args.warmup_exact), args.list, args.rearrange_outside)  ## warmup
    with ctx() as p:
        attention_split(args.attn_type, args.index_type, False, args.list, args.rearrange_outside)
    print('Test done')

if __name__ == "__main__":
    main()


'''
We see recompilation in both these cases, which is not ideal
python attention_split.py -a split_diffres -i index_select
python attention_split.py -a split_diffres -i original

If we ran with exact shapes, ie if we set the True->False in the warmup run, we wouldnt get recompilations as expected, ie:
python attention_split.py -a split_diffres -i index_select -w



python attention_split.py -a split_diffres -i index_select -l
no recomps

python attention_split.py -a split_diffres -i original -l
no recomps

Fixed timing.. moved it after the sum()
 python attention_split.py -a split_diffres -i original -l
 1915.210 ms
 hoisted
 1285.474
'''