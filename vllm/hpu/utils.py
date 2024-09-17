###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from functools import wraps

import habana_frameworks.torch as htorch
import torch

from vllm.hpu.cache_ops import insert_or_update_cache


def with_mark_steps(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        htorch.core.mark_step()
        result = fn(*args, **kwargs)
        del args
        del kwargs
        htorch.core.mark_step()
        return result

    return wrapped


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


class VLLMKVCache(torch.nn.Module):

    def __init__(self):
        super(VLLMKVCache, self).__init__()

    def forward(self, input, cache, block_indices, block_offset):
        insert_or_update_cache(input, cache, block_indices, block_offset)
        return cache

    def fetch_from_cache(self, cache, blocks):
        return cache.index_select(0, blocks)


def process_run_characteristics(times, block_size, prefill=False):
    import statistics
    import math
    import pandas as pd
    
    def summarize_to_dict(d, block_size, prefill=False):
        ret = {}
        for k in d:
            l, graph = d[k]
            bs, seq = k
            mean_time = statistics.mean(l)
            ctx_len = bs*seq
            if not prefill: 
                ctx_len *= block_size 
            ret[k] = {"n": len(l), "mean_time": mean_time, 'time_stddev': statistics.stdev(l), 'mean_gen_tput': bs/mean_time,  'mean_in_tput':ctx_len/mean_time, 'hpugraph_captured':int(graph)} 
        return ret 
    
    summary_dict = summarize_to_dict(times, block_size, prefill)
    run_df = pd.DataFrame.from_dict(summary_dict,orient='index') 
    mode = 'decode' if not prefill else 'prefill'
    run_df.to_csv(f'vllm_soc_{mode}.csv')
    def plot_server_stats_df(df, prefill=False, scale=4):
        import matplotlib.pyplot as plt
        import seaborn as sns
        mode = 'decode' if not prefill else 'prefill'
        n = df['n'].iloc[0]
        plt.rcParams.update({'font.size': 6 * scale})
        fig, axs = plt.subplots(2, 2, tight_layout=True)
        fig.suptitle(f"Server operating characteristic ({mode}, n={n})", fontsize=16 * scale)
        sz = fig.get_size_inches()
        fig.set_size_inches(sz[0]*scale,sz[1]*scale, forward=True)
        numel = df['n'].count()
        annot_kws={"size": scale * 25 / math.sqrt(numel)}
        sns.heatmap(df['mean_time'].unstack()*1000, cmap='RdYlGn_r', ax=axs[0, 0], annot=True, cbar=False, fmt='.3f', square=True, annot_kws=annot_kws)
        axs[0, 0].set_title('mean time [ms]')
        sns.heatmap(df['time_stddev'].unstack()*1000, cmap='RdYlGn_r', ax=axs[0,1], annot=True, cbar=False, fmt='.2f',square=True, annot_kws=annot_kws, vmin=0)
        axs[0, 1].set_title('time stddev [ms]')
        tput_mode = 'gen' if not prefill else 'in'
        sns.heatmap(df[f'mean_{tput_mode}_tput'].unstack(), cmap='RdYlGn', ax=axs[1, 0], annot=True, cbar=False, fmt='.2f',square=True, annot_kws=annot_kws)
        axs[1, 0].set_title(f'mean {tput_mode} tput [tps]')
        sns.heatmap(df['hpugraph_captured'].unstack(), cmap='RdYlGn', ax=axs[1, 1], annot=True, cbar=False, square=True, annot_kws=annot_kws, vmin=0, vmax=1)
        axs[1, 1].set_title('HPUGraph captured')
        for ax in axs.flat:
            ax.set(xlabel='num blocks' if not prefill else 'seq len', ylabel='batch size')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        plt.savefig(f'vllm_soc_{mode}.png')
        plt.close(fig)

    plot_server_stats_df(run_df, prefill=prefill)
    
    return run_df