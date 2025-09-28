import os
import random
import time
import math
import itertools
from numbers import Number
from typing import List, Optional

import torch
from vllm.platforms import current_platform
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser,
                        create_kv_caches_with_random)

if current_platform.is_hpu:
    import habana_frameworks.torch as ht# Gaudi
    from vllm_hpu_extension.ops import batch2block, block2batch
    from vllm_hpu_extension.utils import Matmul, Softmax, VLLMKVCache
    from vllm_hpu_extension import cache_ops, ops
else: 
    from vllm import _custom_ops as ops


NUM_BLOCKS = 1024
PARTITION_SIZE = 512

# Adapting with 1.20
import vllm_hpu_extension.environment as environment

class HfConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type

class Config:
    def __init__(self, model_type: str):
        self.hf_config = HfConfig(model_type=model_type)

# Create an instance of the config
cfg = Config(model_type="llama3")

environment.set_model_config(cfg)


def flatten(in_list):
    return list(itertools.chain(*in_list))

def gather_list(input, indices, v):
    return [input[i] if i is not None else v for i in indices]

def set_flat_pa_inputs(num_seqs, seq_len, block_size, dtype, device): 
    
    max_num_blocks_per_seq = math.ceil(seq_len / block_size)
    
    # block_tables
    block_tables = []
    start_idx = 0
    idx = num_seqs * max_num_blocks_per_seq + 1
    for s in range( num_seqs):
        block_table = [i for i in range(start_idx, start_idx + max_num_blocks_per_seq)]
        start_idx += max_num_blocks_per_seq        
        block_tables.append(block_table)
    # block_list
    block_list = flatten(block_tables)
    
    # block_groups
    block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
    block_groups = flatten(block_groups) 
 
    VLLM_DECODE_BLOCK_BUCKET_STEP = 128
    #print(f"block_list len {len(block_list)} {block_list}")
    block_bucket_size = math.ceil(len(block_list) / VLLM_DECODE_BLOCK_BUCKET_STEP) * VLLM_DECODE_BLOCK_BUCKET_STEP
    #print(f"block_bucket_size {block_bucket_size}")
    indices: List[Any]
    indices = [None] * block_bucket_size
    for i, bid in enumerate(block_list):
        indices[bid-1] = i
    
    padding_fn = lambda tensor, pad_value: gather_list(
        tensor, indices, pad_value)    

    block_list = padding_fn(block_list, 0)
    block_groups = padding_fn(block_groups, -1)
 
    block_groups = torch.tensor(block_groups, dtype=torch.int, device=device)
    block_mapping = torch.nn.functional.relu(block_groups)
    block_mapping = torch.nn.functional.one_hot(block_mapping.to(torch.long),
                                                num_classes=num_seqs)
    
    block_bias = torch.zeros((block_bucket_size, block_size), dtype=dtype, device=device)
    
    # todo: here is eager mode, enable lazy mode
    # will it improve performance?
    lazy_mode = os.environ.get('PT_HPU_LAZY_MODE', 1)
    print(f"lazy_mode {lazy_mode}")
    #if not lazy_mode:
    oob_values = block_groups.lt(0)
    block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
    block_groups.masked_fill_(oob_values, num_seqs)
    #print(f"oob_values {oob_values}")
    
    # block_scales
    ones = torch.ones((block_mapping.size(0), ),
                  device=device,
                  dtype=torch.long)
    sums = batch2block(block2batch(ones, block_mapping), block_mapping)
    block_scales = torch.reciprocal(torch.maximum(ones, sums))
    block_scales = block_scales.to(dtype)

    block_list = torch.tensor(block_list, dtype=torch.int, device=device)
    block_mapping = block_mapping.to(dtype)
   
    '''
    print(f"block_list {block_list}")
    print(f"block_mapping {block_mapping}")
    print(f"block_bias {block_bias}")
    print(f"block_scales {block_scales}")
    print(f"block_groups {block_groups}")
    '''
    
    return block_list, block_mapping, block_bias, block_scales, block_groups

@torch.inference_mode()
def main_hpu(
    version: str,
    num_seqs: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "hpu",
    kv_cache_dtype: Optional[str] = None,
) -> None:
            
    max_num_blocks_per_seq = math.ceil(seq_len / block_size)
    
    query_shape =(num_seqs, num_query_heads, head_size)
    #query_shape =(num_seqs, 1, head_size * num_query_heads)
     
    #
    # Set Key Value Cache
    # default block num as 8k
    hpu_num_blocks = 8192 * 128 // block_size
    kv_cache_shape = (hpu_num_blocks, block_size, num_kv_heads, head_size)
    k_cache = VLLMKVCache()
    v_cache = VLLMKVCache()
    
    key_cache = torch.empty(kv_cache_shape, dtype=dtype, device=device)        
    value_cache = torch.empty(kv_cache_shape, dtype=dtype, device=device)
    
    kv_shape =(max_num_blocks_per_seq * num_seqs,  block_size, num_kv_heads, head_size)
    key  = torch.rand(kv_shape, dtype=dtype, device=device)
    value = torch.rand(kv_shape, dtype=dtype, device=device)

    max_num_blocks = max_num_blocks_per_seq * num_seqs
    block_indices = [i for i in range(1, max_num_blocks + 1)]
    block_indices = torch.tensor(block_indices, dtype=torch.long, device=device)
    block_offsets = None
    key_cache = k_cache(key, key_cache, block_indices, block_offsets)
    value_cache = v_cache(value, value_cache, block_indices, block_offsets)    
    
    #
    # Decode
    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(query_shape, dtype=dtype, device=device)
    query.uniform_(-scale, scale)    

    kv_shape =(num_seqs, num_kv_heads, head_size)
    key  = torch.rand(kv_shape, dtype=dtype, device=device)
    value = torch.rand(kv_shape, dtype=dtype, device=device)
    block_indices = [ 1 for _ in range(num_seqs)]
    block_offsets = [ 1 for _ in range(num_seqs)]
    block_indices = torch.tensor(block_indices, dtype=torch.long, device=device)
    block_offsets = torch.tensor(block_offsets, dtype=torch.long, device=device)

    key_cache = k_cache(key, key_cache, block_indices, block_offsets)
    value_cache = v_cache(value, value_cache, block_indices, block_offsets)
    
    block_list, block_mapping, block_bias, block_scales, block_groups = set_flat_pa_inputs(num_seqs, seq_len, block_size, dtype, device)
         
    # flat_pa parameters
    matmul_qk_op = Matmul()
    matmul_av_op = Matmul()
    batch2block_matmul_op = Matmul()
    block2batch_matmul_op = Matmul()
    keys_fetch_func = k_cache.fetch_from_cache
    values_fetch_func = v_cache.fetch_from_cache
        
    #print(f"block_list {block_list}")
    #print(f"block_mapping {block_mapping}")
    #print(f"block_bias {block_bias}")
    #print(f"block_scales {block_scales}")
    #print(f"block_groups {block_groups}")
    
    # Run benchmark
    def run_hpu_benchmark(num_iters: int, profile: bool = False) -> float:
        # Warmup
        for _ in range(10):
            ops.flat_pa(query, key_cache, value_cache, block_list, block_mapping,
                    block_bias, block_groups, scale, matmul_qk_op,
                    matmul_av_op, batch2block_matmul_op, block2batch_matmul_op,
                    keys_fetch_func, values_fetch_func)

        torch.hpu.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iters): 
            outs = ops.flat_pa(query, key_cache, value_cache, block_list, block_mapping,
                    block_bias, block_groups, scale, matmul_qk_op,
                    matmul_av_op, batch2block_matmul_op, block2batch_matmul_op,
                    keys_fetch_func, values_fetch_func)
        torch.hpu.synchronize()
        end_time = time.perf_counter()
        
        return (end_time - start_time) / num_iters
        
    def run_hpu_graph_benchmark(num_iters: int, profile: bool = False) -> float:          
        g = ht.hpu.HPUGraph()
        s = ht.hpu.Stream()
        with ht.hpu.stream(s):
            g.capture_begin()
            print("query.shape: ", query.shape)
            print("key_cache.shape: ", key_cache.shape)
            print("value_cache.shape: ", value_cache.shape)
            print("block_list.shape: ", block_list.shape)
            print("block_mapping.shape: ", block_mapping.shape)
            print("block_bias.shape: ", block_bias.shape)
            print("block_groups.shape: ", block_groups.shape)
            print("scale: ", scale)
            output = ops.flat_pa(query, key_cache, value_cache, block_list, block_mapping,
                        block_bias, None, block_groups, scale, matmul_qk_op,
                        matmul_av_op, batch2block_matmul_op, block2batch_matmul_op,
                        keys_fetch_func, values_fetch_func)
            g.capture_end()
        
        # Warmup
        for _ in range(10):
            g.replay()
            
        torch.hpu.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iters): 
            g.replay()
        torch.hpu.synchronize()
        end_time = time.perf_counter()      
        return (end_time - start_time) / num_iters
    
    ENABLE_HPU_GRAPH = os.environ.get('ENABLE_HPU_GRAPH', 1)
    print(f"ENABLE_HPU_GRAPH = {ENABLE_HPU_GRAPH}")
    print("Benchmarking...")
    if ENABLE_HPU_GRAPH == 1:
        print("run at hpu graph mode")
        latency = run_hpu_graph_benchmark(num_iters=100, profile=False)
    else:
        latency = run_hpu_benchmark(num_iters=100, profile=False)            
    print(f"Kernel running time: {latency * 1000000:.3f} us")
    
@torch.inference_mode()
def main(
    version: str,
    num_seqs: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "cuda",
    kv_cache_dtype: Optional[str] = None,
) -> None:
    current_platform.seed_everything(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device=device)

    seq_lens = [seq_len for _ in range(num_seqs)]
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int, device=device)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: List[List[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst,
                                dtype=torch.int,
                                device=device)

    # Create the KV cache.
    key_caches, value_caches = create_kv_caches_with_random(NUM_BLOCKS,
                                                            block_size,
                                                            1,
                                                            num_kv_heads,
                                                            head_size,
                                                            kv_cache_dtype,
                                                            dtype,
                                                            device=device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v2":
        num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        tmp_output = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        # Using default kv_scale
        k_scale = v_scale = 1.0

        for _ in range(num_iters):
            if version == "v1":
                ops.paged_attention_v1(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    num_kv_heads,
                    scale,
                    block_tables,
                    seq_lens,
                    block_size,
                    max_seq_len,
                    alibi_slopes,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                )
            elif version == "v2":
                ops.paged_attention_v2(
                    output,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    num_kv_heads,
                    scale,
                    block_tables,
                    seq_lens,
                    block_size,
                    max_seq_len,
                    alibi_slopes,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                )
            else:
                raise ValueError(f"Invalid version: {version}")
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--device",
                        type=str,
                        choices=["hpu", "cuda"],
                        default="cuda")
    parser.add_argument("--version",
                        type=str,
                        choices=["v1", "v2"],
                        default="v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 120, 128, 192, 256],
                        default=128)
    parser.add_argument("--block-size", type=int, choices=[16, 32, 128, 256, 512], default=16)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        help="Data type for kv cache storage. If 'auto', will use model "
        "data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. "
        "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)")
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    
    if args.device == 'hpu':
        main_hpu(
            version=args.version,
            num_seqs=args.batch_size,
            seq_len=args.seq_len,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            use_alibi=args.use_alibi,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            seed=args.seed,
            do_profile=args.profile,
            kv_cache_dtype=args.kv_cache_dtype,
        )
    else:  
        main(
            version=args.version,
            num_seqs=args.batch_size,
            seq_len=args.seq_len,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            use_alibi=args.use_alibi,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            seed=args.seed,
            do_profile=args.profile,
            kv_cache_dtype=args.kv_cache_dtype,
        )
