# SPDX-License-Identifier: Apache-2.0
import os
import torch._dynamo as dynamo

# Get Intel HPU arguments to be passed to torch compile
class HPUCompileConfig:

    def __init__(self, num_layers, is_quantized_with_inc : bool = False):
        self.fullgraph = os.getenv('VLLM_T_COMPILE_FULLGRAPH',
                                   'false').strip().lower() in ("1", "true")
        self.dynamic = os.getenv('VLLM_T_COMPILE_DYNAMIC_SHAPES',
                                 'false').strip().lower() in ("1", "true")

        # Check if FP8  is enabled and then set float specialization
        if is_quantized_with_inc:
            dynamo.config.specialize_float = True
            dynamo.config.cache_size_limit = num_layers * dynamo.config.cache_size_limit

        # modify size of dynamo cache

    def get_compile_args(self):

        print("===> DYNAMO CONFIG CACHE SIZE LIMIT: {}",dynamo.config.cache_size_limit)
        print("===> DYNAMO CONFIG ACCUMULATED CACHE SIZE LIMIT: {}",dynamo.config.accumulated_cache_size_limit)

        if self.dynamic:
            return {
                'backend': 'hpu_backend',
                'fullgraph': self.fullgraph,
                'options': {
                    "force_static_compile": True
                }
            }
        else:
            return {
                'backend': 'hpu_backend',
                'fullgraph': self.fullgraph,
                'dynamic': False
            }


    def set_dynamo_cache_limits(self, num_layers, prompt_buckets, decode_buckets):
        multiplier = 3 if os.getenv('VLLM_REGIONAL_COMPILATION',
                                    'true').lower() == 'true' else 1
        # If we have specialized float, we need to increase the cache size limit
        if dynamo.config.specialize_float:
            multiplier *= num_layers
        cache_size_limit = 1 + multiplier * (prompt_buckets +
                                             decode_buckets)
        dynamo.config.cache_size_limit = max(
            cache_size_limit, dynamo.config.cache_size_limit)
        # Multiply by 8 to follow the original default ratio between
        # the cache_size_limit and accumulated_cache_size_limit
        dynamo.config.accumulated_cache_size_limit = max(
            cache_size_limit * 8,
            dynamo.config.accumulated_cache_size_limit)
        print(f"Setting dynamo cache size limits to {dynamo.config.cache_size_limit} and accumulated cache size limits to {dynamo.config.accumulated_cache_size_limit}")


