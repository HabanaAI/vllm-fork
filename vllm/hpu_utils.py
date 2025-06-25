# SPDX-License-Identifier: Apache-2.0
import os


# Get Intel HPU arguments to be passed to torch compile
class HPUCompileConfig:

    def __init__(self):
        self.fullgraph = os.getenv('VLLM_T_COMPILE_FULLGRAPH',
                                   'false').strip().lower() in ("1", "true")
        self.dynamic = os.getenv('VLLM_T_COMPILE_DYNAMIC_SHAPES',
                                 'false').strip().lower() in ("1", "true")

    def get_compile_args(self):
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
