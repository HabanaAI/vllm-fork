import habana_frameworks.torch as htorch
import torch
import os

class HPUCompileConfig:
    def __init__(self):
        # Inicjalizacja konfiguracji kompilacji
        self.fullgraph = os.getenv('VLLM_T_COMPILE_FULLGRAPH', 'false').strip().lower() in ("1", "true")
        self.dynamic = os.getenv('VLLM_T_COMPILE_DYNAMIC_SHAPES', 'false').strip().lower() in ("1", "true")

    def get_compile_args(self):
        # Zwraca argumenty do przekazania do torch.compile()
        if self.dynamic:
            return {
                'backend': 'hpu_backend',
                'fullgraph': self.fullgraph,
                'options': {"force_static_compile": True}
            }
        else:
            return {
                'backend': 'hpu_backend',
                'fullgraph': self.fullgraph,
                'dynamic': False
            }
