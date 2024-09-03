import sys
import types

# Create dummy habana_frameworks
habana_frameworks = sys.modules['habana_frameworks'] = types.ModuleType('habana_frameworks')
torch = sys.modules['habana_frameworks.torch'] = types.ModuleType('habana_frameworks.torch')
core = sys.modules['habana_frameworks.torch.core'] = types.ModuleType('habana_frameworks.torch.core')

habana_frameworks.torch = torch
torch.core = core
core.mark_step = lambda: print('calling mark_step')

import habana_frameworks.torch as htorch
import torch

# torch.hpu = sys.modules['torch.hpu'] = types.ModuleType('torch.hpu')
# torch.hpu.synchronize = lambda: print('calling synchronize')

class CpuMigration:
    def __init__(self):
        self._migrate_to_cpu()

    def _do_nothing(self):
        pass

    def _return_false(self):
        return False

    def _migrate_to_cpu(self):
        htorch.core.mark_step = self._do_nothing
        torch.hpu.synchronize = self._do_nothing
