import sys
import types
from vllm.utils import is_fake_hpu

if is_fake_hpu():
    print('\n\n\n FAKE_HPU \n\n\n')

class CpuMigration:
    def __init__(self):
        self._create_dummy_modules()
        self._migrate_to_cpu()
    
    def _create_dummy_modules(self):
        sys.modules['habana_frameworks'] = habana_frameworks = types.ModuleType('habana_frameworks')
        sys.modules['habana_frameworks.torch'] = habana_frameworks.torch = types.ModuleType('habana_frameworks.torch')

        sys.modules['habana_frameworks.torch.core'] = habana_frameworks.torch.core = types.ModuleType('habana_frameworks.torch.core')
        sys.modules['habana_frameworks.torch.utils'] = habana_frameworks.torch.utils = types.ModuleType('habana_frameworks.torch.utils')
        sys.modules['habana_frameworks.torch.utils.internal'] = habana_frameworks.torch.utils.internal = types.ModuleType('habana_frameworks.torch.utils.internal')

        habana_frameworks.torch.core = sys.modules['habana_frameworks.torch.core']
        habana_frameworks.torch.utils.internal = sys.modules['habana_frameworks.torch.utils.internal']

        habana_frameworks.torch.core.mark_step = lambda: print('calling mark_step')
        habana_frameworks.torch.utils.internal.is_lazy = lambda: print('calling is_lazy')

        import habana_frameworks.torch as htorch
        import torch

    def _do_nothing(self):
        pass

    def _return_false(self):
        return False

    def _migrate_to_cpu(self):
        htorch.core.mark_step = self._do_nothing
        htorch.utils.internal.is_lazy = self._return_false
        torch.hpu.synchronize = self._do_nothing
