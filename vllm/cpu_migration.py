import habana_frameworks.torch as htorch
import torch

class CpuMigration:
    def __init__(self):
        self._migrate_to_cpu()

    def _do_nothing(self):
        print('check')
        pass

    def _migrate_to_cpu(self):
        htorch.core.mark_step = self._do_nothing
        torch.hpu.synchronize = self._do_nothing
        resource_name = "CPU"
