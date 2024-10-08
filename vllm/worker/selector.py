from vllm.config import DeviceConfig

from vllm.worker.worker import Worker
from vllm.worker.hpu_worker import HPUWorker

def init_worker(*args, **kwargs):
    device_config: DeviceConfig = kwargs.get("device_config")
    if device_config.device_type == 'cuda':
        return Worker(*args, **kwargs)
    elif device_config.device_type == 'hpu':
        return HPUWorker(*args, **kwargs)
    else:
        raise NotImplementedError("Please help to add your preferred backend")