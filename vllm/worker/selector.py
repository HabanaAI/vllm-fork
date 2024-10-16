from vllm.config import DeviceConfig

def init_worker(*args, **kwargs):
    device_config: DeviceConfig = kwargs.get("device_config")
    if device_config.device_type == 'neuron':
        from vllm.worker.neuron_worker import NeuronWorker
        return NeuronWorker(*args, **kwargs)
    elif device_config.device_type == 'tpu':
        from vllm.worker.tpu_worker import TPUWorker
        return TPUWorker(*args, **kwargs)
    elif device_config.device_type == 'cpu':
        from vllm.worker.cpu_worker import CPUWorker
        return CPUWorker(*args, **kwargs)
    elif device_config.device_type == 'hpu':
        from vllm.worker.hpu_worker import HPUWorker
        return HPUWorker(*args, **kwargs)
    elif device_config.device_type == 'openvino':
        from vllm.worker.openvino_worker import OpenVINOWorker
        return OpenVINOWorker(*args, **kwargs)
    elif device_config.device_type == 'xpu':
        from vllm.worker.xpu_worker import XPUWorker
        return XPUWorker(*args, **kwargs)
    else:
        from vllm.worker.worker import Worker
        return Worker(*args, **kwargs)