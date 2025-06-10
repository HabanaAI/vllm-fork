# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from .config import Value, boolean, choice, split_values_and_flags
from .logger import logger

_VLLM_VALUES = {}


def _get_hw(_):
    import habana_frameworks.torch.utils.experimental as htexp
    device_type = htexp._get_device_type()
    match device_type:
        case htexp.synDeviceType.synDeviceGaudi2:
            return "gaudi2"
        case htexp.synDeviceType.synDeviceGaudi3:
            return "gaudi3"
    from vllm_hpu_extension.utils import is_fake_hpu
    if is_fake_hpu():
        return "cpu"
    logger().warning(f'Unknown device type: {device_type}')
    return None


def _get_build(_):
    import re
    import subprocess
    output = subprocess.run("pip show habana-torch-plugin",
                            shell=True,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    version_re = re.compile(r'Version:\s*(?P<version>.*)')
    match = version_re.search(output.stdout)
    if output.returncode == 0 and match:
        return match.group('version')
    # In cpu-test environment we don't have access to habana-torch-plugin
    from vllm_hpu_extension.utils import is_fake_hpu
    result = '0.0.0.0' if is_fake_hpu() else None
    logger().warning(f"Unable to detect habana-torch-plugin version! Returning: {result}")
    return result


def set_vllm_config(cfg):
    global _VLLM_VALUES

    if hasattr(cfg.model_config, 'hf_config'):
        _VLLM_VALUES['model_type'] = cfg.model_config.hf_config.model_type
    else:
        _VLLM_VALUES['model_type'] = cfg.model_config.model_type
    _VLLM_VALUES['prefix_caching'] = cfg.cache_config.enable_prefix_caching

    # t.compile is very picky about what functions we can call inside modules
    # since this is the last step we can force recompilation of config to
    # ensure all values are computed before entering the model
    from vllm_hpu_extension.runtime import get_config
    get_config().finalize()


def _get_vllm_engine_version(_):
    import vllm.envs as envs
    return 'v1' if envs.VLLM_USE_V1 else 'v0'


def _get_pt_bridge_mode(_):
    import habana_frameworks.torch as htorch
    return 'lazy' if htorch.utils.internal.is_lazy() else 'eager'


def VllmValue(name, env_var_type):
    global _VLLM_VALUES
    return Value(name, lambda _: _VLLM_VALUES.get(name, None), env_var_type=env_var_type)


def get_environment():
    values = [
        Value('hw', _get_hw, env_var_type=choice('cpu', 'gaudi', 'gaudi2', 'gaudi3')),
        Value('build', _get_build, env_var_type=str),
        Value('engine_version', _get_vllm_engine_version, env_var_type=str),
        Value('bridge_mode', _get_pt_bridge_mode, env_var_type=choice('eager', 'lazy')),
        VllmValue('model_type', str),
        VllmValue('prefix_caching', boolean),
    ]
    return split_values_and_flags(values)
