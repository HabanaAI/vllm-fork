# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Callable, Dict

import torch

import vllm.envs as envs

logger = logging.getLogger(__name__)

# make sure one process only loads plugins once
plugins_loaded = False


def load_plugins_by_group(group: str) -> Dict[str, Callable]:
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    allowed_plugins = envs.VLLM_PLUGINS

    discovered_plugins = entry_points(group=group)
    if len(discovered_plugins) == 0:
        logger.debug("No plugins for group %s found.", group)
        return {}
    logger.info("Available plugins for group %s:", group)
    for plugin in discovered_plugins:
        logger.info("name=%s, value=%s", plugin.name, plugin.value)
    if allowed_plugins is None:
        logger.info("all available plugins for group %s will be loaded.",
                    group)
        logger.info("set environment variable VLLM_PLUGINS to control"
                    " which plugins to load.")
    plugins = {}
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            try:
                func = plugin.load()
                plugins[plugin.name] = func
                logger.info("plugin %s loaded.", plugin.name)
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)
    return plugins


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global plugins_loaded
    if plugins_loaded:
        return
    plugins_loaded = True

    # some platform-specific configurations
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        # see https://github.com/pytorch/pytorch/blob/43c5f59/torch/_dynamo/config.py#L158
        torch._dynamo.config.disable = True
    elif current_platform.is_hpu():
        # NOTE(kzawora): PT HPU lazy backend (PT_HPU_LAZY_MODE = 1)
        # does not support torch.compile
        # Eager backend (PT_HPU_LAZY_MODE = 0) must be selected for
        # torch.compile support
        _environ = dict(os.environ)
        env_update_dict = {}
        try:
            # NOTE(kzawora) multi-HPU inference with HPUGraphs (lazy-only)
            # requires enabling lazy collectives
            # see https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html # noqa: E501
            # this does nothing for eager/t.compile
            os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = 'true'
            lazy_mode_env_var = os.environ('PT_HPU_LAZY_MODE', None)
            is_lazy = lazy_mode_env_var == '1'
            if lazy_mode_env_var is None:
                import habana_frameworks.torch as htorch
                is_lazy = htorch.utils.internal.is_lazy()
            if is_lazy:
                torch._dynamo.config.disable = True
                env_update_dict['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = 'true'
        finally:
            os.environ.clear()
            os.environ.update(_environ)
            os.environ.update(env_update_dict)
    plugins = load_plugins_by_group(group='vllm.general_plugins')
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():
        func()
