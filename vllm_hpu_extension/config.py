# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import itertools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union

from packaging.specifiers import SpecifierSet
from packaging.version import Version


class Config:
    """ Contains pairs of key/value that can be calculated on demand"""

    def __init__(self, *sources: List[Dict[str, Any]], **extra: Dict[str, Any]):
        self._data = dict(itertools.chain(*[v.items() for v in sources] + [extra.items()]))

    def __getattr__(self, key: str):
        """Allow conveniently querying keys by using dot notation"""
        return self.get(key)

    def __bool__(self) -> bool:
        """Check if config is not empty"""
        return bool(self._data)

    def get(self, key: str):
        """Get key from internal structure, triggering calculation if needed"""
        assert key in self._data, f'Unknown run-time configuration parameter: {key}!'
        value = self._data.get(key)
        if callable(value):
            value = value(self)
            self._data[key] = value
        return value

    def get_all(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return a dict with a subset of keys"""
        keys = keys or self._data.keys()
        return {k: self.get(k) for k in keys}

    def finalize(self) -> None:
        """Trigger calculating all values"""
        self.get_all()


ValueFn: TypeAlias = Callable[Config, Any]


def All(*parts: List[ValueFn]) -> ValueFn:
    """Return True if all functions return True"""
    return lambda cfg: all(p(cfg) for p in parts)


def Not(fn: ValueFn) -> ValueFn:
    """Negate function result"""
    return lambda cfg: not fn(cfg)


def Eq(key: str, value: Any) -> ValueFn:
    """Return True when config[key] == value"""
    return lambda cfg: (cfg.get(key) == value)


def Enabled(key: str) -> ValueFn:
    """Return True when config[key] == True"""
    return Eq(key, True)


def Disabled(key: str) -> ValueFn:
    """Return True when config[key] != True"""
    return Not(Enabled(key))


def FirstEnabled(*keys: List[str]) -> ValueFn:
    """Return first key for which config[key] is True"""
    return lambda cfg: next((k for k in keys if cfg.get(k)), None)


def Lazy() -> ValueFn:
    """Return True if pt bridge is running in lazy mode"""
    return Eq('bridge_mode', 'lazy')


def Hardware(target_hw: str) -> ValueFn:
    """Return True if current hardware == target_hw"""
    return Eq('hw', target_hw)


def Kernel(loader_fn: Callable) -> ValueFn:
    """Return True if loader_fn result is not None and hardware != 'cpu'"""
    def kernel_exists(_):
        return loader_fn() is not None
    return All(kernel_exists, Not(Hardware('cpu')))


def ModelType(target_model_type: str) -> ValueFn:
    """Return True if current model_type == target_model_type"""
    return Eq('model_type', target_model_type)


def Engine(target_engine_version: str) -> ValueFn:
    """Return True if current engine_version == target_engine_version"""
    return Eq('engine_version', target_engine_version)


def VersionRange(*specifiers: List[str]) -> ValueFn:
    """Return True if any of the version specifiers matches current build"""
    specifiers = [SpecifierSet(s) for s in specifiers]
    return lambda cfg: any(Version(cfg.build) in s for s in specifiers)


def choice(*options: List[Any]) -> Callable[Any, Any]:
    """Validates if input is one of the available choices and returns it unchanged"""
    def choice_impl(x):
        assert x in options, f'{x} is not in allowed options: {options}!'
        return x
    return choice_impl


def boolean(x: str) -> Callable[str, bool]:
    """Converts string representation of a bool to its value"""
    return x.lower() in ['true', 't', '1', 'yes', 'y', 'on']


class Env:
    """A callable that fetches values from env variables, applying conversions if necessary"""

    def __init__(self, name: str, value_type: Callable[Any, Any]):
        self.name = name
        self.value_type = value_type

    def __call__(self, _):
        value = os.environ.get(self.name)
        if value is not None:
            try:
                return self.value_type(value)
            except Exception as e:
                msg = f'{self.name}: exception during construction: {e}'
                raise RuntimeError(msg)
        return None


class Value:
    """A callable that returns the value calculated through its dependencies or overriden by an associated experimental flag"""

    def __init__(self, name: str, dependencies: Any, env_var: str = None, env_var_type: Callable[Any, Any] = boolean):
        self.name = name
        self.env_var = env_var if env_var is not None else 'VLLM_' + name.upper()
        self.env_var_type = env_var_type
        self.dependencies = dependencies

    def to_env_flag(self) -> ValueFn:
        """ Return associated experimental flag"""
        return Env(self.env_var, self.env_var_type)

    def __call__(self, config):
        """ Return value from experimental flag if provided by user or calculate it based on dependencies"""
        if (override := config.get(self.env_var)) is not None:
            return override
        if callable(self.dependencies):
            return self.dependencies(config)
        return self.dependencies


def to_dict(collection: List[Union[Value, Env]]) -> Dict[str, Union[Value, Env]]:
    """Convert a list values/envs to a dict"""
    return {c.name: c for c in collection}


def env_flags(values: List[Value]) -> List[Env]:
    """Extract associated env flags from values"""
    return [v.to_env_flag() for v in values]


def split_values_and_flags(values: List[Value]) -> Tuple[Dict[str, Value], Dict[str, Env]]:
    """Converts a list of values and returns dicts for both values and envs"""
    return to_dict(values), to_dict(env_flags(values))
