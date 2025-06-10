# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm_hpu_extension.config import Config
from vllm_hpu_extension.environment import get_environment
from vllm_hpu_extension.features import (get_experimental_flags, get_features,
                                         get_user_flags)
from vllm_hpu_extension.logger import logger

DETECTED = None


def filter_defined(config, keys):
    return {k: v for k, v in config.get_all(keys).items() if v is not None}


def dump(prefix, values):
    if values:
        padding = ' ' * 4
        logger().info(f'{prefix}:')
        for key, value in values.items():
            logger().info(f'{padding}{key}: {value}')


def get_config():

    global DETECTED

    if DETECTED:
        return DETECTED

    user_flags = get_user_flags()
    experimental_flags = get_experimental_flags()
    environment_values, environment_flags = get_environment()
    feature_values, feature_flags = get_features()

    experimental_flags = experimental_flags | environment_flags | feature_flags

    detected = Config(user_flags | experimental_flags | environment_values
                      | feature_values)
    detected.get_all()

    user_flags = filter_defined(detected, user_flags.keys())
    experimental_flags = filter_defined(detected, experimental_flags.keys())
    environment_values = filter_defined(detected, environment_values.keys())
    feature_values = filter_defined(detected, feature_values.keys())

    experimental_flag_names = experimental_flags.keys()
    if len(experimental_flag_names
           ) > 0 and not detected.VLLM_ENABLE_EXPERIMENTAL_FLAGS:
        asterisks = 48 * '*'
        header = f"{asterisks} Warning! {asterisks}"
        footer = '*' * len(header)
        logger().warning(header)
        logger().warning(
            f"Following environment variables are considered experimental: {', '.join(experimental_flag_names)}"
        )
        logger().warning(
            "In future releases using those flags without VLLM_ENABLE_EXPERIMENTAL_FLAGS will trigger a fatal error."
        )
        logger().warning(footer)

    dump('Environment', environment_values)
    dump('Features', feature_values)
    dump('User flags', user_flags)
    dump('Experimental flags', experimental_flags)

    DETECTED = detected

    return detected
