# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from functools import cache


@cache
def logger():
    try:
        from vllm.logger import init_logger
        return init_logger("vllm")
    except ImportError:
        import logging
        return logging.getLogger("vllm")
