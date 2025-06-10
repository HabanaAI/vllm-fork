# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from functools import cache


def _kernel(name):

    def loader(fn):

        @cache
        def loader_impl():
            try:
                return fn()
            except (ImportError, AttributeError):
                from .utils import logger
                logger().warning(f"Could not import HPU {name} kernel. "
                                 "vLLM will use native implementation")

        return loader_impl

    return loader


@_kernel("FusedSDPA")
def fsdpa():
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    return FusedSDPA


@_kernel("FusedRMSNorm")
def rms_norm():
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
    return FusedRMSNorm


@_kernel("block_softmax_adjustment")
def block_softmax_adjustment():
    import torch
    return torch.ops.hpu.block_softmax_adjustment
