# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from unittest.mock import MagicMock

import pytest
import torch

from vllm.spec_decode.metrics import AsyncMetricsCollector


def test_initial_call_returns_none():
    """Expect first call to get metrics to return None.
    """
    spec_decode_sampler = MagicMock()
    spec_decode_sampler.num_accepted_tokens = torch.tensor(0,
                                                           dtype=torch.long,
                                                           device='cuda')
    spec_decode_sampler.num_emitted_tokens = torch.tensor(0,
                                                          dtype=torch.long,
                                                          device='cuda')
    spec_decode_sampler.num_draft_tokens = 0

    collector = AsyncMetricsCollector(spec_decode_sampler)
    collector.init_tensors(rank=0)
    maybe_metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert maybe_metrics is None


def test_second_call_returns_metrics():
    """Expect second call to not return None.
    """
    spec_decode_sampler = MagicMock()
    spec_decode_sampler.num_accepted_tokens = torch.tensor(0,
                                                           dtype=torch.long,
                                                           device='cuda')
    spec_decode_sampler.num_emitted_tokens = torch.tensor(0,
                                                          dtype=torch.long,
                                                          device='cuda')
    spec_decode_sampler.num_draft_tokens = 0

    collect_interval_s = 5.0
    timer = MagicMock()
    timer.side_effect = [
        0.0, collect_interval_s + 0.1, collect_interval_s + 0.2
    ]

    collector = AsyncMetricsCollector(spec_decode_sampler=spec_decode_sampler,
                                      timer=timer,
                                      collect_interval_s=collect_interval_s)
    collector.init_tensors(rank=0)
    _ = collector.maybe_collect_rejsample_metrics(k=5)
    metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert metrics is not None


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_nonzero_rank_noop(rank):
    """Verify nonzero ranks don't collect metrics.
    """
    spec_decode_sampler = MagicMock()
    spec_decode_sampler.num_accepted_tokens = torch.tensor(0,
                                                           dtype=torch.long,
                                                           device='cuda')
    spec_decode_sampler.num_emitted_tokens = torch.tensor(0,
                                                          dtype=torch.long,
                                                          device='cuda')
    spec_decode_sampler.num_draft_tokens = 0

    collector = AsyncMetricsCollector(spec_decode_sampler)
    collector.init_tensors(rank=rank)
    _ = collector.maybe_collect_rejsample_metrics(k=5)
    metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert metrics is None


def test_noop_until_time():
    """Verify metrics aren't collected until enough time passes.
    """
    spec_decode_sampler = MagicMock()
    spec_decode_sampler.num_accepted_tokens = torch.tensor(0,
                                                           dtype=torch.long,
                                                           device='cuda')
    spec_decode_sampler.num_emitted_tokens = torch.tensor(0,
                                                          dtype=torch.long,
                                                          device='cuda')
    spec_decode_sampler.num_draft_tokens = 0

    collect_interval_s = 5.0
    timer = MagicMock()
    timer.side_effect = [
        0.0, collect_interval_s - 0.1, collect_interval_s - 0.1,
        collect_interval_s + 0.1, collect_interval_s + 0.1
    ]

    collector = AsyncMetricsCollector(spec_decode_sampler=spec_decode_sampler,
                                      timer=timer,
                                      collect_interval_s=collect_interval_s)
    collector.init_tensors(rank=0)

    _ = collector.maybe_collect_rejsample_metrics(k=5)
    metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert metrics is None

    _ = collector.maybe_collect_rejsample_metrics(k=5)
    metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert metrics is not None


def test_timer_is_reset():
    """Verify that the internal timer inside AsyncMetricsCollector
    is reset after collection.
    """
    spec_decode_sampler = MagicMock()
    spec_decode_sampler.num_accepted_tokens = torch.tensor(0,
                                                           dtype=torch.long,
                                                           device='cuda')
    spec_decode_sampler.num_emitted_tokens = torch.tensor(0,
                                                          dtype=torch.long,
                                                          device='cuda')
    spec_decode_sampler.num_draft_tokens = 0

    collect_interval_s = 5.0
    timer = MagicMock()
    timer.side_effect = [
        0.0,
        collect_interval_s + 0.1,
        collect_interval_s + 0.1,
        collect_interval_s + 0.2,
        collect_interval_s + 0.2,
        2 * collect_interval_s + 0.1,
        2 * collect_interval_s + 0.1,
    ]

    collector = AsyncMetricsCollector(spec_decode_sampler=spec_decode_sampler,
                                      timer=timer,
                                      collect_interval_s=collect_interval_s)
    collector.init_tensors(rank=0)

    _ = collector.maybe_collect_rejsample_metrics(k=5)
    metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert metrics is not None

    _ = collector.maybe_collect_rejsample_metrics(k=5)
    metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert metrics is None

    _ = collector.maybe_collect_rejsample_metrics(k=5)
    metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert metrics is not None


@pytest.mark.parametrize("has_data", [True, False])
def test_initial_metrics_has_correct_values(has_data: bool):
    """Test correctness of metrics data.
    """
    if has_data:
        num_accepted_tokens = 103
        num_emitted_tokens = 104
        num_draft_tokens = 105
    else:
        num_accepted_tokens = 0
        num_emitted_tokens = 0
        num_draft_tokens = 0
    k = 5

    max_num_emitted_tokens = AsyncMetricsCollector.get_max_num_emitted_tokens(
        num_draft_tokens, k)

    spec_decode_sampler = MagicMock()
    spec_decode_sampler.num_accepted_tokens = torch.tensor(num_accepted_tokens,
                                                           dtype=torch.long,
                                                           device='cuda')
    spec_decode_sampler.num_emitted_tokens = torch.tensor(num_emitted_tokens,
                                                          dtype=torch.long,
                                                          device='cuda')
    spec_decode_sampler.num_draft_tokens = num_draft_tokens

    collect_interval_s = 5.0
    timer = MagicMock()
    timer.side_effect = [
        0.0, collect_interval_s + 0.1, collect_interval_s + 0.2
    ]

    collector = AsyncMetricsCollector(spec_decode_sampler=spec_decode_sampler,
                                      timer=timer,
                                      collect_interval_s=collect_interval_s)
    collector.init_tensors(rank=0)
    _ = collector.maybe_collect_rejsample_metrics(k)
    metrics = collector.maybe_collect_rejsample_metrics(k)

    assert metrics.num_spec_tokens == k
    assert metrics.accepted_tokens == num_accepted_tokens
    assert metrics.draft_tokens == num_draft_tokens
    assert metrics.emitted_tokens == num_emitted_tokens

    if has_data:
        assert (metrics.draft_acceptance_rate == num_accepted_tokens /
                num_draft_tokens)
        assert (metrics.system_efficiency == num_emitted_tokens /
                max_num_emitted_tokens)
    else:
        assert math.isnan(metrics.draft_acceptance_rate)
        assert math.isnan(metrics.system_efficiency)
