# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cleanup_dist_env_and_memory robustness."""
from unittest.mock import MagicMock, patch


def test_cleanup_handles_runtime_error_in_empty_cache():
    """Verify cleanup doesn't crash if empty_cache raises RuntimeError.

    This is a regression test for HPU/Gaudi2 where the allocator
    doesn't implement the DeviceAllocator interface.
    """
    failing_empty_cache = MagicMock(
        side_effect=RuntimeError(
            "Allocator for hpu is not a DeviceAllocator"
        )
    )
    mock_platform = MagicMock()
    mock_platform.empty_cache = failing_empty_cache
    mock_platform.is_cpu.return_value = False

    with patch(
        "vllm.distributed.parallel_state.destroy_model_parallel"
    ), patch(
        "vllm.distributed.parallel_state"
        ".destroy_distributed_environment"
    ), patch(
        "torch.distributed.destroy_process_group"
    ), patch(
        "vllm.distributed.parallel_state.gc.collect"
    ), patch(
        "vllm.distributed.parallel_state.current_platform",
        mock_platform,
    ), patch(
        "torch._C._host_emptyCache",
        side_effect=RuntimeError(
            "Allocator for hpu is not a DeviceAllocator"
        ),
    ):
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
        )
        # Should not raise
        cleanup_dist_env_and_memory()


def test_cleanup_handles_attribute_error_in_host_empty_cache():
    """Verify cleanup handles missing _host_emptyCache (PyTorch < 2.5).
    """
    mock_platform = MagicMock()
    mock_platform.empty_cache = None
    mock_platform.is_cpu.return_value = False

    with patch(
        "vllm.distributed.parallel_state.destroy_model_parallel"
    ), patch(
        "vllm.distributed.parallel_state"
        ".destroy_distributed_environment"
    ), patch(
        "torch.distributed.destroy_process_group"
    ), patch(
        "vllm.distributed.parallel_state.gc.collect"
    ), patch(
        "vllm.distributed.parallel_state.current_platform",
        mock_platform,
    ), patch(
        "torch._C._host_emptyCache",
        side_effect=AttributeError("not available"),
    ):
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
        )
        # Should not raise
        cleanup_dist_env_and_memory()


def test_cleanup_succeeds_on_cpu_platform():
    """Verify cleanup skips host cache clearing on CPU."""
    mock_platform = MagicMock()
    mock_platform.empty_cache = None
    mock_platform.is_cpu.return_value = True

    with patch(
        "vllm.distributed.parallel_state.destroy_model_parallel"
    ), patch(
        "vllm.distributed.parallel_state"
        ".destroy_distributed_environment"
    ), patch(
        "torch.distributed.destroy_process_group"
    ), patch(
        "vllm.distributed.parallel_state.gc.collect"
    ), patch(
        "vllm.distributed.parallel_state.current_platform",
        mock_platform,
    ):
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
        )
        # Should not raise
        cleanup_dist_env_and_memory()
