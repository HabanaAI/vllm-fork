# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from typing import (Any, Callable, Dict, Generic, List, Optional, TextIO,
                    TypeVar, Union)

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import _maybe_force_spawn, get_mp_context, run_method

logger = init_logger(__name__)

T = TypeVar('T')

_TERMINATE = "TERMINATE"  # sentinel

# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'

JOIN_TIMEOUT_S = 2


@dataclass
class Result(Generic[T]):
    """Result of task dispatched to worker"""

    task_id: uuid.UUID
    value: Optional[T] = None
    exception: Optional[BaseException] = None


class ResultFuture(threading.Event, Generic[T]):
    """Synchronous future for non-async case"""

    def __init__(self):
        super().__init__()
        self.result: Optional[Result[T]] = None

    def set_result(self, result: Result[T]):
        self.result = result
        self.set()

    def get(self) -> T:
        self.wait()
        assert self.result is not None
        if self.result.exception is not None:
            raise self.result.exception
        return self.result.value  # type: ignore[return-value]


def _set_future_result(future: Union[ResultFuture, asyncio.Future],
                       result: Result):
    if isinstance(future, ResultFuture):
        future.set_result(result)
        return
    loop = future.get_loop()
    if not loop.is_closed():
        if result.exception is not None:
            loop.call_soon_threadsafe(future.set_exception, result.exception)
        else:
            loop.call_soon_threadsafe(future.set_result, result.value)


class ResultHandler(threading.Thread):
    """Handle results from all workers (in background thread)"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.result_queue = get_mp_context().Queue()
        self.tasks: Dict[uuid.UUID, Union[ResultFuture, asyncio.Future]] = {}

    def run(self):
        for result in iter(self.result_queue.get, _TERMINATE):
            future = self.tasks.pop(result.task_id)
            _set_future_result(future, result)
        # Ensure that all waiters will receive an exception
        for task_id, future in self.tasks.items():
            _set_future_result(
                future,
                Result(task_id=task_id,
                       exception=ChildProcessError("worker died")))

    def close(self):
        self.result_queue.put(_TERMINATE)


class WorkerMonitor(threading.Thread):
    """Monitor worker status (in background thread)"""

    def __init__(self, workers: List['ProcessWorkerWrapper'],
                 result_handler: ResultHandler):
        super().__init__(daemon=True)
        self.workers = workers
        self.result_handler = result_handler
        self._close = False

    def run(self) -> None:
        # Blocks until any worker exits
        dead_sentinels = wait([w.process.sentinel for w in self.workers])
        if not self._close:
            self._close = True

            # Kill / cleanup all workers
            for worker in self.workers:
                process = worker.process
                if process.sentinel in dead_sentinels:
                    process.join(JOIN_TIMEOUT_S)
                if process.exitcode is not None and process.exitcode != 0:
                    logger.error("Worker %s pid %s died, exit code: %s",
                                 process.name, process.pid, process.exitcode)
            # Cleanup any remaining workers
            if logger:
                logger.info("Killing local vLLM worker processes")
            for worker in self.workers:
                worker.kill_worker()
            # Must be done after worker task queues are all closed
            self.result_handler.close()

        for worker in self.workers:
            worker.process.join(JOIN_TIMEOUT_S)

    def close(self):
        if self._close:
            return
        self._close = True
        logger.info("Terminating local vLLM worker processes")
        for worker in self.workers:
            worker.terminate_worker()
        # Must be done after worker task queues are all closed
        self.result_handler.close()


class ProcessWorkerWrapper:
    """Local process wrapper for vllm.worker.Worker,
    for handling single-node multi-GPU tensor parallel."""

    def __init__(self, result_handler: ResultHandler,
                 worker_factory: Callable[[VllmConfig, int], Any],
                 vllm_config: VllmConfig, rank: int) -> None:
        self.mp = get_mp_context()
        self._task_queue = self.mp.Queue()
        self.result_queue = result_handler.result_queue
        self.tasks = result_handler.tasks
        self.process: BaseProcess = self.mp.Process(  # type: ignore[attr-defined]
            target=_run_worker_process,
            name="VllmWorkerProcess",
            kwargs=dict(
                worker_factory=worker_factory,
                task_queue=self._task_queue,
                result_queue=self.result_queue,
                vllm_config=vllm_config,
                rank=rank,
            ),
            daemon=True)

        self.process.start()

    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: Union[str, bytes], args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future
        try:
            self._task_queue.put((task_id, method, args, kwargs))
        except SystemExit:
            raise
        except BaseException as e:
            del self.tasks[task_id]
            raise ChildProcessError("worker died") from e

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        future: ResultFuture = ResultFuture()
        self._enqueue_task(future, method, args, kwargs)
        return future

    async def execute_method_async(self, method: Union[str, bytes], *args,
                                   **kwargs):
        future = asyncio.get_running_loop().create_future()
        self._enqueue_task(future, method, args, kwargs)
        return await future

    def terminate_worker(self):
        try:
            self._task_queue.put(_TERMINATE)
        except ValueError:
            self.process.kill()
        self._task_queue.close()

    def kill_worker(self):
        self._task_queue.close()
        self.process.kill()


def _run_worker_process(
    worker_factory: Callable[[VllmConfig, int], Any],
    task_queue: Queue,
    result_queue: Queue,
    vllm_config: VllmConfig,
    rank: int,
) -> None:
    """Worker process event loop"""

    # Add process-specific prefix to stdout and stderr
    process_name = get_mp_context().current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    # Initialize worker
    worker = worker_factory(vllm_config, rank)
    del worker_factory

    # Accept tasks from the engine in task_queue
    # and return task output in result_queue
    logger.info("Worker ready; awaiting tasks")
    try:
        for items in iter(task_queue.get, _TERMINATE):
            output = None
            exception = None
            task_id, method, args, kwargs = items
            try:
                output = run_method(worker, method, args, kwargs)
            except SystemExit:
                raise
            except KeyboardInterrupt:
                break
            except BaseException as e:
                logger.exception(
                    "Exception in worker %s while processing method %s.",
                    process_name, method)
                exception = e
            result_queue.put(
                Result(task_id=task_id, value=output, exception=exception))
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Worker failed")

    # Flush TunableOp results when TunableOp is enabled and
    # online (in situ) tuning is enabled.
    # Offline tuning API (record_untuned_is_enabled()) only
    # available in PyTorch 2.6 or later.
    if torch.cuda.is_available():
        import torch.cuda.tunable as tunable
        if (tunable.is_enabled() and tunable.tuning_is_enabled()
                and not tunable.record_untuned_is_enabled()):
            tunable.write_file()

    logger.info("Worker exiting")


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Prepend each output line with process-specific prefix"""

    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find('\n', idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]


def set_multiprocessing_worker_envs(parallel_config):
    """ Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent 
    process before worker processes are created"""

    _maybe_force_spawn()

    if (current_platform.is_hpu()
            and parallel_config.distributed_executor_backend == 'mp'
            and envs.VLLM_WORKER_MULTIPROC_METHOD == 'fork'):
        if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", None) is not None:
            logger.warning("On HPU, VLLM_WORKER_MULTIPROC_METHOD=fork might "
                           "cause application hangs on exit. Using "
                           "VLLM_WORKER_MULTIPROC_METHOD=fork anyway, "
                           "as it was explicitly requested.")
        else:
            logger.warning("On HPU, VLLM_WORKER_MULTIPROC_METHOD=fork might "
                           "cause application hangs on exit. Setting "
                           "VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
                           "To override that behavior, please set "
                           "VLLM_WORKER_MULTIPROC_METHOD=fork explicitly.")
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Configure thread parallelism if OMP_NUM_THREADS isn't set
    #
    # Helps to avoid CPU contention. The default of spawning a thread per
    # core combined with multiprocessing for each GPU can have a negative
    # impact on performance. The contention is amplified when running in a
    # container where CPU limits can cause throttling.
    default_omp_num_threads = 1
    if "OMP_NUM_THREADS" not in os.environ and (
            current_parallelism :=
            torch.get_num_threads()) > default_omp_num_threads:
        logger.warning(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism, default_omp_num_threads)
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        torch.set_num_threads(default_omp_num_threads)
