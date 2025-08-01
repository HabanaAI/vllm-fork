# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import contextlib
import gc
import os
import queue
from typing import List, Optional, Set, Tuple, Type

import habana_frameworks.torch as htorch  # noqa:F401
import torch
import torch.distributed
from vllm_hpu_extension.profiler import HabanaMemoryProfiler, format_bytes

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized, get_pp_group,
                              init_distributed_environment)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (bind_kv_cache, hpu_backend_string, hpu_device_string,
                        is_fake_hpu)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.hpu_enc_dec_model_runner import HPUEncoderDecoderModelRunner
from vllm.worker.hpu_model_runner import HPUModelRunner, HPUModelRunnerBase
from vllm.worker.hpu_pooling_model_runner import HPUPoolingModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)

logger = init_logger(__name__)


class HPUWorker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a HPU.

    Each worker is associated with a single HPU. The worker is responsible for
    maintaining the KV cache and executing the model on the HPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[HPUModelRunner]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.hf_config.model_type \
                == model_config.hf_config.model_type) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle", "deepseek_mtp"]) \
                    else {"return_hidden_states": True}

        is_encoder_decoder_model = self._is_encoder_decoder_model()
        ModelRunnerClass: Type[HPUModelRunnerBase] = HPUModelRunner
        is_causal = True
        if self.model_config.runner_type == "pooling":
            ModelRunnerClass = HPUPoolingModelRunner
        elif is_encoder_decoder_model:
            ModelRunnerClass = HPUEncoderDecoderModelRunner
        self.model_runner: HPUModelRunnerBase = ModelRunnerClass(
            vllm_config=vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
            is_causal=is_causal,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[HPUCacheEngine]
        # Initialize gpu_cache as pooling models don't initialize kv_caches
        self.hpu_cache: Optional[List[List[torch.Tensor]]] = None
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)

            if os.getenv('VLLM_PROFILER_ENABLED') == 'full':
                fn = self.model_runner.profiler.full_trace_handler
                with_stack = False
            else:
                fn = torch.profiler.tensorboard_trace_handler
                with_stack = True
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.HPU,
                ],
                with_stack=with_stack,
                on_trace_ready=fn(torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def _is_encoder_decoder_model(self):
        return self.model_config.is_encoder_decoder

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        high_level_profiler = self.model_runner.profiler
        with high_level_profiler.record_event('internal', 'start_profiler'):
            # Clean up the queue
            while True:
                try:
                    high_level_profiler.profiling_trace_events.get_nowait()
                except queue.Empty:
                    break
            self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def _set_env_vars(self):
        local_rank = self.local_rank
        if self.parallel_config.world_size == 1:
            local_rank = -1
        import os
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["ID"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(self.parallel_config.world_size)
        os.environ["RANK"] = str(self.rank)

    def init_device(self) -> None:
        if self.device_config.device.type == "hpu":
            self.device = torch.device("hpu")
            if self.vllm_config.parallel_config.pipeline_parallel_size > 1:
                # When using PCIe cards with pipeline parallelism enabled,
                # set the HPU device using local_rank to maintain NUMA affinity.
                torch.hpu.set_device(self.local_rank)
            else:
                torch.hpu.set_device(self.device)
        elif self.device_config.device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        if self.model_config.quantization == 'inc':
            self._set_env_vars()
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()
        if isinstance(self.model_runner, HPUPoolingModelRunner):
            # recipes we will use the extra memory for graphs/blocks
            free_hpu_memory = torch.hpu.mem_get_info()[0]
            hpu_memory_margin = free_hpu_memory * (
                1 - self.cache_config.gpu_memory_utilization)
            self.model_runner.mem_margin = hpu_memory_margin
            self._warm_up_model()

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        # VLLM_HPU_LOG_STEP_GRAPH_COMPILATION     - will log graph compilations per engine step, only when there was any - highly recommended to use alongside PT_HPU_METRICS_GC_DETAILS! # noqa:E501
        # VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL - will log graph compilations per engine step, always, even if there were none # noqa:E501
        # VLLM_HPU_LOG_STEP_CPU_FALLBACKS         - will log cpu fallbacks per engine step, only when there was any # noqa:E501
        # VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL     - will log cpu fallbacks per engine step, always, even if there were none # noqa:E501
        log_graph_compilation_all = os.environ.get(
            'VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL', '0') != '0'
        log_graph_compilation = os.environ.get(
            'VLLM_HPU_LOG_STEP_GRAPH_COMPILATION',
            '0') != '0' or log_graph_compilation_all
        log_cpu_fallbacks_all = os.environ.get(
            'VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL', '0') != '0'
        log_cpu_fallbacks = os.environ.get('VLLM_HPU_LOG_STEP_CPU_FALLBACKS',
                                           '0') != '0' or log_cpu_fallbacks_all
        if (log_graph_compilation or log_cpu_fallbacks) and \
            execute_model_req is not None:
            from habana_frameworks.torch.hpu.metrics import metric_localcontext
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list
            is_prompt = any([
                seq_group_metadata.is_prompt
                for seq_group_metadata in seq_group_metadata_list
            ])
            # for dummy run in DP, we don't have real seq,
            # so use a dummy context_len here
            if len(seq_group_metadata_list) == 0:
                max_context_len = 128
            else:
                max_context_len = max([
                    max([
                        len(v.prompt_token_ids) + len(v.output_token_ids)
                        for v in seq_group_metadata.seq_data.values()
                    ]) for seq_group_metadata in seq_group_metadata_list
                ])  # whoa, that's some spicy stuff right here
            max_num_blocks = (
                (max_context_len - 1) // self.cache_config.block_size) + 1
            input_stats = (f'is_prompt: {is_prompt}, '
                           f'num_seqs: {len(seq_group_metadata_list)}, '
                           f'max_context_len: {max_context_len}, '
                           f'max_num_blocks {max_num_blocks}')
            gc_ctx = metric_localcontext(
                "graph_compilation"
            ) if log_graph_compilation else contextlib.nullcontext()
            cpu_fallback_ctx = metric_localcontext(
                "cpu_fallback"
            ) if log_cpu_fallbacks else contextlib.nullcontext()
            with gc_ctx as gc_local_metric, \
                cpu_fallback_ctx as cpu_fallback_local_metric:
                output = LocalOrDistributedWorkerBase.execute_model(
                    self, execute_model_req)
            if (log_graph_compilation and gc_local_metric.stats()[0][1]
                    > 0) or log_graph_compilation_all:
                msg = ("VLLM_HPU_STEP_GRAPH_COMPILATION: "
                       f"{gc_local_metric.stats()}, {input_stats}")
                logger.warning(msg)
            if (log_cpu_fallbacks and cpu_fallback_local_metric.stats()[0][1]
                    > 0) or log_cpu_fallbacks_all:
                msg = ("VLLM_HPU_STEP_CPU_FALLBACK: "
                       f"{cpu_fallback_local_metric.stats()}, {input_stats}")
                logger.warning(msg)

            return output

        output = LocalOrDistributedWorkerBase.execute_model(
            self, execute_model_req)
        return output

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        if is_fake_hpu():
            cache_block_size = self.get_cache_block_size_bytes()
            fake_hpu_cache_alloc = 4 * 2**30  # take 4 GiB flat on fake hpu
            num_fake_hpu_blocks = fake_hpu_cache_alloc // cache_block_size
            self.model_runner.bucketing_manager.num_hpu_blocks = \
                    num_fake_hpu_blocks
            return num_fake_hpu_blocks, 0
        with HabanaMemoryProfiler() as m:
            self.model_runner.profile_run()
            torch.hpu.synchronize()
        msg = ("Model profiling run "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        # At this point we should've allocated the maximum workspace for all
        # recipes we will use the extra memory for graphs/blocks
        free_hpu_memory = torch.hpu.mem_get_info()[0]

        cache_block_size = self.get_cache_block_size_bytes()
        graph_reserved_mem = (float(
            os.environ.get('VLLM_GRAPH_RESERVED_MEM', '0.1'))
                              if not self.model_config.enforce_eager else 0)
        graph_headroom = 1 - graph_reserved_mem
        available_hpu_memory = free_hpu_memory * \
            self.cache_config.gpu_memory_utilization
        hpu_memory_margin = free_hpu_memory * (
            1 - self.cache_config.gpu_memory_utilization)
        self.model_runner.mem_margin = hpu_memory_margin
        cache_size_bytes = available_hpu_memory * graph_headroom
        graph_headroom_bytes = available_hpu_memory * (1 - graph_headroom)
        msg = (
            f"Free device memory: {format_bytes(free_hpu_memory)}, "
            f"{format_bytes(available_hpu_memory)} usable "
            f"(gpu_memory_utilization={self.cache_config.gpu_memory_utilization}),"
            f" {format_bytes(graph_headroom_bytes)} reserved for HPUGraphs "
            f"(VLLM_GRAPH_RESERVED_MEM={graph_reserved_mem}), "
            f"{format_bytes(cache_size_bytes)} reserved for KV cache")
        logger.info(msg)
        num_hpu_blocks = int(cache_size_bytes // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_hpu_blocks = max(num_hpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()

        gc.collect()
        return num_hpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(
            num_gpu_blocks, self.cache_config.block_size,
            self.model_config.max_model_len,
            self.parallel_config.pipeline_parallel_size)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.model_runner.bucketing_manager.num_hpu_blocks = (
            num_gpu_blocks // self.parallel_config.pipeline_parallel_size)
        self.model_runner.bucketing_manager.generate_prompt_buckets()
        if not self.model_runner.is_pooler:
            self.model_runner.bucketing_manager.generate_decode_buckets()

        with HabanaMemoryProfiler() as m:
            self._init_cache_engine()
            torch.hpu.synchronize()
        msg = ("Initializing cache engine "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            HPUCacheEngine(self.cache_config, self.model_config,
                           self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.hpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
        bind_kv_cache(self.compilation_config.static_forward_context,
                      self.hpu_cache)

    def _warm_up_model(self) -> None:
        # NOTE(kzawora): We should use virtual engine index here
        # for pipeline parallelism. Using 0 for now.
        if not isinstance(self.model_runner, HPUPoolingModelRunner):
            assert self.hpu_cache is not None
            self.model_runner.warmup_model(self.hpu_cache[0])
        else:
            self.model_runner.warmup_model(None)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.hpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def shutdown(self):
        getattr(self.model_runner, 'shutdown_inc', lambda: None)()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return HPUCacheEngine.get_cache_block_size(self.cache_config,
                                                   self.model_config,
                                                   self.parallel_config)


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    backend = hpu_backend_string()
    init_distributed_environment(parallel_config.world_size,
                                 rank,
                                 distributed_init_method,
                                 local_rank,
                                 backend=backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    if parallel_config.pipeline_parallel_size > 1:
        # torch-ccl xpu need a collective API warm up
        # before calling send/recv API
        get_pp_group().all_reduce(torch.zeros(1).to('hpu'))
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        expected_size = parallel_config.world_size *\
            parallel_config.data_parallel_size
        if torch_world_size != expected_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size * "
                "parallel_config.data_parallel_size "
                f"({torch_world_size} vs. {expected_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        backend = hpu_backend_string()
        torch.distributed.init_process_group(
            backend=backend,
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup & checking conformance.
    device = hpu_device_string()
    dummy_tensor_hpu = torch.ones(1).to(device)
    torch.distributed.all_reduce(dummy_tensor_hpu)
    assert dummy_tensor_hpu.item(
    ) == parallel_config.world_size * parallel_config.data_parallel_size
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
    ensure_kv_transfer_initialized(vllm_config)


def raise_if_cache_size_invalid(num_gpu_blocks, block_size, max_model_len,
                                pipeline_parallel_size) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * (num_gpu_blocks // pipeline_parallel_size)
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")


class HPUCacheEngine(CacheEngine):

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        k_cache_shape = kv_cache_shape
        v_cache_shape = None if self.model_config.use_mla else kv_cache_shape
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        dtype = self.dtype
        if device != 'hpu' and not is_fake_hpu() \
          and self.dtype == torch.float8_e4m3fn:
            dtype = torch.uint8
        for _ in range(self.num_attention_layers):
            key_cache = torch.zeros(k_cache_shape, dtype=dtype, device=device)
            if v_cache_shape is not None:
                value_cache = torch.zeros(v_cache_shape,
                                          dtype=dtype,
                                          device=device)
            else:
                value_cache = None
            kv_layer = (key_cache, value_cache)
            kv_cache.append(kv_layer)
        return kv_cache
