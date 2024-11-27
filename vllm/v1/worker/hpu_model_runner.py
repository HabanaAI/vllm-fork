import collections
from enum import Enum
import functools
import itertools
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, cdiv, is_fake_hpu,
                        is_pin_memory_available)
from vllm.v1.attention.backends.hpu_attn import HPUAttentionBackendV1, HPUAttentionMetadata
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm_hpu_extension.profiler import (HabanaHighLevelProfiler,
                                         HabanaMemoryProfiler, format_bytes)
from vllm_hpu_extension.bucketing import HPUBucketingContext
from vllm_hpu_extension.ops import batch2block, block2batch
import habana_frameworks.torch as htorch
import habana_frameworks.torch.internal.bridge_config as bc
from multiprocessing.pool import ThreadPool

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.engine.detokenizer import Detokenizer

logger = init_logger(__name__)

_TYPE_CACHE = {}
# These values are assumed to be zero in several places.
# Use caution when updating them!
_PAD_SLOT_ID = 0
_PAD_BLOCK_ID = 0

class PhaseType(Enum):
    UNCACHED_PREFILL = 'uncached_prefill'
    CACHED_PREFILL = 'cached_prefill'
    DECODE = 'decode'
    

@dataclass
class PrefillInputData:

    request_ids: List
    prompt_lens: List
    token_ids: List
    position_ids: List
    attn_metadata: List
    logits_indices: List

    def zipped(self):
        return zip(self.request_ids, self.prompt_lens, self.token_ids,
                   self.position_ids, self.attn_metadata, self.logits_indices)


@dataclass
class DecodeInputData:

    num_decodes: int
    token_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    attn_metadata: HPUAttentionMetadata = None
    logits_indices: Optional[torch.Tensor] = None



def flatten(in_list):
    return list(itertools.chain(*in_list))


def gather_list(input, indices, v):
    return [input[i] if i is not None else v for i in indices]


def _async_h2d_tensor_copy(source, device='hpu'):
    assert source.device.type == 'cpu', "Source tensor is not present in host memory!"
    target = torch.empty(source.shape, dtype=source.dtype, device=device)
    target.copy_(source, non_blocking=True)
    return target


class HpuModelAdapter:

    def __init__(self, model, block_size, dtype, enforce_eager):
        self.model = model
        self.prefill_use_fusedsdpa = os.getenv('VLLM_PROMPT_USE_FUSEDSDPA',
                                               '1').lower() in ['1', 'true'] \
                                                and not is_fake_hpu()
        self.block_size = block_size
        self.dtype = dtype
        if not is_fake_hpu() and not htorch.utils.internal.is_lazy(
        ) and not enforce_eager:
            self.model = torch.compile(self.model,
                                       backend='hpu_backend',
                                       dynamic=False)

    def _set_attn_bias(self, attn_metadata, batch_size, seq_len, device,
                       dtype):
        prefix_caching_enabled = attn_metadata.is_prompt and attn_metadata.block_list is not None
        if (attn_metadata is None or (self.prefill_use_fusedsdpa and not prefix_caching_enabled)
                or not attn_metadata.is_prompt):
            return attn_metadata

        prefill_metadata = attn_metadata

        context_lens_t = prefill_metadata.context_lens_tensor
        query_lens_t = prefill_metadata.seq_lens_tensor

        block_list = attn_metadata.block_list
        max_context_len = (block_list.size(-1) //
                           batch_size if block_list is not None else 0)
        max_context_len = max_context_len * self.block_size
        past_mask = torch.arange(0,
                                 max_context_len,
                                 dtype=torch.int32,
                                 device=device)
        past_mask = (past_mask.view(1, -1).expand(batch_size, -1).ge(
            context_lens_t.view(-1, 1)).view(batch_size, 1, -1).expand(
                batch_size, seq_len, -1).view(batch_size, 1, seq_len, -1))

        len_mask = (torch.arange(0, seq_len, device=device,
                                 dtype=torch.int32).view(1, seq_len).ge(
                                     query_lens_t.unsqueeze(-1)).view(
                                         batch_size, 1, 1, seq_len))
        causal_mask = torch.triu(torch.ones((batch_size, 1, seq_len, seq_len),
                                            device=device,
                                            dtype=torch.bool),
                                 diagonal=1)
        mask = causal_mask.logical_or(len_mask)
        mask = torch.concat((past_mask, mask), dim=-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))
        attn_metadata = prefill_metadata._replace(attn_bias=attn_bias)
        return attn_metadata

    def _set_block_mapping(self, metadata, batch_size, device, dtype):
        mask = torch.arange(0,
                            self.block_size,
                            device=device,
                            dtype=torch.int32).unsqueeze(0)
        mask = mask >= metadata.block_usage.unsqueeze(-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))

        if not is_fake_hpu() and htorch.utils.internal.is_lazy():
            block_mapping = torch.nn.functional.one_hot(metadata.block_groups,
                                                        num_classes=batch_size)
        else:
            # Unfortunately one_hot on CPU/torch.compile mode/eager mode
            # doesn't handle out of bounds classes so we need to convert
            # all negative values to 0 (block_mapping) or bs (block_groups)
            block_groups = metadata.block_groups.to(torch.long)
            block_mapping = torch.nn.functional.relu(block_groups)
            block_mapping = torch.nn.functional.one_hot(block_mapping,
                                                        num_classes=batch_size)
            oob_values = block_groups.lt(0)
            block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
            block_groups.masked_fill_(oob_values, batch_size)
            metadata = metadata._replace(block_groups=block_groups)
        block_mapping = block_mapping.to(dtype)
        metadata = metadata._replace(block_mapping=block_mapping,
                                     attn_bias=attn_bias)
        return metadata

    def _set_block_scales(self, metadata, device):
        block_mapping = metadata.block_mapping
        ones = torch.ones((block_mapping.size(0), ),
                          device=device,
                          dtype=block_mapping.dtype)
        sums = batch2block(block2batch(ones, block_mapping), block_mapping)
        block_scales = torch.reciprocal(torch.maximum(ones, sums))
        metadata = metadata._replace(block_scales=block_scales)
        return metadata

    def _set_indices_and_offsets(self, metadata, block_size, is_prompt):
        slot_mapping = metadata.slot_mapping.flatten()
        indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        if is_prompt:
            indices = indices.unflatten(0, (-1, block_size))[:, 0]
            offsets = None
        else:
            offsets = torch.fmod(slot_mapping, block_size)
        metadata = metadata._replace(block_offsets=offsets,
                                     block_indices=indices)
        return metadata

    def _update_metadata(self, attn_metadata, batch_size, seq_len, device,
                         dtype):
        if attn_metadata.is_prompt:
            attn_metadata = self._set_attn_bias(attn_metadata, batch_size,
                                                seq_len, device, dtype)
        else:
            attn_metadata = self._set_block_mapping(attn_metadata, batch_size,
                                                    device, dtype)
            attn_metadata = self._set_block_scales(attn_metadata, device)
        attn_metadata = self._set_indices_and_offsets(attn_metadata,
                                                      self.block_size,
                                                      attn_metadata.is_prompt)
        return attn_metadata

    def forward(self, *args, **kwargs):
        kwargs = kwargs.copy()
        input_ids = kwargs['input_ids']
        kwargs['attn_metadata'] = self._update_metadata(
            kwargs['attn_metadata'], input_ids.size(0), input_ids.size(1),
            input_ids.device, self.dtype)
        hidden_states = self.model(*args, **kwargs)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        return hidden_states

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def generate_proposals(self, *args, **kwargs):
        return self.model.generate_proposals(*args, **kwargs)

    # sampler property will be used by spec_decode_worker
    # don't rename
    @property
    def sampler(self):
        return self.model.sampler


def _maybe_wrap_in_hpu_graph(*args, **kwargs):
    return htorch.hpu.wrap_in_hpu_graph(
        HpuModelAdapter(*args, **kwargs), disable_tensor_cache=True
    ) if False and htorch.utils.internal.is_lazy() else HpuModelAdapter(
        *args, **kwargs)


def subtuple(obj: object,
             typename: str,
             to_copy: List[str],
             to_override: Optional[Dict[str, object]] = None):
    if obj is None:
        return None
    if to_override is None:
        to_override = {}
    fields = set(to_copy) | set(to_override.keys())
    if type(obj) is dict:
        values = {key: obj[key] for key in fields if key in obj}
    else:
        values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = collections.namedtuple(typename,
                                                       ' '.join(fields))
    return _TYPE_CACHE[typename](**values)


def trim_attn_metadata(metadata: HPUAttentionMetadata) -> object:
    # NOTE(kzawora): To anyone working on this in the future:
    # Trimming metadata is required when using HPUGraphs.
    # Attention metadata is going to be hashed by PT bridge, and
    # appropriate HPUGraphs will be matched based on all inputs' hash.

    # Before you put more keys in here, make sure you know their
    # value type and make sure you know how it's going to be hashed.
    # You can find that information in input_hash function
    # in habana_frameworks/torch/hpu/graphs.py. You can also hash
    # it manually with torch.hpu.graphs.input_hash(attention_metadata)

    # If you use primitive types here - they will get hashed based
    # on their value. You *will* get lots of excessive graph captures
    # (and an OOM eventually) if you decide to put something like
    # seq_len int here.
    # If you absolutely need a scalar, put it in a tensor. Tensors
    # get hashed using their metadata, not their values:
    # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
    # input_hash(123) != input_hash(321)
    # input_hash("abc") != input_hash("cba")
    attention_metadata = subtuple(metadata, 'TrimmedAttentionMetadata', [
        'attn_bias', 'seq_lens_tensor', 'context_lens_tensor', 'block_list',
        'block_mapping', 'block_usage', 'slot_mapping', 'is_prompt',
        'block_indices', 'block_offsets', 'block_scales', 'block_groups'
    ])
    return attention_metadata


def next_pow2(value: int, base: int):
    res = base
    while value > 1:
        value = (value + 1) // 2
        res *= 2
    return res


def round_up(value: int, k: int):
    return (value + k - 1) // k * k


def pad_list(list, k, v):
    target_len = round_up(len(list), k)
    padding = target_len - len(list)
    return list + [v] * padding


def precompute_indices_and_offsets(block_size, slot_mapping, is_prompt):
    slot_mapping = slot_mapping.flatten()
    indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    if is_prompt:
        indices = indices.unflatten(0, (-1, block_size))[:, 0]
        offsets = None
    else:
        offsets = torch.fmod(slot_mapping, block_size)
    return indices, offsets


class HPUModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        #TODO(kzawora): remove this, this is ugly and only used for diagnostics
        self._ENGINE_ITER = 0
        # TODO: use ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        #TODO(kzawora): remove this, this is for debug purposes only
        self._tokenizer = Detokenizer(
            vllm_config.model_config.tokenizer).tokenizer
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Model-related.
        self.num_attn_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        self.use_hpu_graph = not self.model_config.enforce_eager
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        self.cudagraph_batch_sizes = [1, 2, 4] + [i for i in range(8, 513, 8)]
        self.max_batch_size = 256  # TODO(kzawora): fix this garbage
        self.input_ids = torch.zeros(
            (self.max_batch_size, self.max_num_tokens),
            dtype=torch.int32,
            device=self.device)
        self.positions = torch.zeros(
            (self.max_batch_size, self.max_num_tokens),
            dtype=torch.int64,
            device=self.device)
        self.prefill_positions = torch.tensor(
            range(self.max_model_len),
            device="cpu",
        ).to(torch.int32).reshape(1, -1)
        
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_prefill_batch_size = 16 # TODO(kzawora): add some knob for that
        self.padding_aware_scheduling = True # TODO(kzawora): add some knob for that
        self.padding_ratio_threshold = 0.9 # TODO(kzawora): add some knob for that
        self.use_contiguous_pa = os.environ.get('VLLM_CONTIGUOUS_PA',
                                                'true').lower() == 'true'
        self.seen_configs: set = set()
        self.enable_bucketing = os.environ.get(
            'VLLM_DISABLE_BUCKETING', 'false').lower() not in ['true', '1']
        if self.enable_bucketing:
            logger.info("Bucketing is ON.")
            self.bucketing_ctx = HPUBucketingContext(self.max_num_seqs,
                                                        self.max_prefill_batch_size,
                                                        self.block_size,
                                                        self.scheduler_config.max_num_batched_tokens)
            self.graphed_buckets: Set[Any] = set()
        else:
            logger.info("Bucketing is OFF.")
        self.skip_warmup = os.environ.get('VLLM_SKIP_WARMUP',
                                          'false').lower() == 'true'


    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the requests from the persistent batch.
        stopped_req_ids = set().union(
            scheduler_output.preempted_req_ids,
            scheduler_output.finished_req_ids,
        )
        removed_req_indices: List[int] = []
        for req_id in stopped_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Update the states of the running requests.
        for req_data in scheduler_output.scheduled_running_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]
            req_index = self.input_batch.req_id_to_index[req_id]

            # Update the num_computed_tokens.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)

            # Update the block table.
            num_new_blocks = len(req_data.new_block_ids)
            if num_new_blocks == 0:
                continue
            start_index = len(req_state.block_ids)
            end_index = start_index + num_new_blocks
            req_state.block_ids.extend(req_data.new_block_ids)
            self.input_batch.block_table_cpu[
                req_index, start_index:end_index] = req_data.new_block_ids

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            sampling_params = req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=req_data.prompt_token_ids,
                prompt=req_data.prompt,
                mm_inputs=req_data.mm_inputs,
                mm_positions=req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=req_data.block_ids,
                num_computed_tokens=req_data.num_computed_tokens,
                output_token_ids=[],
            )
            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = req_data.block_ids
            req_state.num_computed_tokens = req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

        # THIS MOVES ALL THE DECODES TO THE FIRST N IN BATCH.
        # Condense the batched states if there are empty indices.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        # ALL THE PREFILLS ARE THE LAST M IN THE BATCH.
        # These are added at the end after the bacth is condensed.
        self.input_batch.num_prefills = len(req_ids_to_add)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state, None)

    def _prepare_sampling(self,
                          scheduler_output: "SchedulerOutput",
                          start_idx: Optional[int] = None,
                          end_idx: Optional[int] = None,
                          pad_to: Optional[int] = None) -> SamplingMetadata:
        skip_copy = True
        if start_idx is None and end_idx is None:
            if (scheduler_output.finished_req_ids
                    or scheduler_output.preempted_req_ids):
                skip_copy = False
            if (scheduler_output.scheduled_new_reqs
                    or scheduler_output.scheduled_resumed_reqs):
                skip_copy = False
        else:
            #TODO(kzawora): something smells... kinda fishy in here
            req_ids = self.input_batch.req_ids[start_idx:end_idx]
            finished_req_ids = any([
                req_id in scheduler_output.finished_req_ids
                for req_id in req_ids
            ])
            preempted_req_ids = any([
                req_id in scheduler_output.preempted_req_ids
                for req_id in req_ids
            ])
            scheduled_new_reqs = any([
                req_id in scheduler_output.scheduled_new_reqs
                for req_id in req_ids
            ])
            scheduled_resumed_reqs = any([
                req_id in scheduler_output.scheduled_resumed_reqs
                for req_id in req_ids
            ])

            if (finished_req_ids or preempted_req_ids):
                skip_copy = False
            if (scheduled_new_reqs or scheduled_resumed_reqs):
                skip_copy = False

        # Create the sampling metadata.
        sampling_metadata = self.input_batch.make_sampling_metadata(
            skip_copy=skip_copy,
            start_idx=start_idx,
            end_idx=end_idx,
            pad_to=pad_to)
        return sampling_metadata

    def get_habana_paged_attn_buffers(self,
                                      block_tables,
                                      slot_mapping,
                                      bucketing=True):

        last_block_usage = [
            slot[0] % self.block_size + 1 for slot in slot_mapping
        ]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
        block_usage = [[self.block_size] * (len(bt) - 1) + [lbu]
                       for bt, lbu in zip(block_tables, last_block_usage)
                       if bt]

        block_list = flatten(block_tables)
        block_groups = flatten(block_groups)
        block_usage = flatten(block_usage)
        assert len(block_list) == len(block_groups)
        assert len(block_list) == len(block_usage)

        padding_fn = None
        if self.use_contiguous_pa:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            if bucketing:
                block_bucket_size = self.bucketing_ctx.get_padded_decode_num_blocks(
                    block_bucket_size)
            indices: List[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value)
        else:
            block_bucket_size: int
            if bucketing:
                block_bucket_size = self.bucketing_ctx.get_padded_decode_num_blocks(
                    len(block_list))
            else:
                block_bucket_size = len(block_list)
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, pad_value)

        block_list = padding_fn(block_list, _PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        block_list = torch.tensor(block_list, dtype=torch.long, device='cpu')
        block_groups = torch.tensor(block_groups,
                                    dtype=torch.long,
                                    device='cpu')
        block_usage = torch.tensor(block_usage,
                                   dtype=self.model_config.dtype,
                                   device='cpu')

        return block_list, block_groups, block_usage

    def _get_padded_prefill_dims(self, num_prefills, max_prompt_len, bucketing):
        if bucketing:
            padded_batch_size = self.bucketing_ctx.get_padded_batch_size(
                num_prefills, True)
            padded_prompt_len = self.bucketing_ctx.get_padded_prompt_seq_len(
                max_prompt_len)
        else:
            #NOTE(kzawora): On HPU prompt length needs to be block_size
            # aligned, so we're padding to that, even if bucketing
            # is disabled.
            padded_batch_size = num_prefills
            padded_prompt_len = math.ceil(
                max_prompt_len / self.block_size) * self.block_size
        assert padded_prompt_len <= self.max_model_len           
        return padded_batch_size, padded_prompt_len

        
    
    def _prepare_prefill_inputs(self,
                                num_scheduled_tokens: List[int],
                                bucketing=True) -> PrefillInputData:
        # Each prefill run separately with shape [1, padded_prompt_len].
        # So we create lists that will be used in execute_model().

        prefill_request_ids = []
        prefill_prompt_lens = []
        prefill_token_ids = []
        prefill_position_ids = []
        prefill_attn_metadata = []
        prefill_logits_indices = []

        # DECODES are the first num_decodes REQUESTS.
        # PREFILLS are the next num_reqs - num_decodes REQUESTS.
        num_reqs = self.input_batch.num_reqs
        num_decodes = self.input_batch.num_decodes
        # NOTE(kzawora): This loop was initially implemented as 
        # for batch_idx in range(num_decodes, num_reqs, max_prefill_batch_size)
        # but was changed to accomodate variable loop step size for 
        # padding-aware scheduling  
        batch_idx = num_decodes
        while batch_idx < num_reqs:
            # Find the largest batch size in range [1, max_prefill_batch_size] 
            # that can fit within specified token budget
            num_prefills: int
            padded_batch_size: int
            padded_prompt_len: int
            padded_num_tokens: int
            padding_ratio: float
            batch_req_ids: List[int]
            prompt_lens: List[int]
            for possible_batch_size in reversed(range(1, self.max_prefill_batch_size + 1)):
                if batch_idx + possible_batch_size > num_reqs:
                    continue
                num_prefills = possible_batch_size
                batch_req_ids = self.input_batch.req_ids[batch_idx:batch_idx + num_prefills]
                prompt_lens = num_scheduled_tokens[batch_idx:batch_idx + num_prefills]
                max_prompt_len = max(prompt_lens)
                num_tokens = sum(prompt_lens)
                padded_batch_size, padded_prompt_len = self._get_padded_prefill_dims(num_prefills, max_prompt_len, bucketing)
                padded_num_tokens = padded_batch_size * padded_prompt_len
                padding_ratio = 1 - (num_tokens/padded_num_tokens)
                is_within_token_budget = padded_batch_size * padded_prompt_len < self.scheduler_config.max_num_batched_tokens
                is_within_padding_ratio_threshold = padding_ratio < self.padding_ratio_threshold 
                can_schedule = is_within_token_budget and is_within_padding_ratio_threshold
                # If padding aware scheduling is off, we'll break on the first 
                # loop iteration (==max_prefill_batch_size). 
                # Else, we'll break on first batch size that fits token budget.
                if not self.padding_aware_scheduling or can_schedule:
                    break
            context_lens = self.input_batch.num_computed_tokens_cpu[batch_idx:batch_idx + num_prefills]
            use_prefix_caching = any(context_lens)
            # TODO(kzawora): this is an ugly hack for prefix caching, remove that once batch padding works properly (idk why it doesn't)
            if use_prefix_caching:
                padded_batch_size = num_prefills

            padded_prompt_lens = [
                padded_prompt_len for _ in range(padded_batch_size)
            ]
            #logger.info(f"Using {'cached' if use_prefix_caching else 'uncached'} prefill batch size {num_prefills} padded to [{padded_batch_size}, {padded_prompt_len}] with context {max(context_lens)} (token budget: {self.scheduler_config.max_num_batched_tokens}, num_tokens: {padded_num_tokens}, padding ratio: {padding_ratio:.2f})")

            # TOKEN_IDS.
            token_ids = torch.zeros((padded_batch_size, padded_prompt_len),
                                    dtype=torch.int32,
                                    device='cpu')
            # POSITIONS.
            positions = torch.zeros((padded_batch_size, padded_prompt_len),
                                    dtype=torch.int32,
                                    device='cpu')
            # SLOT_MAPPING.
            # The "slot" is the "physical index" of a token in the KV cache.
            # Look up the block_idx in the block table (logical<>physical map)
            # to compute this.
            slot_mapping = torch.zeros((padded_batch_size, padded_prompt_len),
                                       dtype=torch.int32,
                                       device='cpu')
            slot_mapping.fill_(_PAD_SLOT_ID)
                
            for i, (prompt_len, context_len) in enumerate(zip(prompt_lens, context_lens)):
                # Prepare and sanitize token ids (cpu)
                batch_offset = batch_idx + i
                token_ids[i, :prompt_len] = torch.from_numpy(self.input_batch.token_ids_cpu[batch_offset, context_len:context_len+prompt_len])
                #token_ids[i, prompt_len:] = 0 # no need to sanitize - buffer is pre-filled with 0s
                
                # Prepare and sanitize positions ids (cpu)
                positions[i, :prompt_len] = self.prefill_positions[:, context_len:context_len+prompt_len]
                #positions[i, prompt_len:] = 0 # no need to sanitize - buffer is pre-filled with 0s

                # Prepare and sanitize slot_mapping (cpu)
                flat_prefill_positions = positions[i, :prompt_len].flatten()
                block_numbers = self.input_batch.block_table_cpu_tensor[batch_offset,
                    flat_prefill_positions // self.block_size]
                block_offsets = flat_prefill_positions % self.block_size
                slot_mapping[i, :prompt_len] = block_numbers * self.block_size + block_offsets
                #slot_mapping[i, prompt_len:] = _PAD_SLOT_ID # no need to sanitize - buffer is pre-filled with _PAD_SLOT_IDs
            slot_mapping = slot_mapping.long()

            logits_indices = torch.zeros(padded_batch_size,
                                         dtype=torch.int32,
                                         device='cpu')
            query_start_loc = torch.empty((num_prefills + 1, ),
                                          dtype=torch.int32,
                                          device="cpu")
            query_start_loc_np = query_start_loc.numpy()
            query_start_loc_np[0] = 0

            # logits indices in prefill must account for padding: last token logits
            # will be emitted at index (idx - 1) * padded_seq_len + seq_len[idx] - 1
            np.cumsum(padded_prompt_lens[:num_prefills],
                      out=query_start_loc_np[1:])
            query_start_loc_np[:num_prefills] += num_scheduled_tokens[
                batch_idx:batch_idx + num_prefills]
            logits_indices[:num_prefills] = query_start_loc[:num_prefills] - 1

            # HPU should *not* sync here with CPU
            seq_lens_tensor = torch.zeros((padded_batch_size),
                                          dtype=torch.int32,
                                          device='cpu')
            seq_lens_tensor[:num_prefills] = torch.tensor(prompt_lens,
                                                          device='cpu')
            token_ids_device = _async_h2d_tensor_copy(token_ids, self.device)
            positions_device = _async_h2d_tensor_copy(positions, self.device)
            seq_lens_tensor_device = _async_h2d_tensor_copy(
                seq_lens_tensor, self.device)
            slot_mapping_device = _async_h2d_tensor_copy(
                slot_mapping, self.device)
            logits_indices_device = _async_h2d_tensor_copy(
                logits_indices, self.device)

            prefill_request_ids.append(batch_req_ids)
            prefill_prompt_lens.append(prompt_lens)
            prefill_token_ids.append(token_ids_device)
            prefill_position_ids.append(positions_device)
            prefill_logits_indices.append(logits_indices_device)
            attn_metadata = None
            if use_prefix_caching:
                # Prefix caching
                num_blocks = np.ceil(context_lens / self.block_size).astype(np.int32).tolist()
                max_num_blocks = max(num_blocks)
                #if bucketing:
                #    max_num_blocks = self.bucketing_ctx.get_padded_decode_num_blocks(max_num_blocks)
                prefix_block_tables = torch.zeros((padded_batch_size, max_num_blocks), dtype=torch.int32,device='cpu')
                for i, n in enumerate(num_blocks):
                    prefix_block_tables[i, :n] = self.input_batch.block_table_cpu_tensor[i, :n]
                context_lens_tensor = torch.zeros((padded_batch_size), dtype=torch.int32, device='cpu')
                context_lens_tensor[:num_prefills] = torch.tensor(context_lens, device='cpu')
                
                block_list_device = _async_h2d_tensor_copy(prefix_block_tables.flatten(), self.device)
                context_lens_tensor_device = _async_h2d_tensor_copy(context_lens_tensor, self.device)
                attn_metadata =  HPUAttentionMetadata.make_cached_prefill_metadata(
                        seq_lens_tensor=seq_lens_tensor_device,
                        context_lens_tensor=context_lens_tensor_device,
                        num_prefills=num_prefills,
                        num_prefill_tokens=sum(prompt_lens),
                        slot_mapping=slot_mapping_device,
                        block_list=block_list_device
                    )
            else:
                attn_metadata =  HPUAttentionMetadata.make_prefill_metadata(
                        seq_lens_tensor=seq_lens_tensor_device,
                        num_prefills=num_prefills,
                        num_prefill_tokens=sum(prompt_lens),
                        slot_mapping=slot_mapping_device,
                    )
            #import pdb; pdb.set_trace()
            # ATTN_METADATA.
            prefill_attn_metadata.append(attn_metadata)
            batch_idx += num_prefills

        return PrefillInputData(request_ids=prefill_request_ids,
                                prompt_lens=prefill_prompt_lens,
                                token_ids=prefill_token_ids,
                                position_ids=prefill_position_ids,
                                attn_metadata=prefill_attn_metadata,
                                logits_indices=prefill_logits_indices)

    def _prepare_decode_inputs(self,
                               num_scheduled_tokens,
                               bucketing=True) -> DecodeInputData:
        # Decodes run as one single padded batch with shape [batch, 1]
        #
        # We need to set _PAD_SLOT_ID for the padding tokens in the
        # slot_mapping, such that the attention KV cache insertion
        # logic knows to ignore those indicies. Otherwise, the
        # padding data can be dummy since we have a causal mask.

        num_reqs = self.input_batch.num_reqs
        num_decodes = self.input_batch.num_decodes
        if num_decodes == 0:
            return DecodeInputData(num_decodes=0)

        # PAD FOR STATIC SHAPES.
        padded_batch_size: int
        if bucketing:
            padded_batch_size = self.bucketing_ctx.get_padded_batch_size(
                num_decodes, False)
        else:
            padded_batch_size = num_decodes

        # POSITIONS. [batch, 1]
        # We slice at the end, since we use the positions for gathering.
        positions = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu.reshape(-1, 1))
        index = positions.to(torch.int64)
        positions = positions[:padded_batch_size]

        # TOKEN_IDS. [batch, 1]
        token_ids = torch.zeros((padded_batch_size, 1), dtype=torch.int32)
        token_ids[:num_decodes] = torch.gather(
            input=torch.from_numpy(self.input_batch.token_ids_cpu),
            dim=1,
            index=index,
        )[:num_decodes]

        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_number = torch.gather(
            input=self.input_batch.block_table_cpu_tensor,
            dim=1,
            index=(index // self.block_size))
        # NOTE(kzawora): the "-1" is what causes this entire thing to work
        # properly and have good accuracy - why? beats me...
        block_offsets = (index - 1) % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # Set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        slot_mapping[num_decodes:] = _PAD_SLOT_ID
        slot_mapping = slot_mapping[:padded_batch_size]

        # BLOCK_TABLE [batch, max_num_blocks_per_req]
        context_lens = self.input_batch.num_computed_tokens_cpu[:num_decodes]
        num_blocks = np.ceil(context_lens / self.block_size).astype(
            np.int32).tolist()
        block_tables_list = []
        for i, n in enumerate(num_blocks):
            block_tables_list.append(
                self.input_batch.block_table_cpu_tensor[i, :n].tolist())

        # CONTEXT_LENS [batch_size]
        #context_lens = (positions.reshape(-1) + 1)

        block_list, block_groups, block_usage = self.get_habana_paged_attn_buffers(
            block_tables_list, slot_mapping.tolist(), bucketing)

        logits_indices = torch.zeros(padded_batch_size,
                                     dtype=torch.int32,
                                     device='cpu')
        query_start_loc = torch.empty((num_decodes + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens[:num_decodes],
                  out=query_start_loc_np[1:])
        logits_indices[:num_decodes] = query_start_loc[1:] - 1
        num_decode_tokens = torch.tensor(np.sum(context_lens), device='cpu')

        # CPU<>HPU sync *should not* happen here.
        token_ids_device = _async_h2d_tensor_copy(token_ids, self.device)
        positions_device = _async_h2d_tensor_copy(positions, self.device)
        logits_indices_device = _async_h2d_tensor_copy(logits_indices,
                                                       self.device)
        block_list_device = _async_h2d_tensor_copy(block_list, self.device)
        block_usage_device = _async_h2d_tensor_copy(block_usage, self.device)
        block_groups_device = _async_h2d_tensor_copy(block_groups, self.device)
        num_decode_tokens_device = _async_h2d_tensor_copy(
            num_decode_tokens, self.device)
        slot_mapping_device = _async_h2d_tensor_copy(slot_mapping, self.device)

        return DecodeInputData(
            num_decodes=num_decodes,
            token_ids=token_ids_device,
            position_ids=positions_device,
            logits_indices=logits_indices_device,
            attn_metadata=HPUAttentionMetadata.make_decode_metadata(
                block_list=block_list_device,
                block_usage=block_usage_device,
                block_groups=block_groups_device,
                num_decode_tokens=num_decode_tokens_device,
                slot_mapping=slot_mapping_device,
            ))

    def _prepare_inputs(
            self,
            scheduler_output: "SchedulerOutput",
            bucketing=True
    ) -> Tuple[PrefillInputData, Optional[DecodeInputData]]:

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0

        num_reqs = self.input_batch.num_reqs
        num_decodes = self.input_batch.num_decodes

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)

            # NOTE: assert that all the decodes are "decodes".
            if idx < num_decodes:
                assert num_tokens == 1

        return (
            self._prepare_prefill_inputs(num_scheduled_tokens, bucketing),
            self._prepare_decode_inputs(num_scheduled_tokens, bucketing),
        )

    def _seq_len(self, attn_metadata):
        return attn_metadata.slot_mapping.size(1)

    def _num_blocks(self, attn_metadata):
        if attn_metadata.block_list is None:
            return 0
        return attn_metadata.block_list.numel()
    
    def _phase(self, attn_metadata):
        phase_type : PhaseType
        is_prompt = attn_metadata.is_prompt
        is_prefix_cached = is_prompt and attn_metadata.block_list is not None
        if is_prompt and is_prefix_cached:
            phase_type = PhaseType.CACHED_PREFILL
        elif is_prompt and not is_prefix_cached:
            phase_type = PhaseType.UNCACHED_PREFILL
        elif not is_prompt:
            phase_type = PhaseType.DECODE
        else:
            raise ValueError("Unrecognized pass type, likely due to malformed attention metadata")
        return phase_type
    
    def _check_config(self, batch_size, seq_len, num_blocks, attn_metadata, warmup_mode):
        phase = self._phase(attn_metadata)
        cfg = (batch_size, seq_len, num_blocks, phase)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen and not warmup_mode:
            phase = phase.value
            logger.warning("Configuration: (%s, %s, %s, %s) was not warmed-up!",
                           phase, batch_size, seq_len, num_blocks)

    def _execute_model_generic(self, token_ids, position_ids, attn_metadata,
                               logits_indices, kv_caches, warmup_mode=False):
        # FORWARD.
        batch_size = token_ids.size(0)
        seq_len = self._seq_len(attn_metadata)
        num_blocks = self._num_blocks(attn_metadata)
        
        is_prompt = attn_metadata.is_prompt
        self._check_config(batch_size, seq_len, num_blocks, attn_metadata, warmup_mode)
        additional_kwargs = {}
        if htorch.utils.internal.is_lazy() and not self.model_config.enforce_eager:
            use_graphs = self._use_graphs(batch_size, seq_len, num_blocks, is_prompt)
            additional_kwargs.update(
                {"bypass_hpu_graphs": not use_graphs})
        trimmed_attn_metadata = trim_attn_metadata(attn_metadata)
        hidden_states = self.model.forward(input_ids=token_ids,
                                           positions=position_ids,
                                           attn_metadata=trimmed_attn_metadata,
                                           kv_caches=kv_caches)
        #hidden_states = hidden_states[:num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states, None)
        return logits

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        # NOTE(kzawora): Since scheduler doesn't differentiate between prefills
        # and decodes, we must handle mixed batches. In _update_states we make
        # sure that first self.input_batch.num_decodes requests are decodes,
        # and remaining ones until the end are prefills. _update_states also
        # handles changes in request cache based on scheduler outputs and
        # previous iterations (e.g. keeping block tables and context lengths up
        # to date, creating, pruning and updating request caches, and some more stuff)

        # If num_decodes == self.input_batch.num_reqs, then batch is all decode, and only a single decode forward pass will be executed in this method.
        # If num_decodes == 0, then batch is all prefill, and only prefill forward passes will be executed  in this method.
        # If neither apply, then batch is mixed, and both prefill and decode forward passes will be executed in this method.

        # First, we will execute all decodes (if any) in a single batch,
        # then we'll execute prefills in batches of up to max_prefill_batch_size elements.
        # All shapes used in forward passes are bucketed appropriately to mitigate risk of graph recompilations.

        # We can do sampling directly after executing each forward pass (split_sampler=True),
        # or execute all forward passes, join the results and execute it once (split_sampler=False).
        # Everything is done asynchronously - the only sync point is the place
        # where we copy the generated tokens back to the host.

        # Example: If a batch has 6 requests, 3 prefills and 3 decodes, the unprocessed sequences in batch will be laid as follows:
        # [D0, D1, D2, P0, P1, P2]
        # If we assume max_prefill_batch_size=2, and split_sampler=True the flow of this method will look as follows:
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding)
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1], [P2] (BS=2 + BS=1 bucket, no seqs padding)
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS2 [P0, P1]
        # prefill[iter 0] compute_logits BS2 [P0, P1]
        # prefill[iter 0] sampler BS2 [P0, P1] -> [tokP0, tokP1]
        # prefill[iter 1] forward pass BS1 [P0, P1]
        # prefill[iter 1] compute_logits BS1 [P0, P1]
        # prefill[iter 1] sampler BS1 [P0, P1] -> [tokP2]
        # prefill concat sampler results [tokP0, tokP1], [tokP2] -> [tokP0, tokP1, tokP2]
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2]
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        # Example2: Same thing, but with max_prefill_batch_size=4:
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding)
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1, P2, 0] (BS=4 bucket, 1 seq padding)
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS4 [P0, P1, P2, 0]
        # prefill[iter 0] compute_logits BS4 [P0, P1, P2, 0]
        # prefill[iter 0] sampler BS4 [P0, P1, P2, 0] -> [tokP0, tokP1, tokP2, 0]
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0]
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        # Example2: Same thing, but max_prefill_batch_size=4 and split_sampler=False:
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding)
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1, P2, 0] (BS=4 bucket, 1 seq padding)
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # prefill[iter 0] forward pass BS4 [P0, P1, P2, 0]
        # prefill[iter 0] compute_logits BS4 [P0, P1, P2, 0]
        # Join the prefill and decode on device into [D0, D1, D2, 0, P0, P1, P2, 0]
        # joint sampler BS8 [D0, D1, D2, 0, P0, P1, P2, 0] -> [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0]
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        self._update_states(scheduler_output)
        prefill_data, decode_data = self._prepare_inputs(
            scheduler_output, bucketing=self.enable_bucketing)
        num_reqs = self.input_batch.num_reqs
        num_decodes = decode_data.num_decodes
        num_prefills = num_reqs - num_decodes
        num_padded_decodes = decode_data.token_ids.shape[
            0] if num_decodes > 0 else 0

        #FIXME(kzawora): Currently there's no handling of logprobs. Fix that later.
        logprob_token_ids = None
        logprobs = None
        split_sampler = True
        prefill_output_device = None
        decode_output_device = None

        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_decode_bs, 1]
        if num_decodes > 0:
            htorch.core.mark_step()
            logits_device = self._execute_model_generic(
                decode_data.token_ids, decode_data.position_ids,
                decode_data.attn_metadata, decode_data.logits_indices, self.kv_caches)
            htorch.core.mark_step()
            if split_sampler:
                sampling_metadata = self._prepare_sampling(
                    scheduler_output,
                    start_idx=0,
                    end_idx=num_decodes,
                    pad_to=num_padded_decodes)
                htorch.core.mark_step()
                sampler_output = self.model.sample(
                    logits=logits_device, sampling_metadata=sampling_metadata)
                decode_output_device = sampler_output.sampled_token_ids
                htorch.core.mark_step()
            else:
                decode_output_device = logits_device
            htorch.core.mark_step()

        ######################### PREFILLS #########################
        # Prefills run with shape [padded_prefill_bs, padded_prefill_len]
        if num_prefills > 0:
            prefill_seq_offset_start = num_decodes
            htorch.core.mark_step()
            prefill_output_list = []
            for idx, (req_id, prompt_len, token_ids, position_ids,
                      attn_metadata,
                      logits_indices) in enumerate(prefill_data.zipped()):
                htorch.core.mark_step()
                logits_device = self._execute_model_generic(
                    token_ids, position_ids, attn_metadata, logits_indices, self.kv_caches)
                htorch.core.mark_step()
                if split_sampler:
                    num_curr_prefills = token_ids.shape[0]
                    prefill_seq_offset_end = prefill_seq_offset_start + num_curr_prefills
                    if prefill_seq_offset_start == prefill_seq_offset_end:
                        import pdb; pdb.set_trace()
                    sampling_metadata = self._prepare_sampling(
                        scheduler_output,
                        start_idx=prefill_seq_offset_start,
                        end_idx=prefill_seq_offset_end,
                        pad_to=num_curr_prefills)
                    htorch.core.mark_step()
                    sampler_output = self.model.sample(
                        logits=logits_device,
                        sampling_metadata=sampling_metadata)
                    sampled_token_ids_device = sampler_output.sampled_token_ids
                    htorch.core.mark_step()
                    prefill_seq_offset_end = prefill_seq_offset_start
                    prefill_output_list.append(sampled_token_ids_device)
                else:
                    prefill_output_list.append(logits_device)                    
            prefill_output_device = torch.cat(prefill_output_list, dim=0)
            htorch.core.mark_step()

        ################### (maybe) SAMPLING ###################
        #NOTE(kzawora): It might be better to do separate sampling
        # for prefills and decodes, since they will have more predictable
        # shapes. Or it might not. Idk. I implemented both.
        # In my testing, split_sampler=False was a bit faster (Llama3.1-8B@GSM8K),
        # no differences in accuracy observed. YMMV.
        # HPU <-> CPU sync happens in this section

        sampled_token_ids_cpu: torch.Tensor
        if split_sampler:
            # If sampler was split, we already have tokens. Let's copy the data to CPU as is, and then discard padded tokens.
            prefill_output_cpu = prefill_output_device.cpu(
            ) if prefill_output_device is not None else None
            decode_output_cpu = decode_output_device.cpu(
            ) if decode_output_device is not None else None
            # From this point onward, all operations are done on CPU.

            # Discard garbage tokens from prefills and/or decodes
            if prefill_output_cpu is not None and decode_output_cpu is not None:
                sampled_token_ids_cpu = torch.cat(
                    (decode_output_cpu[:num_decodes],
                     prefill_output_cpu[:num_prefills]),
                    dim=0)
            else:
                sampled_token_ids_cpu = decode_output_cpu[:
                                                          num_decodes] if decode_output_cpu is not None else prefill_output_cpu[:
                                                                                                                                num_prefills]
        else:
            # If sampler was not split, we need to sample on device before copying to CPU.
            joint_logits_device: torch.Tensor
            if decode_output_device is not None and prefill_output_device is not None:
                joint_logits_device = torch.cat(
                    (decode_output_device, prefill_output_device), dim=0)
            else:
                joint_logits_device = decode_output_device if decode_output_device is not None else prefill_output_device
            # NOTE(kzawora): this stuff is not gonna work
            assert False, "imma be real, this ain't gonna work chief"
            sampled_token_ids_device = torch.argmax(joint_logits_device, dim=1)
            sampled_token_ids_padded_cpu = sampled_token_ids_device.cpu()
            # From this point onward, all operations are done on CPU.

            # Discard garbage tokens from prefills and/or decodes
            # NOTE(kzawora): If we have 3 prefills and 3 decodes, and both
            # are padded to 4, the sampled tokens tensor looks as follows:
            # [ D0, D1, D2, 0, P0, P1, P2, 0]
            #   ^___^___^      ^___^___^
            # Here, we're selecting these elements and discard the
            # padding in the middle (after prefill tokens) and at the end of the
            # tensor (after decode tokens)
            # https://numpy.org/doc/stable/reference/generated/numpy.r_.html
            sampled_token_ids_cpu = sampled_token_ids_padded_cpu[
                np.r_[:num_decodes,
                      num_padded_decodes:num_padded_decodes + num_prefills]]

        sampled_token_ids_list = sampled_token_ids_cpu.tolist()
        ######### UPDATE REQUEST STATE WITH GENERATED TOKENS #########
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]

            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            token_id = sampled_token_ids_list[i]
            self.input_batch.token_ids_cpu[i, seq_len] = token_id
            req_state.output_token_ids.append(token_id)

        ################## RETURN ##################
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids_cpu=sampled_token_ids_cpu,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )

        if False:
            for req_id in self.input_batch.req_ids[:num_reqs]:
                req_idx = self.input_batch.req_id_to_index[req_id]
                token_ids = self.input_batch.token_ids_cpu[req_idx]
                prompt = self._tokenizer.decode(
                    token_ids[:self.input_batch.
                              num_prompt_tokens_cpu[req_idx]])

                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                req_state = self.requests[req_id]
                generated = self._tokenizer.decode(req_state.output_token_ids)
                phase = 'prefill' if req_idx >= decode_data.num_decodes else 'decode'
                logger.info(
                    f'[ENGINE_ITER {self._ENGINE_ITER}] REQ:{req_id} IDX:{req_idx} {phase} generated token: {self._tokenizer.decode(sampled_token_ids_cpu[req_idx])!r}, all generated so far: {generated!r}'
                )
        self._ENGINE_ITER += 1
        #import pdb; pdb.set_trace()
        return model_runner_output

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with HabanaMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            self.model = _maybe_wrap_in_hpu_graph(
                self.model,
                self.block_size,
                dtype=self.model_config.dtype,
                enforce_eager=self.model_config.enforce_eager)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    def _use_graphs(self, batch_size, seq_len, num_blocks, phase):
        if self.model_config.enforce_eager:
            return False
        if self.skip_warmup:
            return True
        return (batch_size, seq_len, num_blocks, phase) in self.graphed_buckets
    
    def log_graph_warmup_summary(self, buckets, is_prompt, total_mem):
        num_candidates = len(buckets)
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        graphed = list(c[:2] for c in self.graphed_buckets
                       if c[2] == is_prompt)
        if num_candidates == 0:
            num_candidates = 1
        msg = (f'{phase} captured:{len(graphed)} '
               f'({100 * len(graphed) / num_candidates:.1f}%) '
               f'used_mem:{format_bytes(total_mem)} '
               f'buckets:{sorted(list(graphed))}')
        logger.info(msg)
        
    def warmup_scenario(self,
                        batch_size,
                        seq_or_block,
                        is_prompt,
                        kv_caches) -> None:
        """Dummy warmup run for memory usage and graph compilation."""

        query_seq_len = seq_or_block if is_prompt else 1
        input_ids = torch.zeros((batch_size, query_seq_len),
                                dtype=torch.int32,
                                device='cpu')
        position_ids = torch.zeros((batch_size, query_seq_len),
                                dtype=torch.int32,
                                device='cpu')
        slot_mapping = torch.zeros((batch_size, query_seq_len),
                                dtype=torch.int64,
                                device='cpu')
        
        input_ids_device = _async_h2d_tensor_copy(input_ids, self.device)
        position_ids_device = _async_h2d_tensor_copy(position_ids, self.device)
        slot_mapping_device = _async_h2d_tensor_copy(slot_mapping, self.device)

        if is_prompt:
            seq_lens = torch.zeros((batch_size), dtype=torch.int32, device='cpu')
            seq_lens.fill_(seq_or_block)
            seq_lens_device = _async_h2d_tensor_copy(seq_lens, self.device)
            attn_metadata = HPUAttentionMetadata.make_prefill_metadata(
                seq_lens_tensor=seq_lens_device,
                num_prefills=batch_size,
                num_prefill_tokens=batch_size*seq_or_block,
                slot_mapping=slot_mapping_device
            )
        else:
            block_tables = [x.tolist() for x in np.array_split(np.arange(seq_or_block), batch_size)]
            block_list, block_groups, block_usage = self.get_habana_paged_attn_buffers(block_tables=block_tables, slot_mapping=slot_mapping, bucketing=True)
            block_list_device = _async_h2d_tensor_copy(block_list, self.device)
            block_usage_device = _async_h2d_tensor_copy(block_usage, self.device)
            block_groups_device = _async_h2d_tensor_copy(block_groups, self.device)
            attn_metadata = HPUAttentionMetadata.make_decode_metadata(
                block_list=block_list_device,
                block_usage=block_usage_device,
                block_groups=block_groups_device,
                num_decode_tokens=batch_size,
                slot_mapping=slot_mapping_device
            )
            
        logits_indices = torch.arange(0,batch_size, device='cpu')
        logits_indices_device = _async_h2d_tensor_copy(logits_indices, self.device)
        # Dummy run.
        htorch.core.mark_step()
        logits = self._execute_model_generic(input_ids_device, position_ids_device, attn_metadata, logits_indices_device, kv_caches, True)
        # TODO: do sampling on logits, warmup sampler and prefill joiner
        htorch.core.mark_step()
        temperature = torch.ones(batch_size, dtype=torch.float32, device='cpu')
        top_p = torch.ones(batch_size, dtype=torch.float32, device='cpu')
        top_k = torch.ones(batch_size, dtype=torch.float32, device='cpu')
        temperature_device = _async_h2d_tensor_copy(temperature, self.device)
        top_p_device = _async_h2d_tensor_copy(top_p, self.device)
        top_k_device = _async_h2d_tensor_copy(top_k, self.device)
        generators = {i:None for i in range(batch_size)} # NOTE(kzawora): idk what to set here
        max_num_logprobs = 0 # NOTE(kzawora): idk what to set here
        # NOTE(kzawora: do this in a smarter way)
        return None
        htorch.core.mark_step()
        sampling_metadata = SamplingMetadata(
            temperature=temperature_device,
            all_greedy=False, # hacky
            all_random=True, # hacky
            top_p=top_p_device,
            top_k=top_k_device,
            no_top_p=True,
            no_top_k=True,
            generators=generators,
            max_num_logprobs=max_num_logprobs,
        )
        tokens_all_random = self.model.sample(logits, sampling_metadata)
        htorch.core.mark_step()
        sampling_metadata = SamplingMetadata(
            temperature=temperature_device,
            all_greedy=True, # hacky
            all_random=False, # hacky
            top_p=top_p_device,
            top_k=top_k_device,
            no_top_p=True,
            no_top_k=True,
            generators=generators,
            max_num_logprobs=max_num_logprobs,
        )
        tokens_all_greedy = self.model.sample(logits, sampling_metadata)
        htorch.core.mark_step()
        sampling_metadata = SamplingMetadata(
            temperature=temperature_device,
            all_greedy=False, # hacky
            all_random=False, # hacky
            top_p=top_p_device,
            top_k=top_k_device,
            no_top_p=True,
            no_top_k=True,
            generators=generators,
            max_num_logprobs=max_num_logprobs,
        )
        tokens_mixed = self.model.sample(logits, sampling_metadata)
        htorch.core.mark_step()
        return tokens_all_random, tokens_all_greedy, tokens_mixed

    def log_warmup(self, phase, i, max_i, batch_size, seq_len):
        free_mem = format_bytes(
            HabanaMemoryProfiler.current_free_device_memory())
        dim = "num_blocks"
        if phase == "Prompt":
            dim = "seq_len"
        msg = (f"[Warmup][{phase}][{i+1}/{max_i}] "
               f"batch_size:{batch_size} "
               f"{dim}:{seq_len} "
               f"free_mem:{free_mem}")
        logger.info(msg)

    def warmup_all_buckets(self, buckets, is_prompt, kv_caches):
        for i, (batch_size, seq_len) in enumerate(reversed(buckets)):
            self.log_warmup('Prompt' if is_prompt else 'Decode', i,
                            len(buckets), batch_size, seq_len)
            self.warmup_scenario(batch_size, seq_len, is_prompt, kv_caches)
            torch.hpu.synchronize()

    def warmup_graphs(self,
                      strategy,
                      buckets,
                      is_prompt,
                      kv_caches,
                      available_mem,
                      starting_mem=0,
                      total_batch_seq=0.001):
        total_mem = starting_mem
        idx = 0
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        num_candidates = len(buckets)
        ordering : Union[Callable[[Any], Tuple[Any, Any]], \
            Callable[[Any], Tuple[Any, Any, Any]]]
        if strategy == 'min_tokens':
            ordering = lambda b: (b[0] * b[1], b[1], b[0])
        elif strategy == 'max_bs':
            ordering = lambda b: (-b[0], b[1])
        else:
            raise NotImplementedError(
                f'Unsupported graph allocation strategy: {strategy}')
        buckets = list(sorted(buckets, key=ordering))
        captured_all = True
        for idx, (batch_size, seq_len) in enumerate(buckets):
            # Graph memory usage is proportional to seq dimension in a batch
            batch_seq = batch_size * seq_len if is_prompt else batch_size
            mem_estimate = batch_seq / total_batch_seq * total_mem
            if mem_estimate >= available_mem:
                captured_all = False
                continue
            graphed_bucket = (batch_size, seq_len, is_prompt)
            if graphed_bucket in self.graphed_buckets:
                continue
            self.graphed_buckets.add(graphed_bucket)
            self.log_warmup(phase, idx, num_candidates, batch_size, seq_len)
            with HabanaMemoryProfiler() as mem_prof:
                self.warmup_scenario(batch_size, seq_len, is_prompt, kv_caches)
            #TODO(kzawora): align_workers
            used_mem = mem_prof.consumed_device_memory
            available_mem -= used_mem
            total_mem += used_mem
            total_batch_seq += batch_seq

        return total_mem, total_batch_seq, captured_all

    @torch.inference_mode()
    def warmup_model(self) -> None:
        kv_caches = self.kv_caches
        if profile := os.environ.get('VLLM_PT_PROFILE', None):
            phase, bs, seq_len, graph = profile.split('_')
            is_prompt = phase == 'prompt'
            graphs = graph == 't'
            if graphs:
                self.graphed_buckets.add((int(bs), int(seq_len), is_prompt))
            self.warmup_scenario(int(bs), int(seq_len), is_prompt, kv_caches,
                                 True)
            raise AssertionError("Finished profiling")
        if self.skip_warmup:
            logger.info("Skipping warmup...")
            return
        #self.profiler.start('internal', 'warmup')
        max_blocks = kv_caches[0][0].size(0)
        self.bucketing_ctx.generate_prompt_buckets()
        self.bucketing_ctx.generate_decode_buckets(max_blocks)

        if not htorch.utils.internal.is_lazy() and not self.enforce_eager:
            cache_size_limit = len(self.bucketing_ctx.prompt_buckets) + len(
                self.bucketing_ctx.decode_buckets) + 1
            torch._dynamo.config.cache_size_limit = max(
                cache_size_limit, torch._dynamo.config.cache_size_limit)
            # Multiply by 8 to follow the original default ratio between
            # the cache_size_limit and accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = max(
                cache_size_limit * 8,
                torch._dynamo.config.accumulated_cache_size_limit)

        start_mem = HabanaMemoryProfiler.current_device_memory_usage()
        start_time = time.perf_counter()

        compile_only_mode_context = functools.partial(bc.env_setting,
                                                      "PT_COMPILE_ONLY_MODE",
                                                      True)
        can_use_compile_only_mode = True
        try:
            with compile_only_mode_context():
                pass
            logger.debug("Using PT_COMPILE_ONLY_MODE.")
        except KeyError:
            can_use_compile_only_mode = False
            logger.warning('Cannot use PT_COMPILE_ONLY_MODE. '
                           'Warmup time will be negatively impacted. '
                           'Please update Gaudi Software Suite.')
        with compile_only_mode_context(
        ) if can_use_compile_only_mode else contextlib.nullcontext():
            self.warmup_all_buckets(self.bucketing_ctx.prompt_buckets,
                                    True, kv_caches)
            self.warmup_all_buckets(self.bucketing_ctx.decode_buckets,
                                    False, kv_caches)

            if not self.model_config.enforce_eager and htorch.utils.internal.is_lazy():
                assert self.mem_margin is not None, \
                    ("HabanaWorker.determine_num_available_blocks needs "
                    "to be called before warming up the model.")
                free_mem = HabanaMemoryProfiler.current_free_device_memory()
                graph_free_mem = free_mem - self.mem_margin
                #TODO(kzawora): align_workers
                graph_free_mem = graph_free_mem
                prompt_graph_mem_ratio = float(
                    os.environ.get('VLLM_GRAPH_PROMPT_RATIO', '0.3'))
                prompt_available_memory = (prompt_graph_mem_ratio *
                                           graph_free_mem)
                decode_available_memory = (graph_free_mem -
                                           prompt_available_memory)
                msg = (
                    f"Using {format_bytes(graph_free_mem)}"
                    f"/{format_bytes(free_mem)} "
                    "of free device memory for HPUGraphs, "
                    f"{format_bytes(prompt_available_memory)} for prompt and "
                    f"{format_bytes(decode_available_memory)} for decode "
                    f"(VLLM_GRAPH_PROMPT_RATIO={prompt_graph_mem_ratio})")
                logger.info(msg)
                prompt_strategy = os.environ.get('VLLM_GRAPH_PROMPT_STRATEGY',
                                                 'min_tokens')
                decode_strategy = os.environ.get('VLLM_GRAPH_DECODE_STRATEGY',
                                                 'max_bs')
                mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                    self.warmup_graphs(
                    prompt_strategy, self.bucketing_ctx.prompt_buckets,
                    True, kv_caches, prompt_available_memory)
                mem_post_decode, decode_batch_seq, decode_captured_all = \
                    self.warmup_graphs(
                    decode_strategy, self.bucketing_ctx.decode_buckets,
                    False, kv_caches, decode_available_memory)

                # Not all prompt buckets were captured, but all decode buckets
                # were captured and we have some free graph-allocated space
                # left. Let's try to use it for capturing more prompt buckets.
                if (mem_post_decode + mem_post_prompt < graph_free_mem
                        and not prompt_captured_all and decode_captured_all):
                    mem_post_prompt, _, prompt_captured_all = (
                        self.warmup_graphs(
                            prompt_strategy,
                            self.bucketing_ctx.prompt_buckets, True,
                            kv_caches,
                            graph_free_mem - mem_post_prompt - mem_post_decode,
                            mem_post_prompt, prompt_batch_seq))

                # Not all decode buckets were captured, but all prompt buckets
                # were captured and we have some free graph-allocated space
                # left. Let's try to use it for capturing more decode buckets.
                if mem_post_decode + mem_post_prompt < graph_free_mem \
                    and not decode_captured_all \
                        and prompt_captured_all:
                    mem_post_decode, _, _ = self.warmup_graphs(
                        decode_strategy,
                        self.bucketing_ctx.decode_buckets, False,
                        kv_caches,
                        graph_free_mem - mem_post_prompt - mem_post_decode,
                        mem_post_decode, decode_batch_seq)

                self.log_graph_warmup_summary(
                    self.bucketing_ctx.prompt_buckets, True,
                    mem_post_prompt)
                self.log_graph_warmup_summary(
                    self.bucketing_ctx.decode_buckets, False,
                    mem_post_decode)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        elapsed_time = end_time - start_time
        msg = (
            f"Warmup finished in {elapsed_time:.0f} secs, "
            f"allocated {format_bytes(end_mem - start_mem)} of device memory")
        logger.info(msg)
        #self.profiler.end()

    @torch.inference_mode()
    def profile_run(self) -> None:
        return
        """Profile to measure peak memory during forward pass."""

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers

        # Run empty prefill forwards - prefill max batch and prefill max seq 
        self.warmup_scenario(batch_size=1,
                        seq_or_block=self.max_model_len,
                        is_prompt=True,
                        kv_caches=kv_caches)
        max_seq_len = math.ceil((self.max_num_tokens//self.max_prefill_batch_size) / self.block_size) * self.block_size
        self.warmup_scenario(batch_size=self.max_prefill_batch_size,
                        seq_or_block=max_seq_len,
                        is_prompt=True,
                        kv_caches=kv_caches)

    @torch.inference_mode()
    def capture_model(self) -> None:
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with set_forward_context(None):
            # Trigger CUDA graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            for num_tokens in reversed(self.cudagraph_batch_sizes):
                self.model(
                    self.input_ids[:num_tokens],
                    self.positions[:num_tokens],
                    kv_caches=self.kv_caches,
                    attn_metadata=None,
                )

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = HPUAttentionBackendV1.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        dtype = self.dtype
        if self.device != 'hpu' and not is_fake_hpu() \
          and self.dtype == torch.float8_e4m3fn:
            dtype = torch.uint8
        for _ in range(self.num_attn_layers):
            key_cache = torch.zeros(kv_cache_shape,
                                    dtype=dtype,
                                    device=self.device)
            value_cache = torch.zeros(kv_cache_shape,
                                      dtype=dtype,
                                      device=self.device)
            kv_layer = (key_cache, value_cache)
            self.kv_caches.append(kv_layer)
        if self.enable_bucketing:
            self.bucketing_ctx.num_hpu_blocks = num_blocks
        htorch.hpu.synchronize()


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List[MultiModalKwargs]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    block_ids: List[int]
    num_computed_tokens: int
    output_token_ids: List[int]

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        self.pin_memory = pin_memory

        self.req_ids: List[Optional[str]] = [None] * max_num_reqs
        self.req_id_to_index: Dict[str, int] = {}

        self.token_ids_cpu = np.empty((max_num_reqs, max_model_len),
                                      dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_output_tokens_cpu = np.empty(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens_cpu = np.empty(max_num_reqs, dtype=np.int32)

        # Attention-related.
        self.block_table = torch.zeros((max_num_reqs, max_num_blocks_per_req),
                                       device=self.device,
                                       dtype=torch.int32)
        self.block_table_cpu_tensor = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_cpu = self.block_table_cpu_tensor.numpy()

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs, ),
                                       dtype=torch.float32,
                                       device=device)
        self.temperature_cpu_tensor = torch.empty((max_num_reqs, ),
                                                  dtype=torch.float32,
                                                  device="cpu",
                                                  pin_memory=pin_memory)
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: Set[str] = set()
        self.random_reqs: Set[str] = set()

        self.top_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: Set[str] = set()

        self.top_k = torch.empty((max_num_reqs, ),
                                 dtype=torch.int32,
                                 device=device)
        self.top_k_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: Set[str] = set()

        # req_index -> generator
        self.generators: Dict[int, torch.Generator] = {}

        self.num_logprobs: Dict[str, int] = {}
        self.prompt_logprob_reqs: Set[str] = set()

        self.num_prefills = 0

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        self.req_ids[req_index] = req_id
        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        #self.num_output_tokens_cpu[req_index] = request.num_output_tokens
        self.num_prompt_tokens_cpu[req_index] = len(request.prompt_token_ids)
        num_blocks = len(request.block_ids)
        self.block_table_cpu[req_index, :num_blocks] = request.block_ids

        sampling_params = request.sampling_params
        self.temperature_cpu[req_index] = sampling_params.temperature
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.greedy_reqs.add(req_id)
        else:
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        self.top_k_cpu[req_index] = sampling_params.top_k
        if sampling_params.top_k > 0:
            self.top_k_reqs.add(req_id)

        self.generators[req_index] = request.generator

        num_logprobs = sampling_params.logprobs
        if num_logprobs is not None and num_logprobs > 0:
            self.num_logprobs[req_id] = num_logprobs
        if sampling_params.prompt_logprobs:
            self.prompt_logprob_reqs.add(req_id)

    def remove_request(self, req_id: str) -> Optional[int]:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.req_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.prompt_logprob_reqs.discard(req_id)
        return req_index

    def clear(self) -> None:
        self.req_ids = [None] * self.max_num_reqs
        self.req_id_to_index.clear()
        self.greedy_reqs.clear()
        self.random_reqs.clear()
        self.top_p_reqs.clear()
        self.top_k_reqs.clear()
        self.generators.clear()
        self.num_logprobs.clear()
        self.prompt_logprob_reqs.clear()

    def condense(self, empty_req_indices: List[int]) -> None:
        if self.num_reqs == 0:
            # The batched states are empty.
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = self.num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Swap the states.
            req_id = self.req_ids[last_req_index]
            self.req_ids[empty_index] = req_id
            self.req_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # TODO(woosuk): Optimize the copy of token_ids_cpu and
            # block_table_cpu.
            self.token_ids_cpu[empty_index] = self.token_ids_cpu[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table_cpu[empty_index] = self.block_table_cpu[
                last_req_index]
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

    def make_sampling_metadata(self,
                               skip_copy,
                               start_idx: Optional[int] = None,
                               end_idx: Optional[int] = None,
                               pad_to: Optional[int] = None):
        if start_idx is None and end_idx is None and pad_to is None:
            return self._make_sampling_metadata_all(skip_copy=skip_copy)
        return self._make_sampling_metadata_range(skip_copy,
                                                  start_idx,
                                                  end_idx,
                                                  pad_to=pad_to)

    def _make_sampling_metadata_range(
            self,
            skip_copy: bool = False,
            start_idx: Optional[int] = None,
            end_idx: Optional[int] = None,
            pad_to: Optional[int] = None) -> SamplingMetadata:
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = self.num_reqs
        max_num_reqs = len(self.req_ids)
        end_idx = min(end_idx,max_num_reqs)
        num_seqs = end_idx - start_idx
        padding_needed = max(0, pad_to - num_seqs)
        req_ids = self.req_ids[start_idx:end_idx]
        if not skip_copy:
            self.temperature[start_idx:end_idx].copy_(
                self.temperature_cpu_tensor[start_idx:end_idx],
                non_blocking=True)
            self.top_p[start_idx:end_idx].copy_(
                self.top_p_cpu_tensor[start_idx:end_idx], non_blocking=True)
            self.top_k[start_idx:end_idx].copy_(
                self.top_k_cpu_tensor[start_idx:end_idx], non_blocking=True)

        all_greedy = all([req_id in self.greedy_reqs for req_id in req_ids])
        all_random = all([req_id in self.random_reqs for req_id in req_ids])
        if all_greedy and all_random:
            import pdb
            pdb.set_trace()  #WTF?!
        no_top_p = not any([req_id in self.top_p_reqs for req_id in req_ids])
        no_top_k = not any([req_id in self.top_k_reqs for req_id in req_ids])
        # NOTE(kzawora): Generators are used by sampler row-wise. If we got a
        # generator for element 5, but it's first row in a batch,
        # we need to assign that generator to index 0 - hence the
        # i:generators.get(req_id) rather than req_id:generators.get(req_id)
        generators = {
            i: self.generators.get(req_id, None)
            for i, req_id in enumerate(
                range(start_idx, end_idx + padding_needed))
        }
        temperature_device = self.temperature[start_idx:end_idx +
                                              padding_needed]
        top_p_device = self.top_p[start_idx:end_idx + padding_needed]
        tok_k_device = self.top_k[start_idx:end_idx + padding_needed]
        if end_idx + padding_needed >= max_num_reqs:
            # NOTE(kzawora): this is janky, but [start_idx:end_idx+padding_needed]
            # falls apart once your padding exceeds max_num_reqs (and it happens pretty
            # often, you could increase the temperature/topp/topk allocation, but
            # you cannot really make any guarantees ahead of time on the amount of padding you'll use)
            # this is kind of a temporary fix, no idea on its performance impact...
            temperature_device = torch.empty(pad_to,
                                             device=self.temperature.device,
                                             dtype=self.temperature.dtype)
            top_p_device = torch.empty(pad_to,
                                       device=self.top_p.device,
                                       dtype=self.top_p.dtype)
            top_k_device = torch.empty(pad_to,
                                       device=self.top_k.device,
                                       dtype=self.top_k.dtype)
            # D2D copy
            temperature_device[:num_seqs].copy_(
                self.temperature[start_idx:end_idx], non_blocking=True)
            top_p_device[:num_seqs].copy_(self.top_p[start_idx:end_idx],
                                          non_blocking=True)
            top_k_device[:num_seqs].copy_(self.top_k[start_idx:end_idx],
                                          non_blocking=True)

        return SamplingMetadata(
            temperature=temperature_device,
            all_greedy=all_greedy,
            all_random=all_random,
            top_p=top_p_device,
            top_k=tok_k_device,
            no_top_p=no_top_p,
            no_top_k=no_top_k,
            generators=generators,
            max_num_logprobs=self.max_num_logprobs,
        )

    def _make_sampling_metadata_all(
        self,
        skip_copy: bool = False,
    ) -> SamplingMetadata:
        if not skip_copy:
            self.temperature[:self.num_reqs].copy_(
                self.temperature_cpu_tensor[:self.num_reqs], non_blocking=True)
            self.top_p[:self.num_reqs].copy_(
                self.top_p_cpu_tensor[:self.num_reqs], non_blocking=True)
            self.top_k[:self.num_reqs].copy_(
                self.top_k_cpu_tensor[:self.num_reqs], non_blocking=True)
        return SamplingMetadata(
            temperature=self.temperature[:self.num_reqs],
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=self.top_p[:self.num_reqs],
            top_k=self.top_k[:self.num_reqs],
            no_top_p=self.no_top_p,
            no_top_k=self.no_top_k,
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def num_decodes(self) -> int:
        return self.num_reqs - self.num_prefills

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def max_num_logprobs(self) -> int:
        return max(self.num_logprobs.values()) if self.num_logprobs else 0

    @property
    def no_logprob(self) -> bool:
        return len(self.num_logprobs) == 0

    @property
    def no_prompt_logprob(self) -> bool:
        return len(self.prompt_logprob_reqs) == 0
