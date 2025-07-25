# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MooncakeStore Connector for Distributed Machine Learning Inference
The MooncakeStoreConnector transfers KV caches between prefill vLLM workers
(KV cache producer) and decode vLLM workers (KV cache consumer) using a
database-style KVStore.
"""
import hashlib
from typing import TYPE_CHECKING, Union

import torch

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.utils import (
    model_aware_kv_ops_helper as kv_helper)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import (
        ModelInputForGPUWithSamplingMetadata, )
    from vllm.worker.hpu_model_runner import (
        ModelInputForHPUWithSamplingMetadata, )

import habana_frameworks.torch as htorch
from vllm_hpu_extension.utils import VLLMKVCache

logger = init_logger(__name__)


class MooncakeStoreConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.kv_transfer_config = config.kv_transfer_config
        self.kv_helper = kv_helper(config)
        self.local_tp_rank = local_rank
        self.rank = rank
        self.block_size = config.cache_config.block_size
        # Init kv_store
        if self.kv_transfer_config.kv_connector == "MooncakeStoreConnector":
            # Check if MOONCAKE_CONFIG_PATH is set
            import os
            use_mooncake_store = os.getenv('MOONCAKE_CONFIG_PATH') is not None

            if not use_mooncake_store:
                raise ValueError(
                    "To use MooncakeStoreConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                from vllm.distributed.kv_transfer.kv_lookup_buffer.mooncake_store import (  # noqa: E501
                    MooncakeStore)
                logger.info(
                    "Initializing KVStoreConnector under kv_transfer_config %s",
                    self.kv_transfer_config)
                self.kv_store = MooncakeStore(config)
        else:
            logger.error("Can not find %s",
                         self.kv_transfer_config.kv_connector)

        assert self.kv_store is not None
        if config.cache_config.cache_dtype == "fp8_inc":
            self.dtype = torch.float8_e4m3fn
        else:
            self.dtype = torch.bfloat16
        self.cache_k = VLLMKVCache()
        self.cache_v = VLLMKVCache()

    def close(self) -> None:
        """Close the buffer and release resources.
        This method is responsible for cleaning up resources related to the 
        connector when it is no longer needed.
        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        self.kv_store.close()

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_heads, head_size = self.kv_helper.get_model_args(model_executable)

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            store_key_prefix = self.tensor_hash(current_tokens)
            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]
                key_cache, value_cache = self.kv_helper.get_kv_from_cache(
                    kv_cache, num_heads, head_size)
                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            kvcache_to_sent = torch.stack((keys, values), dim=0)
            store_kvcache_key = f"{store_key_prefix}_{self.local_tp_rank}"
            self.kv_store.put(store_kvcache_key, kvcache_to_sent)

            hidden_key = f"{store_key_prefix}_hidden_{self.local_tp_rank}"
            self.kv_store.put(hidden_key,
                              hidden_or_intermediate_states[start_pos:end_pos])

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        bypass_model_exec = True
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        hidden_or_intermediate_states_for_one_req = []

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # This can happen during inflight batching. See:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You should set --enable_chunked_prefill=False "
                               "and --max_num_batched_tokens "
                               "should be equal to max_seq_len_to_capture")
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]

            # get roi for current seq
            load_key_prefix = self.tensor_hash(current_tokens)
            load_kvcache_key = f"{load_key_prefix}_{self.local_tp_rank}"
            remote_kv = self.kv_store.get(load_kvcache_key)
            hidden_key = f"{load_key_prefix}_hidden_{self.local_tp_rank}"
            hidden = self.kv_store.get(hidden_key)

            if remote_kv is None or hidden is None:
                # didn't find any match.
                bypass_model_exec = False
                continue

            num_computed_tokens = current_tokens.shape[0]

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # call self.kv_store to get kv layer by layer
            for layer_id in range(start_layer, end_layer):
                layer = model_executable.model.layers[layer_id]
                # get kvcache object
                kv_cache = kv_caches[layer_id - start_layer]

                # get remote kvcache
                remote_k, remote_v = remote_kv[0][layer_id], remote_kv[1][
                    layer_id]

                self.kv_helper.put_kv_to_cache(model_executable, remote_k,
                                               remote_v, layer, kv_cache,
                                               slot_mapping, start_pos,
                                               end_pos)

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            logger.warning(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states_hpu(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForHPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        if self.rank != 0 and self.kv_helper.use_mla():
            # only the first rank will send kv cache when using MLA.
            return
        input_tokens_tensor_cpu = model_input.input_tokens.to(
            "cpu")  # shape: [batch_size, seq_len_padding_to_128]
        torch.hpu.synchronize()
        seq_lens = model_input.attn_metadata.seq_lens  # 2D list
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_heads, head_size = self.kv_helper.get_model_args(model_executable)

        for idx, slen in enumerate(seq_lens):
            if slen == 1:  # we think this is a padding sequence, so we skip it
                continue

            current_tokens_cpu = input_tokens_tensor_cpu[idx][:slen]
            store_key_prefix = self.tensor_hash(current_tokens_cpu)
            logger.debug("send token len: %s, token: %s", slen,
                         current_tokens_cpu)

            padded_total_size = (slen + self.block_size -
                                 1) // self.block_size * self.block_size
            current_slot_mapping = model_input.attn_metadata.slot_mapping[
                idx][:padded_total_size]
            htorch.core.mark_step()
            # ==== graph should start here ======
            keys, values = [], []
            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]
                if self.kv_helper.use_mla():
                    # for MLA model,  we hold only one tensor to store
                    # both K and V cache. so Value cache tensor is empty.
                    key_cache = kv_cache[0]
                    keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                else:
                    key_cache, value_cache = kv_cache[0], kv_cache[1]
                    keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                    values.append(
                        value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            if self.kv_helper.use_mla():
                # we pack kv together, only need send one tensor
                kvcache_to_sent = keys
                store_kvcache_key = f"{store_key_prefix}_{self.rank}"
                hidden_key = f"{store_key_prefix}_hidden_{self.rank}"
            else:
                values = torch.cat(values, dim=0)
                kvcache_to_sent = torch.stack((keys, values), dim=0)
                store_kvcache_key = f"{store_key_prefix}_{self.local_tp_rank}"
                hidden_key = f"{store_key_prefix}_hidden_{self.local_tp_rank}"

            self.kv_store.put(store_kvcache_key, kvcache_to_sent)
            logger.debug("put kv cache key: %s", store_kvcache_key)

            self.kv_store.put(
                hidden_key,
                hidden_or_intermediate_states[idx].unsqueeze(0).cpu())
            # ==== graph should end here ======
            htorch.core.mark_step()

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states_hpu(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForHPUWithSamplingMetadata",
        attn_metadata: AttentionMetadata, kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForHPUWithSamplingMetadata"]:
        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True
        input_tokens_tensor_cpu = model_input.input_tokens.to("cpu")
        torch.hpu.synchronize()

        seq_lens = model_input.attn_metadata.seq_lens_tensor.tolist()
        slot_mapping = attn_metadata.slot_mapping
        hidden_or_intermediate_states_for_one_req: list[torch.Tensor] = []
        start_block_idx = 0
        num_heads, head_size = self.kv_helper.get_model_args(model_executable)

        for idx, slen in enumerate(seq_lens):
            current_tokens = input_tokens_tensor_cpu[idx][:slen]
            padded_total_size = (slen + self.block_size -
                                 1) // self.block_size * self.block_size
            current_slot_mapping = slot_mapping[idx][:padded_total_size]
            num_blocks = (slen + 127) // 128
            end_block_idx = start_block_idx + num_blocks

            # we think this is a padding sequence, so we skip it.
            # but we still need write kv cache.
            if slen == 1:
                for i in range(model_executable.model.start_layer,
                               model_executable.model.end_layer):
                    cur_layer_idx = i - model_executable.model.start_layer
                    kv_cache = kv_caches[cur_layer_idx]
                    key_cache, value_cache = kv_cache[0], kv_cache[1]
                    if self.kv_helper.use_mla():
                        padding_k_tensor = torch.zeros(
                            (self.block_size, head_size),
                            dtype=self.dtype,
                            device="hpu")
                        self.cache_k(padding_k_tensor, key_cache,
                                     current_slot_mapping)
                    else:
                        padding_k_tensor = torch.zeros(
                            (self.block_size, num_heads, head_size),
                            dtype=self.dtype,
                            device="hpu")
                        padding_v_tensor = torch.zeros(
                            (self.block_size, num_heads, head_size),
                            dtype=self.dtype,
                            device="hpu")
                        self.cache_k(padding_k_tensor, key_cache,
                                     current_slot_mapping)
                        self.cache_v(padding_v_tensor, value_cache,
                                     current_slot_mapping)
                # the first one should never be padding,
                # so we can append the first one.
                if len(hidden_or_intermediate_states_for_one_req):
                    hidden_or_intermediate_states_for_one_req.append(
                        hidden_or_intermediate_states_for_one_req[0])
                start_block_idx = end_block_idx
                continue

            # get roi for current seq
            load_key_prefix = self.tensor_hash(current_tokens)
            if self.kv_helper.use_mla():
                # For deepseek, we only need recv first rank
                load_kvcache_key = f"{load_key_prefix}_0"
                hidden_key = f"{load_key_prefix}_hidden_0"
            else:
                load_kvcache_key = f"{load_key_prefix}_{self.local_tp_rank}"
                hidden_key = f"{load_key_prefix}_hidden_{self.local_tp_rank}"

            remote_kv = self.kv_store.get(load_kvcache_key)
            hidden = self.kv_store.get(hidden_key)

            if remote_kv is None or hidden is None:
                logger.warning("Didn't find any match, load_key_prefix: %s",
                               load_kvcache_key)
                bypass_model_exec = False
                continue

            htorch.core.mark_step()
            torch.hpu.synchronize()

            # put received KV caches into paged memory layer by layer
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):
                cur_layer_idx = i - model_executable.model.start_layer
                kv_cache = kv_caches[cur_layer_idx]
                key_cache, value_cache = kv_cache[0], kv_cache[1]
                if self.kv_helper.use_mla():
                    remote_k = remote_kv[cur_layer_idx]
                    self.cache_k(remote_k, key_cache, current_slot_mapping)
                else:
                    remote_k, remote_v = remote_kv[0][i], remote_kv[1][i]
                    self.cache_k(remote_k, key_cache, current_slot_mapping)
                    self.cache_v(remote_v, value_cache, current_slot_mapping)

            hidden_or_intermediate_states_for_one_req.append(hidden.to("hpu"))
            start_block_idx = end_block_idx
            htorch.core.mark_step()

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.warning(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None
        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0).to("hpu")

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    @staticmethod
    def tensor_hash(tensor: torch.Tensor) -> int:
        """Calculate the hash value of the tensor."""
        tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
        hash_object = hashlib.blake2b(tensor_bytes)
        hash_hex = hash_object.hexdigest()
        return int(hash_hex[:16], 16)
