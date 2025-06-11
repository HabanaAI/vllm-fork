# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2.5-Omni model (thinker part)."""

import math
from copy import copy
from functools import partial
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Set, Tuple, Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniConfig, Qwen2_5OmniThinkerConfig)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer, Qwen2_5_VisionTransformerStaticShape,
    Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs, Qwen2_5_VLProcessingInfo,
    Qwen2_5_VLVideoEmbeddingInputs, Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs)
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioInputs, Qwen2AudioProcessingInfo,
    _get_feat_extract_output_lengths)
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (AudioProcessorItems, DictEmbeddingItems,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        PlaceholderFeaturesInfo,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import _Backend, current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import decode_tokens, encode_tokens

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import get_vit_attn_backend

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)
is_hpu = current_platform.is_hpu()


def _qwen2_5_omni_thinker_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    audio_feature_lengths = hf_inputs.get("audio_feature_lengths",
                                          torch.empty((0, )))

    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)

    video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
    video_grid_sizes = video_grid_thw.prod(-1)

    return dict(
        input_audio_features=MultiModalFieldConfig.flat_from_sizes(
            "audio", audio_feature_lengths, dim=1),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
        pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_embeds=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_grid_thw=MultiModalFieldConfig.batched("video"),
        second_per_grid_ts=MultiModalFieldConfig.batched("video"),
    )


class Qwen2_5OmniThinkerMultiModalDataParser(Qwen2VLMultiModalDataParser):

    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={
                    "input_audio_features", "audio_feature_lengths"
                },
                fields_factory=_qwen2_5_omni_thinker_field_config,
            )

        return super()._parse_audio_data(data)


class Qwen2_5OmniAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen2_5OmniAudioEncoderConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {self.num_heads}).")
        self.scaling = self.head_dim**-0.5
        self.is_decoder = False
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Detect attention implementation.
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
        if self.attn_backend not in {_Backend.FLASH_ATTN, _Backend.TORCH_SDPA}:
            raise RuntimeError(
                f"Qwen2.5-VL does not support {self.attn_backend} backend now."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        seq_length, all_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states).\
            reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).\
            reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).\
            reshape(seq_length, self.num_heads, -1)

        if self.attn_backend == _Backend.FLASH_ATTN:
            from flash_attn import flash_attn_varlen_func

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output = flash_attn_varlen_func(query_states,
                                                 key_states,
                                                 value_states,
                                                 cu_seqlens,
                                                 cu_seqlens,
                                                 max_seqlen,
                                                 max_seqlen,
                                                 dropout_p=0.0)
            attn_output = attn_output.reshape(seq_length, all_dim)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            if attention_mask is None:
                attention_mask = torch.zeros(
                    [1, seq_length, key_states.shape[0]],
                    device=query_states.device,
                    dtype=torch.bool)
                for i in range(1, len(cu_seqlens)):
                    attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                                   cu_seqlens[i - 1]:cu_seqlens[i]] = True

            query_states = query_states.transpose(0, 1)
            key_states = key_states.transpose(0, 1)
            value_states = value_states.transpose(0, 1)

            if is_hpu:
                from habana_frameworks.torch.hpex.kernels import FusedSDPA
                attn_output = FusedSDPA.apply(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    0.0,
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                )
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output


class Qwen2_5OmniAudioEncoderLayer(nn.Module):

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen2_5OmniAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = _ACTIVATION_REGISTRY[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape
                `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are
                indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in
                a given layer of size `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all
                attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 or \
           hidden_states.dtype == torch.bfloat16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_value,
                                        max=clamp_value)

        outputs = (hidden_states, )

        return outputs


class SinusoidsPositionEmbedding(nn.Module):

    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(
                "SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = \
          torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = \
          torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time),
                       torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniAudioEncoder(nn.Module):

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding \
                                                else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins,
                               embed_dim,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(embed_dim,
                               embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.positional_embedding = \
            SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = \
          nn.ModuleList([
              Qwen2_5OmniAudioEncoderLayer(config)
              for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device

    def postprocess_hpu(
        self,
        padded_token_audio,
        padded_aftercnn_lens,
        aftercnn_lens,
    ):
        # avgpool.stride is 2
        padded_aftercnn_lens = padded_aftercnn_lens // 2
        aftercnn_lens = aftercnn_lens // 2
        padded_token_audio_list = \
          padded_token_audio.split(padded_aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for i, each_audio_states in enumerate(padded_token_audio_list):
            token_audio_list.append(each_audio_states[:aftercnn_lens[i], :])
        token_audio = torch.cat(token_audio_list, dim=0)

        return token_audio

    def forward(
        self,
        input_features=None,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2,
                                    chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = \
          self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        cu_seqlens = torch.cat((
            torch.zeros(1,
                        device=padded_mask_after_cnn.device,
                        dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )).to(torch.int32)

        padded_embed = \
          nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = \
          nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + \
            self.positional_embedding.positional_embedding[
                : padded_embed.shape[1], :
            ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat((
            torch.zeros(1,
                        device=padded_mask_after_cnn.device,
                        dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )).to(torch.int32)

        for _, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(
                each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        return token_audio

    def padded_and_mask_function(self,
                                 tensor_list,
                                 tensor_len,
                                 padding_value=0,
                                 padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self,
                                         input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output
        length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class Qwen2_5OmniAudioEncoderStaticShape(Qwen2_5OmniAudioEncoder):

    def pre_attn(
        self,
        input_features,
        feature_lens,
        audio_buckets,
    ):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2,
                                    chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = \
          self.padded_and_mask_function(
              chunk_list, chunk_lengths, padding_value=0, audio_buckets=audio_buckets,
          )

        cu_seqlens = torch.cat((
            torch.zeros(1,
                        device=padded_mask_after_cnn.device,
                        dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )).to(torch.int32)

        padded_mask_after_cnn = torch.fill(padded_mask_after_cnn, True)
        attention_mask = torch.zeros(
            [1, padded_mask_after_cnn.sum(),
             padded_mask_after_cnn.sum()],
            device=input_features.device,
            dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                           cu_seqlens[i - 1]:cu_seqlens[i]] = True

        return padded_feature, padded_mask, padded_mask_after_cnn, \
                padded_mask_after_cnn.sum().unsqueeze(0), attention_mask

    def forward(
        self,
        input_features=None,
        aftercnn_lens=None,
        padded_mask=None,
        padded_mask_after_cnn=None,
        attention_mask=None,
    ):
        padded_embed = nn.functional.gelu(
            self.conv1(input_features)) * padded_mask
        padded_embed = nn.functional.gelu(\
            self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + \
          self.positional_embedding.positional_embedding[
                                    : padded_embed.shape[1],
                                    :
                                    ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]

        for _, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(hidden_states, None, attention_mask)

            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.permute(
                1, 0)).permute(1, 0)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        return token_audio

    def post_attn(
        self,
        padded_token_audio,
        padded_aftercnn_lens,
        aftercnn_lens,
    ):
        # avgpool.stride is 2
        padded_aftercnn_lens = padded_aftercnn_lens // 2
        aftercnn_lens = aftercnn_lens // 2
        padded_token_audio_list = padded_token_audio.split(
            padded_aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for i, each_audio_states in enumerate(padded_token_audio_list):
            token_audio_list.append(each_audio_states[:aftercnn_lens[i], :])
        token_audio = torch.cat(token_audio_list, dim=0)

        return token_audio

    def padded_and_mask_function(self,
                                 tensor_list,
                                 tensor_len,
                                 padding_value=0,
                                 audio_buckets=None):
        padded_len = audio_buckets.get_multimodal_bucket(tensor_len.sum())
        padded_chunk_num = math.ceil(padded_len / (self.n_window * 2))
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(padded_chunk_num, dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (padded_chunk_num, max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (padded_chunk_num, max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    def get_audio_embeddings(
        self,
        input_features,
        audio_feature_lengths,
        audio_feat_lengths,
        audio_buckets,
    ):
        offset = 0
        audio_outputs = []
        for feature_len in audio_feature_lengths:
            input_feature = input_features[offset:offset + feature_len]
            offset = offset + feature_len
            padded_feature, padded_mask, padded_mask_after_cnn, \
              padded_aftercnn_lens, attention_mask = \
                  self.pre_attn(input_feature,
                                feature_len.unsqueeze(0),
                                audio_buckets)
            audio_features = self.forward(padded_feature, padded_aftercnn_lens,
                                          padded_mask, padded_mask_after_cnn,
                                          attention_mask)
            audio_features = self.post_attn(audio_features,
                                            padded_aftercnn_lens,
                                            audio_feat_lengths)
            audio_outputs.append(audio_features)
        return torch.cat(audio_outputs, dim=0)


class Qwen2_5OmniThinkerProcessingInfo(Qwen2AudioProcessingInfo,
                                       Qwen2_5_VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2_5OmniConfig).thinker_config

    def get_hf_processor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, List[float]]] = None,
        **kwargs: object,
    ) -> Qwen2_5OmniProcessor:
        if fps is not None:
            kwargs["fps"] = fps
        processor = self.ctx.get_hf_processor(
            Qwen2_5OmniProcessor,
            image_processor=self.get_image_processor(min_pixels=min_pixels,
                                                     max_pixels=max_pixels,
                                                     size=size),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|AUDIO|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|IMAGE|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|VIDEO|>"
        return processor

    def get_feature_extractor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None, "image": None, "video": None}


class Qwen2_5OmniThinkerDummyInputsBuilder(
        BaseDummyInputsBuilder[Qwen2_5OmniThinkerProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()

        audio_token: str = hf_processor.audio_token
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        return (audio_token * num_audios + image_token * num_images +
                video_token * num_videos)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        feature_extractor = self.info.get_feature_extractor()

        target_audio_length = min(
            feature_extractor.chunk_length,
            30,
        ) * feature_extractor.sampling_rate
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        mm_data = {
            "audio":
            self._get_dummy_audios(length=target_audio_length,
                                   num_audios=num_audios),
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(width=target_width,
                                   height=target_height,
                                   num_frames=target_num_frames,
                                   num_videos=num_videos),
        }

        return mm_data


class Qwen2_5OmniThinkerMultiModalProcessor(
        BaseMultiModalProcessor[Qwen2_5OmniThinkerProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return Qwen2_5OmniThinkerMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        if audios:
            # NOTE: Qwen2.5-Omni processor accept "audio"
            mm_data["audio"] = audios
            mm_kwargs = dict(**mm_kwargs, )

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        input_features = hf_inputs.pop('input_features', None)
        feature_attention_mask = hf_inputs.get('feature_attention_mask', None)
        if ('input_audio_features' not in hf_inputs
                and input_features is not None):
            if feature_attention_mask is not None:
                input_features = input_features.permute(
                    0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
            hf_inputs['input_audio_features'] = input_features
        if ('audio_feature_lengths' not in hf_inputs
                and feature_attention_mask is not None):
            hf_inputs['audio_feature_lengths'] = feature_attention_mask.sum(-1)
        return hf_inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _qwen2_5_omni_thinker_field_config(hf_inputs)

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargs,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        Qwen2.5-Omni reimplements this function to handle `use_audio_in_video`.
        """
        unbound_prompt_updates = self._get_prompt_updates(
            mm_items,
            hf_processor_mm_kwargs,
            mm_kwargs,
        )
        mm_prompt_updates = self._bind_and_group_updates(
            unbound_prompt_updates)

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = hf_processor_mm_kwargs.get(
            "use_audio_in_video", False)

        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                mm_prompt_updates,
                prompt_ids,
                mm_item_counts,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
                use_audio_in_video=use_audio_in_video)

            tokenizer = self.info.get_tokenizer()
            prompt = decode_tokens(tokenizer, prompt_ids)
        else:
            (
                prompt_ids,
                prompt,
                mm_placeholders,
            ) = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
                mm_item_counts,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
                use_audio_in_video=use_audio_in_video)

        tokenizer = self.info.get_tokenizer()
        prompt = decode_tokens(tokenizer, prompt_ids)

        if use_audio_in_video:
            mm_kwargs["use_audio_in_video"] = True

        return prompt_ids, prompt, mm_placeholders

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        audio_feature_lengths = out_mm_kwargs.get("audio_feature_lengths")
        feature_attention_mask = out_mm_kwargs.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = _get_feat_extract_output_lengths(
                audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1))
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            item_idx += audio_in_video_item_idx

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model")

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        use_audio_in_video = hf_processor_mm_kwargs.get(
            "use_audio_in_video", False)
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx

            audio_num_features = audio_output_lengths[audio_in_video_item_idx +
                                                      item_idx]
            video_grid_thw = out_mm_kwargs["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = hf_processor_mm_kwargs.get(
                "second_per_grid_ts", None)
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[item_idx]
            else:
                video_second_per_grid_t = 1.0

            return MRotaryEmbedding.omni_get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )

        video_replacement_fn = (
            get_replacement_qwen2_use_audio_in_video if use_audio_in_video else
            partial(get_replacement_qwen2_vision, modality="video"))

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision,
                                    modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _apply_hf_processor_main(
        self,
        prompt: Union[str, list[int]],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        """
        Qwen2.5-Omni reimplements this function to handle text only.
        """
        if isinstance(prompt, str):
            if enable_hf_prompt_update:
                return self._apply_hf_processor_text_mm(
                    prompt_text=prompt,
                    mm_items=mm_items,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                )
            tokenizer = self.info.get_tokenizer()
            prompt_ids = encode_tokens(tokenizer, prompt)
        else:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt)

        mm_kwargs = self._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return prompt_ids, mm_kwargs, False

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalKwargs:
        """
        Qwen2.5-Omni reimplements this function to handle `use_audio_in_video`.
        """
        mm_counts = mm_items.get_all_counts()

        use_audio_in_video = hf_processor_mm_kwargs.get(
            "use_audio_in_video", False)
        if use_audio_in_video and "video" in mm_counts:
            assert "audio" in mm_counts
            mm_counts["audio"] -= mm_counts["video"]

        _, mm_kwargs, _ = self._apply_hf_processor_text_mm(
            prompt_text=self.dummy_inputs.get_dummy_text(mm_counts),
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return mm_kwargs

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
        use_audio_in_video: bool = False,
    ) -> None:
        if use_audio_in_video:
            mm_item_counts = copy(mm_item_counts)
            if "video" in mm_item_counts:
                assert "audio" in mm_item_counts
                mm_item_counts["audio"] -= mm_item_counts["video"]
        super()._validate_mm_placeholders(mm_placeholders, mm_item_counts)


class Qwen2_5OmniConditionalGenerationMixin:

    def _validate_and_reshape_mm_tensor(self,
                                        mm_input: object,
                                        name: str,
                                        dim: int = 0) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input), dim=dim)
        else:
            return torch.concat(mm_input, dim=dim)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[Qwen2AudioInputs]:
        input_audio_features = kwargs.pop('input_audio_features', None)
        audio_feature_lengths = kwargs.pop('audio_feature_lengths', None)
        feature_attention_mask = kwargs.pop('feature_attention_mask', None)
        if input_audio_features is None:
            return None
        input_audio_features = self._validate_and_reshape_mm_tensor(
            input_audio_features, 'input_audio_features', dim=1)
        if feature_attention_mask is not None:
            feature_attention_mask = self._validate_and_reshape_mm_tensor(
                feature_attention_mask, 'feature_attention_mask')
        if not isinstance(input_audio_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of audio input features. "
                             f"Got type: {type(input_audio_features)}")
        return Qwen2AudioInputs(input_features=input_audio_features,
                                audio_feature_lengths=audio_feature_lengths,
                                feature_attention_mask=feature_attention_mask)

    def _parse_and_validate_image_input(
        self,
        **kwargs: Dict[str, Any],
    ) -> Optional[Qwen2_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
        self,
        **kwargs: Dict[str, Any],
    ) -> Optional[Qwen2_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw)

    def _process_audio_input(
        self,
        audio_input: Qwen2AudioInputs,
        audio_hashes: List[str] = None,
        cached_audio_features: torch.Tensor = None,
    ) -> torch.Tensor:

        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]
        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)
        if audio_feature_lengths.ndim == 2:
            assert audio_feature_lengths.shape[
                0] == 1 or audio_feature_lengths.shape[1] == 1
            if audio_feature_lengths.shape[0] == 1:
                audio_feature_lengths = audio_feature_lengths.squeeze(0)
            else:
                audio_feature_lengths = audio_feature_lengths.squeeze(1)

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                audio_feature_lengths))

        if is_hpu:
            assert isinstance(self.audio_tower,
                              Qwen2_5OmniAudioEncoderStaticShape)
            audio_outputs = self.audio_tower.get_audio_embeddings(
                input_features.to(self.audio_tower.dtype),
                audio_feature_lengths=audio_feature_lengths,
                audio_feat_lengths=audio_feat_lengths,
                audio_buckets=self.audio_buckets)
        else:
            audio_outputs = self.audio_tower(
                input_features.to(self.audio_tower.dtype),
                feature_lens=audio_feature_lengths,
                aftercnn_lens=audio_feat_lengths,
            )
        return audio_outputs.split(audio_output_lengths.tolist())

    def _process_image_input(
            self,
            image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        grid_thw = image_input["image_grid_thw"]
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        assert grid_thw.ndim == 2

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        if is_hpu:
            assert isinstance(self.visual,
                              Qwen2_5_VisionTransformerStaticShape)
            image_embeds = self.visual.get_image_embeds(
                pixel_values,
                grid_thw=grid_thw,
                vision_buckets=self.vision_buckets,
            )
        else:
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs,
            video_hashes: List[str] = None,
            cached_video_embeds: torch.Tensor = None) -> torch.Tensor:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype)
        if is_hpu:
            assert isinstance(self.visual,
                              Qwen2_5_VisionTransformerStaticShape)
            video_embeds = self.visual.get_image_embeds(
                pixel_values_videos,
                grid_thw=grid_thw,
                vision_buckets=self.vision_buckets,
            )
        else:
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniThinkerForConditionalGeneration(
        nn.Module, SupportsMultiModal, SupportsPP,
        Qwen2_5OmniConditionalGenerationMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        thinker_config: Qwen2_5OmniThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config)
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        # force "use_flash_attention_2=True" to audio tower to align
        # the results.
        if flash_attn is not None:
            audio_config = thinker_config.audio_config
            audio_config._attn_implementation_autoset = True
            audio_config._attn_implementation = "flash_attention_2"
        else:
            logger.warning(
                "flash_attn is not available, the model may not yield the "
                "exactly same result as the transformers implementation "
                "in the audio tower part.")

        if is_hpu:
            Qwen2_5OmniAudioEncoder = Qwen2_5OmniAudioEncoderStaticShape
            qwen2_5_visionTransformer = Qwen2_5_VisionTransformerStaticShape
        else:
            qwen2_5_visionTransformer = Qwen2_5_VisionTransformer
        self.audio_tower = Qwen2_5OmniAudioEncoder(thinker_config.audio_config)
        self.visual = qwen2_5_visionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.quant_config = quant_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=thinker_config.text_config,
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds"
                             ) and "video" not in mm_input_by_modality:
                mm_input_by_modality[
                    "video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key in ("input_audio_features"
                             ) and "audio" not in mm_input_by_modality:
                mm_input_by_modality[
                    "audio"] = self._parse_and_validate_audio_input(**kwargs)
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += audio_embeddings
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:

            # TODO (ywang96): support overlapping modalitiy embeddings so that
            # `use_audio_in_video` will work on V1.
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, [
                    self.config.image_token_index,
                    self.config.video_token_index,
                    self.config.audio_token_index
                ])
        return inputs_embeds

    def get_multimodal_embeddings_v0(
            self, **kwargs: object) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)

        if audio_input is None and image_input is None and video_input is None:
            return None

        multimodal_embeddings: List[Tuple[NestedTensors, str]] = []

        if audio_input is not None:
            audio_embeds = self._process_audio_input(audio_input)
            multimodal_embeddings.append((audio_embeds, "audio"))
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            multimodal_embeddings.append((image_embeds, "image"))
        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            multimodal_embeddings.append((video_embeds, "video"))
        return multimodal_embeddings

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is None:
            return inputs_embeds
        for embeddings, modality in multimodal_embeddings:
            if modality == "audio":
                placeholder_token_id = self.config.audio_token_index
            if modality == "image":
                placeholder_token_id = self.config.image_token_index
            if modality == "video":
                placeholder_token_id = self.config.video_token_index
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, embeddings, placeholder_token_id)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings_v0(**kwargs)
            inputs_embeds = self.get_input_embeddings_v0(
                input_ids, multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "token2wav."],
        )
        loaded_weights = loader.load_weights(weights,
                                             mapper=self.hf_to_vllm_mapper)

        return loaded_weights
