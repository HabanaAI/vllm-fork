# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Optional, TypedDict

import torch
from torch import nn
from transformers import BatchFeature, Gemma3Config, Gemma3Processor
from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
# yapf: disable
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, BoundPromptUpdate,
                                        PlaceholderFeaturesInfo,
                                        PromptReplacement, PromptTargetMatch,
                                        PromptUpdate, PromptUpdateDetails,
                                        find_mm_placeholders,
                                        replace_token_matches)
# yapf: enable
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, flatten_bn, greedy_plan,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)
is_hpu = current_platform.is_hpu()
is_lazy = os.environ.get('PT_HPU_LAZY_MODE', '0') == '1' if is_hpu else False

is_hpu = current_platform.is_hpu()


class Gemma3ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """
    Shape: `(num_patches_total, num_channels, height, width)`

    `num_patches_total` is the total number of patches
    over each image over each prompt in the batch.
    """

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""


Gemma3ImageInputs = Gemma3ImagePixelInputs


class Gemma3ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Gemma3Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(Gemma3Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def _resolve_image_kwargs(
        self,
        processor: Gemma3Processor,
        keys: set[str],
    ) -> dict[str, Any]:
        image_processor = processor.image_processor
        kwargs = processor._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        )

        images_kwargs = kwargs["images_kwargs"]

        def _resolve_kw(key: str):
            val = getattr(image_processor, key)
            if val is None:
                val = images_kwargs[key]

            return val

        return {k: _resolve_kw(k) for k in keys}

    def get_num_crops(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Gemma3Processor],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        images_kwargs = self._resolve_image_kwargs(
            processor, {
                "do_pan_and_scan", "pan_and_scan_min_crop_size",
                "pan_and_scan_max_num_crops",
                "pan_and_scan_min_ratio_to_activate"
            })

        do_pan_and_scan = images_kwargs["do_pan_and_scan"]
        pan_and_scan_min_crop_size = images_kwargs[
            "pan_and_scan_min_crop_size"]
        pan_and_scan_max_num_crops = images_kwargs[
            "pan_and_scan_max_num_crops"]
        pan_and_scan_min_ratio_to_activate = images_kwargs[
            "pan_and_scan_min_ratio_to_activate"]

        if not do_pan_and_scan:
            return 0

        if envs.VLLM_USE_V1:
            logger.warning_once(
                "`do_pan_and_scan=True` has suboptimal results on V1 "
                "because of the simplified attention pattern being used.")

        # Based on Gemma3ImageProcessor.pan_and_scan
        if image_width >= image_height:
            if image_width / image_height < pan_and_scan_min_ratio_to_activate:
                return 0

            num_crops_w = min(
                int(math.floor(image_width / pan_and_scan_min_crop_size)),
                int(math.floor(image_width / image_height + 0.5)),
            )

            num_crops_w = max(2, num_crops_w)
            num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
            num_crops_h = 1
        else:
            if image_height / image_width < pan_and_scan_min_ratio_to_activate:
                return 0

            num_crops_h = min(
                int(math.floor(image_height / pan_and_scan_min_crop_size)),
                int(math.floor(image_height / image_width + 0.5)),
            )

            num_crops_h = max(2, num_crops_h)
            num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
            num_crops_w = 1

        crop_size_w = int(math.ceil(image_width / num_crops_w))
        crop_size_h = int(math.ceil(image_height / num_crops_h))

        if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
            return 0

        return num_crops_w * num_crops_h

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Gemma3Processor],
    ) -> PromptUpdateDetails[str]:
        if processor is None:
            processor = self.get_hf_processor()

        boi_token = processor.boi_token

        num_crops = self.get_num_crops(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )

        if num_crops == 0:
            image_text = boi_token
        else:
            crops_image_tokens = " ".join(boi_token for _ in range(num_crops))
            image_text = (
                f"Here is the original image {boi_token} and here are some "
                f"crops to help you see better {crops_image_tokens}")

        repl_full = image_text.replace(boi_token,
                                       processor.full_image_sequence)

        tokenizer = processor.tokenizer
        vocab = tokenizer.get_vocab()
        image_token_id = vocab[tokenizer.image_token]

        return PromptUpdateDetails.select_token_id(repl_full, image_token_id)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Gemma3Processor],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        num_crops = self.get_num_crops(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )
        image_seq_len = processor.image_seq_length

        return (num_crops + 1) * image_seq_len

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        images_kwargs = self._resolve_image_kwargs(
            processor, {"pan_and_scan_max_num_crops"})
        max_num_crops = images_kwargs["pan_and_scan_max_num_crops"]

        # Result in the max possible feature size (h:w = max_num_crops:1)
        return ImageSize(height=50 * max_num_crops, width=50)


class Gemma3DummyInputsBuilder(BaseDummyInputsBuilder[Gemma3ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.boi_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class Gemma3MultiModalProcessor(BaseMultiModalProcessor[Gemma3ProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
        )

        # HF processor pops the `num_crops` kwarg, which is needed by vLLM
        if (images := mm_data.get("images")) is not None:
            parsed_images = (self._get_data_parser().parse_mm_data({
                "image":
                images
            }).get_items("image", ImageProcessorItems))
            image_sizes = [
                parsed_images.get_image_size(i)
                for i in range(len(parsed_images))
            ]
            hf_processor = self.info.get_hf_processor(**mm_kwargs)

            num_crops = [
                self.info.get_num_crops(image_width=size.width,
                                        image_height=size.height,
                                        processor=hf_processor)
                for size in image_sizes
            ]
            processed_outputs["num_crops"] = torch.tensor(num_crops)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_crops = hf_inputs.get("num_crops", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_crops + 1),
            num_crops=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.boi_token

        def get_replacement_gemma3(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)

            image_size = images.get_image_size(item_idx)
            return self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_gemma3,
            )
        ]

    def _apply_token_matches(
        self,
        prompt: list[int],
        mm_matches: Mapping[str, Sequence[PromptTargetMatch]],
        mm_item_counts: Mapping[str, int],
    ) -> list[int]:
        token_ids = super()._apply_token_matches(
            prompt,
            mm_matches,
            mm_item_counts,
        )

        # "\n\n\n" and "\n\n\n\n" are single tokens
        # Since our replacement can insert "\n\n" next to "\n"
        # tokens, we have to combine them to be consistent with
        # the output of the tokenizer
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        newline_1 = vocab["\n"]
        newline_2 = vocab["\n\n"]
        newline_3 = vocab["\n\n\n"]
        newline_4 = vocab["\n\n\n\n"]

        token_ids = replace_token_matches(
            token_ids,
            [newline_1, newline_2],
            [newline_3],
        )
        token_ids = replace_token_matches(
            token_ids,
            [newline_2, newline_1],
            [newline_3],
        )
        token_ids = replace_token_matches(
            token_ids,
            [newline_2, newline_2],
            [newline_4],
        )

        return token_ids

    def _find_mm_placeholders(
        self,
        mm_prompt_updates: Mapping[str, Sequence[BoundPromptUpdate]],
        new_token_ids: list[int],
        mm_item_counts: Mapping[str, int],
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        # We need to detect "\n\n" inside "\n\n\n" and "\n\n\n\n"
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        newline_1 = vocab["\n"]
        newline_2 = vocab["\n\n"]
        newline_3 = vocab["\n\n\n"]
        newline_4 = vocab["\n\n\n\n"]

        def get_repl_toks(tok: int) -> list[int]:
            if tok == newline_3:
                return [newline_1, newline_2]
            if tok == newline_4:
                return [newline_2, newline_2]

            return [tok]

        repl_token_ids = list[int]()
        repl_orig_idxs = list[int]()
        for orig_idx, orig_tok in enumerate(new_token_ids):
            repl_toks = get_repl_toks(orig_tok)
            repl_token_ids.extend(repl_toks)
            repl_orig_idxs.extend(orig_idx for _ in range(len(repl_toks)))

        repls = find_mm_placeholders(mm_prompt_updates, repl_token_ids,
                                     mm_item_counts)

        return {
            modality: [
                PlaceholderFeaturesInfo(
                    modality=p.modality,
                    item_idx=p.item_idx,
                    start_idx=repl_orig_idxs[p.start_idx],
                    tokens=p.tokens,
                    is_embed=p.is_embed,
                ) for p in placeholders
            ]
            for modality, placeholders in repls.items()
        }


class Gemma3MultiModalProjector(nn.Module):

    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size,
                        config.text_config.hidden_size))

        self.mm_soft_emb_norm = GemmaRMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps)

        self.patches_per_image = int(config.vision_config.image_size //
                                     config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size,
                                     stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image,
            self.patches_per_image)
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)


@MULTIMODAL_REGISTRY.register_processor(Gemma3MultiModalProcessor,
                                        info=Gemma3ProcessingInfo,
                                        dummy_inputs=Gemma3DummyInputsBuilder)
class Gemma3ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP,
                                     SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.sliding_window = getattr(config.text_config,
                                      "interleaved_sliding_window", None)

        self.vision_tower = SiglipVisionModel(config.vision_config,
                                              quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_tower"))
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Gemma3ForCausalLM"],
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.language_model.logits_processor.scale *= logit_scale

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)
        if is_hpu:
            self.graphed_multimodal_buckets = None

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        image_size = self.config.vision_config.image_size
        expected_dims = (3, image_size, image_size)
        if data.shape[1:] != expected_dims:
            raise ValueError(
                "The expected shape of pixel values per image per batch is "
                f"{expected_dims}. You supplied {tuple(data.shape)}.")
        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Gemma3ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        num_crops = kwargs.pop("num_crops", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Gemma3 does not support image_embeds."
        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        if not isinstance(num_crops, (torch.Tensor, list)):
            raise ValueError("Incorrect type of num_crops. "
                             f"Got type: {type(num_crops)}")

        pixel_values = flatten_bn(pixel_values, concat=True)
        num_crops = flatten_bn(num_crops, concat=True)

        return Gemma3ImagePixelInputs(
            type="pixel_values",
            pixel_values=self._validate_pixel_values(pixel_values),
            # TODO.. some bug in adding 1 here to num_crops..
            # currently assuming no panscan so just passing in torch.ones
            # seeing 0 + 1 = 0 here sometimes!! hence wrapping in torch.tensor
            num_patches=num_crops + 1 if not is_hpu else \
                torch.ones(num_crops.shape, dtype=num_crops.dtype).to(
                    pixel_values.device))

    def _image_pixels_to_features(
        self,
        vision_tower: SiglipVisionModel,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        return vision_tower(pixel_values)

    def _process_image_input(
            self, image_input: Gemma3ImageInputs) -> list[torch.Tensor]:
        assert self.vision_tower is not None

        pixel_values = image_input["pixel_values"]
        num_patches = image_input["num_patches"]

        image_features = self._image_pixels_to_features(
            self.vision_tower,
            pixel_values,
        )

        if is_hpu and len(self.graphed_multimodal_buckets) > 1:
            batch_breakdown = greedy_plan(pixel_values.shape[0], \
                    self.vision_buckets.multimodal_buckets)
            start_idx = 0
            image_embeds_multibatches = []

            for i in batch_breakdown:
                end_idx = start_idx + i
                batch_sliced_image_features = \
                        image_features[start_idx:end_idx, ...]
                if is_lazy:
                    image_embeds_multibatches += \
                            [self.multi_modal_projector(
                                batch_sliced_image_features,
                                bypass_hpu_graphs=i
                                not in self.graphed_multimodal_buckets)]
                else:
                    image_embeds_multibatches += \
                            [self.multi_modal_projector( \
                                batch_sliced_image_features)]
                start_idx = end_idx
            image_embeds = torch.cat(image_embeds_multibatches, dim=0)
        else:
            image_embeds = self.multi_modal_projector(image_features)
        return [
            e.flatten(0, 1) for e in image_embeds.split(num_patches.tolist())
        ]

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        if is_hpu:
            self.graphed_multimodal_buckets = kwargs.pop(
                'graphed_multimodal_buckets', [])
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_index,
            )
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            if is_hpu:
                raise AssertionError("hpu_model_runner should be computing \
                        inputs_embeds")
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)

            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            if vision_embeddings is not None:
                kwargs = self.prepare_attn_masks(
                    input_ids,
                    positions,
                    mask_dtype=self.dtype,
                    **kwargs,
                )
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds,
                                                  **kwargs)

        return hidden_states

    def hpu_build_mask(self, input_ids: torch.Tensor,
                       mask_dtype: torch.dtype) -> torch.Tensor:
        bs, seq_len = input_ids.shape
        device = input_ids.device
        img_tokens = self.config.mm_tokens_per_image
        image_token_index = self.config.image_token_index
        # bool causal mask (True == masked)
        causal_bool = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), 1)
        mask_bool = causal_bool.unsqueeze(0).unsqueeze(0).expand(
            bs, 1, -1, -1).clone()

        # pre-compute a few broadcastable helpers
        img_pos = (input_ids == image_token_index)  # [B,S]
        img_row = img_pos.unsqueeze(1).unsqueeze(3)  # [B,1,S,1]
        img_col = img_pos.unsqueeze(1).unsqueeze(2)  # [B,1,1,S]

        img_pos_cum = torch.cumsum(img_pos, 1)
        img_causal = torch.arange(seq_len, device=device).unsqueeze(0) \
            - img_pos_cum + (img_pos_cum // img_tokens + 1) * img_tokens + 1
        img_causal = torch.cat((img_causal[:, :1] - 1, img_causal[:, :-1]), 1) \
            .clamp_(0, seq_len - 1) \
            .unsqueeze(1).unsqueeze(3)                          # [B,1,S,1]
        ind = torch.arange(seq_len, device=device).view(1, 1, 1,
                                                        -1)  # [1,1,1,S]

        # positions we must *unmask*  (row img  ∧  col img
        # ∧  col < img_causal)
        allow = img_row & img_col & (ind < img_causal)
        mask_bool &= ~allow  # flip to False

        # 4)   final bfp16/32 version
        out = torch.zeros_like(mask_bool, dtype=mask_dtype) \
            .masked_fill(mask_bool, float("-inf"))

        return out

    def prepare_attn_masks(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask_dtype: torch.dtype,
        **kwargs,
    ):
        kwargs["has_images"] = True
        seq_lens = []
        if is_hpu:
            seq_len = input_ids.shape[1]
            bs = input_ids.shape[0]
            kwargs["seq_lens"] = [seq_len] * bs
            seq_lens.append(seq_len)
        else:
            # NOTE(woosuk): Here, we distinguish the sequences
            # by the position id 0.
            # This is a HACK. Fix this.
            start_idices = (positions == 0).cpu().nonzero()
            num_seqs = len(start_idices)
            for i in range(num_seqs):
                start_idx = start_idices[i].item()
                if i < num_seqs - 1:
                    end_idx = start_idices[i + 1].item()
                else:
                    end_idx = len(input_ids)
                seq_lens.append(end_idx - start_idx)
            kwargs["seq_lens"] = seq_lens

        global_attn_masks = []
        local_attn_masks = []
        start_idx = 0
        for seq_len in seq_lens:
            if is_hpu:
                global_attn_mask = self.hpu_build_mask(input_ids, mask_dtype)
            else:
                end_idx = start_idx + seq_len
                input_token_ids = input_ids[start_idx:end_idx]
                start_idx = end_idx
                bs = 1
                # Create a global causal mask.
                global_attn_mask = torch.empty(
                    bs,
                    1,
                    seq_len,
                    seq_len,
                    dtype=mask_dtype,
                    device=input_ids.device,
                )
                global_attn_mask.fill_(float("-inf"))
                # Fill the lower triangle with 0.
                global_attn_mask = global_attn_mask.triu(diagonal=1)

                # Consider the bidirectional attention between image tokens.
                img_mask = torch.zeros_like(global_attn_mask)
                img_pos = (input_token_ids == self.config.image_token_index)

                img_mask[:, :, :, img_pos] += 1
                img_mask[:, :, img_pos, :] += 1
                global_attn_mask = torch.where(img_mask == 2, 0,
                                               global_attn_mask)

            global_attn_masks.append(global_attn_mask)

            if self.sliding_window is not None:
                if is_hpu and kwargs['attn_metadata'].use_window_sdpa:
                    # In HPU, no need to create local attn_mask(save memory)
                    # if slice_sdpa kernel is used for this input.
                    local_attn_masks = None
                else:
                    # Create a local causal mask with sliding window (1024).
                    local_attn_mask = torch.ones_like(global_attn_mask)
                    local_attn_mask = torch.tril(local_attn_mask,
                                                 diagonal=-self.sliding_window)
                    local_attn_mask = torch.where(local_attn_mask == 0,
                                                  global_attn_mask,
                                                  float("-inf"))
                    local_attn_masks.append(local_attn_mask)
        kwargs["global_attn_masks"] = global_attn_masks
        kwargs["local_attn_masks"] = local_attn_masks
        return kwargs

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower")
