# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from typing import Any, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig

from transformers.processing_utils import ProcessorMixin

class HashableDict(dict):
    """
    A dictionary that can be hashed by lru_cache.
    """

    # NOTE: pythonic dict is not hashable,
    # we override on it directly for simplicity
    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))


class HashableList(list):
    """
    A list that can be hashed by lru_cache.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))


def _merge_mm_kwargs(model_config: "ModelConfig", **kwargs):
    base_kwargs = model_config.mm_processor_kwargs
    if base_kwargs is None:
        base_kwargs = {}

    merged_kwargs = {**base_kwargs, **kwargs}

    # NOTE: Pythonic dict is not hashable and will raise unhashable type
    # error when calling `cached_get_processor`, therefore we need to
    # wrap it to a hashable dict.
    for key, value in merged_kwargs.items():
        if isinstance(value, dict):
            merged_kwargs[key] = HashableDict(value)
        if isinstance(value, list):
            merged_kwargs[key] = HashableList(value)
    return merged_kwargs

def get_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    processor_cls: type[ProcessorMixin] = ProcessorMixin,
    **kwargs: Any,
):
    """Load a processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor

    processor_factory = (AutoProcessor
                         if processor_cls == ProcessorMixin else processor_cls)

    try:
        processor = processor_factory.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return cast(ProcessorMixin, processor)


cached_get_processor = lru_cache(get_processor)


def get_feature_extractor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load an audio feature extractor for the given model name 
    via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoFeatureExtractor
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the feature extractor. If the feature "
                "extractor is a custom extractor not yet available in the "
                "HuggingFace transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    return cast(FeatureExtractionMixin, feature_extractor)


cached_get_feature_extractor = lru_cache(get_feature_extractor)


def cached_feature_extractor_from_config(
    model_config: "ModelConfig",
    **kwargs: Any,
):
    return cached_get_feature_extractor(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
        **_merge_mm_kwargs(model_config, **kwargs),
    )


def get_image_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load an image processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoImageProcessor
    from transformers.image_processing_utils import BaseImageProcessor

    try:
        processor = AutoImageProcessor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the image processor. If the image processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return cast(BaseImageProcessor, processor)


def get_video_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load a video processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers.image_processing_utils import BaseImageProcessor

    processor = get_processor(
        processor_name,
        *args,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )

    return cast(BaseImageProcessor, processor.video_processor)
