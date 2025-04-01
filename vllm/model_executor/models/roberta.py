# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import RobertaConfig

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.embedding import RobertaEmbedding
from vllm.model_executor.layers.pooler import CrossEncodingPooler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bert import BertEmbeddingModel, BertModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.transformers_utils.config import (
    get_cross_encoder_activation_function)

from .interfaces import SupportsCrossEncoding


def roberta_task_weights_filter(
    all_weights: Iterable[Tuple[str, torch.Tensor]]
) -> Tuple[Iterable[Tuple[str, torch.Tensor]], Iterable[Tuple[str,
                                                              torch.Tensor]]]:
    """
    Separate task-specific weights that are applied on top
    of the encoder-decoder bert base.
    To do so, return two generators over the original iterator.
    Also, remove the "roberta." prefix to make it loadable
    from vanilla BertModel.
    """
    # Copy of a lazy iterator without in-memory overhead so both
    # iterators can be iterated upon independently.
    all_weights1, all_weights2 = itertools.tee(all_weights)

    def encoder_decoder_weights():
        for name, weight in all_weights1:
            if name.startswith("roberta."):
                yield (name[len("roberta."):], weight)

    return encoder_decoder_weights(), ((n, w) for n, w in all_weights2
                                       if not n.startswith("roberta."))


# Adapted from transformers
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class RobertaEmbeddingModel(BertEmbeddingModel):
    """A model that uses Roberta to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def _build_model(self,
                     vllm_config: VllmConfig,
                     prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config,
                         prefix=prefix,
                         embedding_class=RobertaEmbedding)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        # Separate weights in "roberta"-prefixed and all else (not in memory).
        # For use with models like FacebookAI/roberta-base.
        bert_weights, task_weights = roberta_task_weights_filter(weights)
        loaded = self.model.load_weights(bert_weights)
        if not len(loaded):
            # Fix for models like `sentence-transformers/stsb-roberta-base-v2`
            # which use the same architecture, but have no "roberta" prefix.
            loaded = self.model.load_weights(task_weights)
        assert len(loaded), "Unable to load RobertaEmbeddingModel"


class RobertaForSequenceClassification(nn.Module, SupportsCrossEncoding):
    """A model that uses Roberta to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       roberta: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.default_activation_function = \
            get_cross_encoder_activation_function(config)

        self.num_labels = config.num_labels
        self.roberta = BertModel(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "bert"),
                                 embedding_class=RobertaEmbedding,
                                 add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self._pooler = CrossEncodingPooler(config, self.classifier)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        bert_weights, task_weights = roberta_task_weights_filter(weights)
        self.roberta.load_weights(bert_weights)

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in task_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.roberta(input_ids=input_ids,
                            position_ids=positions,
                            kv_caches=kv_caches,
                            inputs_embeds=inputs_embeds,
                            intermediate_tensors=intermediate_tensors,
                            attn_metadata=attn_metadata,
                            token_type_ids=token_type_ids)
