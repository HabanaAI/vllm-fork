import torch
from torch import nn
from transformers import RobertaConfig
from typing import Optional

from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)

from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("roberta_embedding")
class RobertaEmbedding(CustomOp):
    """Original roberta embedding."""

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size,
                                                padding_idx=self.padding_idx)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.position_ids = nn.Parameter(
            torch.empty((1, config.max_position_embeddings)), )

        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type != "absolute":
            raise ValueError("Only 'absolute' position_embedding_type" +
                             " is supported")

    def forward_hpu(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        inputs_embeds = self.word_embeddings(input_ids)

        # Replace position ids because in RoBERTa models
        # they have to start at padding_idx + 1 and ignore
        # existing padding tokens
        # Modified replace position ids as HPU set position_ids and input_ids as [batch_size, bucket_size]
        # References:
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L133
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L1669
        pos_list = []
        token_list = []
        for offset in range(position_ids.size()[0]):
            pos_list.append(position_ids[offset])
            token_list.append(input_ids[offset])

        for index, (positions, tokens,
                    seq_len) in enumerate(zip(pos_list, token_list, seq_lens)):
            # Verify assumption that incoming position are
            # always a sequence from 0 to N.
            expected_pos = torch.arange(positions.size()[0],
                                        dtype=torch.long,
                                        device=inputs_embeds.device)
            valid_input_mask = expected_pos < seq_len
            expected_pos = expected_pos * valid_input_mask
            assert torch.equal(positions, expected_pos)
            position_ids[index] = create_position_ids_from_input_ids(
                tokens, self.padding_idx, seq_len)

        # Position embeddings.
        position_embeddings = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=inputs_embeds.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward_cuda(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        inputs_embeds = self.word_embeddings(input_ids)

        # Replace position ids because in RoBERTa models
        # they have to start at padding_idx + 1 and ignore
        # existing padding tokens
        # References:
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L133
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L1669
        pos_list = []
        token_list = []
        offset = 0
        for seq_len in seq_lens:
            pos_list.append(position_ids[offset:offset + seq_len])
            token_list.append(input_ids[offset:offset + seq_len])
            offset += seq_len

        new_pos_list = []
        for positions, tokens in zip(pos_list, token_list):
            # Verify assumption that incoming position are
            # always a sequence from 0 to N.
            expected_pos = torch.arange(positions.size()[0],
                                        dtype=torch.long,
                                        device=inputs_embeds.device)
            assert torch.equal(positions, expected_pos)
            new_pos_list.append(
                create_position_ids_from_input_ids(tokens, self.padding_idx))
        position_ids = torch.cat(new_pos_list)

        # Position embeddings.
        position_embeddings = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=inputs_embeds.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


# Adapted from transformers
def create_position_ids_from_input_ids(input_ids,
                                       padding_idx,
                                       seq_len=None,
                                       past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA.
    if seq_len is not None:
        valid_input_mask = torch.arange(input_ids.size()[0],
                                        dtype=torch.int,
                                        device=input_ids.device)
        valid_input_mask = valid_input_mask < seq_len

    mask = input_ids.ne(padding_idx).int()

    incremental_indices = (torch.cumsum(mask, dim=0).type_as(mask) +
                           past_key_values_length) * mask

    if seq_len is not None:
        return (incremental_indices.long() + padding_idx) * valid_input_mask
    else:
        return incremental_indices.long() + padding_idx
