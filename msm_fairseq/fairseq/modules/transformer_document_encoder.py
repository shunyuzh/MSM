# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
)
import random
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
np.set_printoptions(threshold=np.inf)

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class TransformerDocumentEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
            self,
            padding_idx: int,
            vocab_size: int,
            num_encoder_layers: int = 6,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop : float = 0.0,
            max_seq_len: int = 256,
            num_segments: int = 2,
            use_position_embeddings: bool = True,
            offset_positions_by_padding: bool = True,
            encoder_normalize_before: bool = False,
            apply_bert_init: bool = False,
            activation_fn: str = "relu",
            learned_pos_embedding: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            num_docmask: int = 2,
            check_mode: str = "no",
            window_size: int = -1,
            doc_sentences: int = 32,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable

        self.num_docmask = num_docmask
        self.check_mode = check_mode
        self.window_size = window_size
        self.doc_sentences = doc_sentences

        print("debug check_mode", check_mode)
        print("debug window_size", window_size)
        print("debug num_document_encoder_layers", num_encoder_layers)
        print("debug doc_sentences", doc_sentences)

        # self.embed_tokens = nn.Embedding(
        #     self.vocab_size, self.embedding_dim, self.padding_idx
        # )

        self.embed_scale = embed_scale

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        # add sentence Mask token
        self.doc_mask_embeddings = (
            nn.Embedding(self.num_docmask, self.embedding_dim, padding_idx=None)
            if self.num_docmask > 0
            else None
        )

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    export=export,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            # freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def forward(
            self,
            tokens: torch.Tensor,
            sentence_features: torch.Tensor = None,
            sentence_mask: torch.Tensor = None,
            segment_labels: torch.Tensor = None,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
            end_layer: int = -1,
            enable_nomask: bool = False,
            debug_mode: str = "none"
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = None

        # x = self.embed_tokens(tokens)
        #  B x C  tokens = sentence_features
        x = sentence_features
        batch_size = x.size(0)
        if not enable_nomask:
            # create sentence mask embeds
            mask_tokens = torch.eye(batch_size).type_as(tokens).to(device=x.device)
            mask_embeds = self.doc_mask_embeddings(mask_tokens)
            mask_embeds = mask_embeds * (torch.eye(batch_size).to(device=x.device).unsqueeze(-1)).type_as(mask_embeds)

            # apply sentence mask
            x = x.unsqueeze(0)
            x_indices = (1 - torch.eye(batch_size).to(device=x.device)).type_as(mask_embeds).unsqueeze(-1)
            x = x * x_indices
            # B x B x C
            x = x + mask_embeds
        else:
            x = x.unsqueeze(0).expand(batch_size, -1, -1)

        if "doc_block" in self.check_mode:
            # print("doc_model into doc_block")
            d = self.doc_sentences
            pse = torch.arange(0, batch_size) // d
            mask = pse.unsqueeze(1) == pse.unsqueeze(0)
            mask = (mask == 1)
            y2 = x[mask].reshape(batch_size, d, -1)
            x = y2
            padding_mask = None

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            position_tokens = torch.zeros(x.size(0), x.size(1)).type_as(tokens).to(device=x.device)
            # print("position_tokens ", position_tokens.cpu().detach().numpy())
            x += self.embed_positions(position_tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.emb_layer_norm is not None:
            # print("type x before LN ", x.type())
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)


        for i, layer in enumerate(self.layers):
            if end_layer != -1 and i >= end_layer:
                break
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.layerdrop):
                x, _ = layer(x, self_attn_padding_mask=padding_mask)
                if not last_state_only:
                    inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep

    def forward_layers(self, tokens, inner_states, start_layer, last_state_only):
        padding_mask = tokens.eq(self.padding_idx)
        if self.traceable:
            raise NotImplementedError
        else:
            x = inner_states[-1]
            inner_states = list(inner_states)

        for i, layer in enumerate(self.layers):
            if i < start_layer:
                continue
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.layerdrop):
                x, _ = layer(x, self_attn_padding_mask=padding_mask)
                if not last_state_only:
                    inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
