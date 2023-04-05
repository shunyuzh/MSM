# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.roberta import (
    RobertaModel,
    RobertaHubInterface,
    RobertaLMHead
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
    TransformerSentenceEncoderLayer,
    TransformerDocumentEncoder,
    ProjectionHead,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.tasks.msm import MSMTaskSpace

from fairseq.models.transformer import TransformerDecoder
from collections import namedtuple
import random
# from transformers import BertModel, BertTokenizer, BertForMaskedLM

def index_3d_tensor_with_1d_index(tensor, index):
    batch_size = tensor.size(0)
    word_num = tensor.size(1)

    index = index.clone()
    index.masked_fill_(index == -1, 0)
    index_offset = torch.arange(0, batch_size * word_num, word_num).cuda()
    index = index_offset + index
    tensor = tensor.reshape(batch_size * word_num, -1)
    return torch.index_select(tensor, 0, index).view(batch_size, -1)


@register_model('msm')
class MSMModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return {
            'msm.base': '',
            'msm.large': '',
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--recovery-layer', type=int, default=0,
                            help='which layer to do word recovery attention. 0 means after word embedding. '
                                 'For base model with 12 layers, 12 means after the final layer')
        parser.add_argument('--recovery-start-layer', type=int, default=-1,
                            help='after word recovery attention, which layer start from to recover the input words.'
                                 ' -1 means equals to recovery-layer. 0 means start from first layer. '
                                 'For base model with 12 layers, 11 means after the final layer. ')
        parser.add_argument('--recovery-model', type=str, default='trilinear',
                            help='what kind of model used for word recovery attention. Could be (trilinear, multihead, multihead_reuse)')
        parser.add_argument('--wr_tgt_predict', default=False, action='store_true',
                            help='whether output debug information')
        parser.add_argument('--enable_word_recovery', default=False, action='store_true',
                            help='whether output debug information')
        parser.add_argument('--pseudo_enable_word_recovery', default=False, action='store_true',
                            help='whether output debug information')
        parser.add_argument('--enable-projection-head-bn', default=False, action='store_true',
                            help='whether output debug information')
        parser.add_argument('--sync-bn', default=False, action='store_true',
                            help='Whether use sync batch norm.')
        parser.add_argument('--head-dim-multiples', default=1, type=int,
                            help='the middle dim and out dim of projection head will multiply this parameter')
        parser.add_argument('--head-layers', default=2, type=int,
                            help='the layer of projection head')
        parser.add_argument('--sa-lg-bn', default=False, action='store_true',
                            help='sa use language specific bn')


        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        # parser.add_argument('--decoder-learned-pos', action='store_true',
        #                     help='use learned positional embeddings in the decoder')
        # parser.add_argument('--decoder-normalize-before', action='store_true',
        #                     help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=True, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

        # add sentence reconstruction params
        parser.add_argument('--max-target-positions', type=int, default=128,
                            help='max target positions')
        parser.add_argument('--max-source-positions', type=int, default=512,
                            help='max target positions')
        parser.add_argument('--tokens-per-sample-for-sentence', default=64, type=int,
                            help='tokens per sample for sentence')
        parser.add_argument('--share-lmhead-embedding', default=False, action='store_true',
                            help='share the decoder head embedding with encoder 0 layer')
        parser.add_argument('--split-decoder-embed', default=False, action='store_true',
                            help='split-decoder-embed')
        parser.add_argument('--only-lm', default=False, action='store_true',
                            help='only train decoder')
        parser.add_argument('--freeze-encoder', default=False, action='store_true',
                            help='freeze encoder params')
        parser.add_argument('--encoder-coef', type=float, metavar='D', default=1,
                            help='freeze encoder params')
        # add document model params
        parser.add_argument('--enable-docencoder', default=False, action='store_true',
                            help='whether to enable document encoder')
        parser.add_argument('--document-encoder-layers', type=int, default=2,
                            help='document encoder layers')
        parser.add_argument('--enable-doc-head', default=False, action='store_true',
                            help='add projection head on doc model')
        parser.add_argument('--enable-share-head', default=False, action='store_true',
                            help='query and key share a same head enable_share_head')
        parser.add_argument('--enable-sent2sent', default=False, action='store_true',
                            help='add crop model')
        parser.add_argument('--window-size', default=5, type=int,
                            help='tokens per sample for sentence')
        parser.add_argument('--enable-ict', default=False, action='store_true',
                            help='add ict model')
        parser.add_argument('--check-mode', type=str, default='doc_block',
                            help='dat loader mode for document model')
        parser.add_argument('--inbatch-mode', type=str, default='zero',
                            help='in batch negatives mode for document model')
        parser.add_argument('--enable-logs', default=False, action='store_true',
                            help='enable_logs')
        parser.add_argument('--enable-nomask', default=False, action='store_true',
                            help='no doc mask, enable_nomask')
        parser.add_argument('--debug-mode', type=str, default='no',
                            help='debug_mode')
        parser.add_argument('--enable-balance', default=False, action='store_true',
                            help='enable_balance')
        parser.add_argument('--logits-bias', default=0.5, type=float,
                            help='logits_bias')
        parser.add_argument('--enable-sep-encoder', default=False, action='store_true',
                            help='enable_sep_encoder')
        parser.add_argument('--enable-sep-head', default=False, action='store_true',
                            help='enable_sep_head')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        encoder = MSMEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(self, src_tokens, task=MSMTaskSpace.mlm, features_only=False, return_all_hiddens=False, classification_head_name=None, **kwargs):
        if classification_head_name is not None:
            features_only = True
        sw_x, x, extra = self.decoder(src_tokens, task, features_only, return_all_hiddens, **kwargs)
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return sw_x, x, extra

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

def Embedding(num_embeddings, embedding_dim, padding_idx, from_pretrained_weight=None):
    if from_pretrained_weight==None:
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        new_weight = from_pretrained_weight.clone().detach().requires_grad_(True)
        m = nn.Embedding.from_pretrained(new_weight, padding_idx=padding_idx)
    return m

class MSMEncoder(FairseqDecoder):
    """RoBERTa encoder.

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

        if self.args.enable_projection_head:
            self.projection_head_dic = nn.ModuleDict()
            if self.args.enable_docencoder and self.args.enable_doc_head :
                self.projection_head_dic[MSMTaskSpace.document_model.name] = nn.ModuleList([
                    ProjectionHead(
                        in_dim=self.args.encoder_embed_dim,
                        mid_dim=self.args.encoder_embed_dim,
                        out_dim=self.args.encoder_embed_dim,
                        layer=2,
                        args=self.args
                    ),
                    ProjectionHead(
                        in_dim=self.args.encoder_embed_dim,
                        mid_dim=self.args.encoder_embed_dim,
                        out_dim=self.args.encoder_embed_dim,
                        layer=2,
                        args=self.args
                    )])

        # RoBERTa is a sentence encoder model, so users will intuitively trim
        # encoder layers. However, the implementation uses the fairseq decoder,
        # so we fix here.
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
            args.decoder_layers_to_keep = args.encoder_layers_to_keep
            args.encoder_layers_to_keep = None

        print('max position', args.max_positions)
        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
        )

        # TransformerDocumentEncoder
        print("model args.doc_sentences", args.doc_sentences)
        self.document_encoder = TransformerDocumentEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.document_encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            check_mode=args.check_mode,
            window_size=args.window_size,
            doc_sentences=args.doc_sentences,
        )
        if args.enable_sep_encoder:
            self.sep_encoder = TransformerDocumentEncoder(
                padding_idx=dictionary.pad(),
                vocab_size=len(dictionary),
                num_encoder_layers=args.document_encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                layerdrop=args.encoder_layerdrop,
                max_seq_len=args.max_positions,
                num_segments=0,
                encoder_normalize_before=True,
                apply_bert_init=True,
                activation_fn=args.activation_fn,
                check_mode=args.check_mode,
                window_size=args.window_size,
                doc_sentences=args.doc_sentences,
            )
            if args.enable_sep_head:
                self.projection_head_dic2 = nn.ModuleDict()
                self.projection_head_dic2[MSMTaskSpace.document_model.name] = nn.ModuleList([
                    ProjectionHead(
                        in_dim=self.args.encoder_embed_dim,
                        mid_dim=self.args.encoder_embed_dim,
                        out_dim=self.args.encoder_embed_dim,
                        layer=2,
                        args=self.args
                    ),
                    ProjectionHead(
                        in_dim=self.args.encoder_embed_dim,
                        mid_dim=self.args.encoder_embed_dim,
                        out_dim=self.args.encoder_embed_dim,
                        layer=2,
                        args=self.args
                    )])

        self.lm_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight,
        )

        self.padding_idx = dictionary.pad()

        if self.args.freeze_encoder:
            for name, param in self.sentence_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, src_tokens, task=MSMTaskSpace.mlm, features_only=False, return_all_hiddens=False,
                masked_tokens=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            lg_idx (LongTensor): lg idx of sequence '(batch)'
            lg_idx_dataset (LongTensor): lg idx of word '(batch, src_len)'
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
                  :param return_all_hiddens:
                  :param features_only:
                  :param src_tokens:
                  :param task:
        """

        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        sw_x = None

        # x is  B x T x C
        # contrastive loss for MSM task
        if self.args.enable_docencoder and task == MSMTaskSpace.document_model:
            # sentence_vector is B x C
            sentence_vector = x[:, 0]
            # if sample is None, it is [0, 2, 1, 1,....] 2 means sent sep, 1 means pad
            sentence_mask = src_tokens[:, 1].eq(2)
            extra['sentence_mask'] = sentence_mask
            if self.args.enable_sent2sent:
                # Crop model:
                if self.args.window_size < self.args.doc_sentences:
                    # (1) window_crop, select neighbour sent
                    doc_output = self.window_crop(sentence_vector)
                elif self.args.window_size == self.args.doc_sentences:
                    # (2) random_crop, random crop in the whole doc
                    doc_output = self.random_crop(sentence_vector)
                else:
                    # (3) keep the same
                    doc_output = sentence_vector
            else:
                # Into document model
                doc_output, _ = self.document_encoder_feature(src_tokens, sentence_vector, return_all_hiddens=return_all_hiddens,
                                                              check_mode=self.args.check_mode, sentence_mask=sentence_mask, lg_id=unused['variable_index'])
            if self.args.enable_doc_head:
                if self.args.enable_share_head:
                    projection_heads = self.projection_head_dic[task.name]
                    extra["contrastive_state"] = projection_heads[0](sentence_vector)
                    extra["doc_output"] = projection_heads[0](doc_output)
                else:
                    if unused['variable_index'] == 24 and self.args.enable_sep_head:
                        projection_heads = self.projection_head_dic2[task.name]
                    else:
                        projection_heads = self.projection_head_dic[task.name]
                    extra["contrastive_state"] = projection_heads[0](sentence_vector)
                    extra["doc_output"] = projection_heads[1](doc_output)
            else:
                extra["contrastive_state"] = sentence_vector
                extra["doc_output"] = doc_output

        if not features_only:
            if task == MSMTaskSpace.document_model:
                pass
            else:
                x = self.output_layer(x, masked_tokens=masked_tokens)

        return sw_x, x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def document_encoder_feature(self, src_tokens, sentence_features, return_all_hiddens=False, check_mode='no', sentence_mask=None, lg_id=None, **unused):
        if lg_id == 24 and self.args.enable_sep_encoder:
            # sep doc_encoder of lg en
            inner_states, _ = self.sep_encoder(
                src_tokens,
                sentence_features=sentence_features,
                sentence_mask=sentence_mask,
                last_state_only=not return_all_hiddens,
                enable_nomask=self.args.enable_nomask,
                debug_mode=self.args.debug_mode,
            )
        else:
            # normal encoder
            inner_states, _ = self.document_encoder(
                src_tokens,
                sentence_features=sentence_features,
                sentence_mask=sentence_mask,
                last_state_only=not return_all_hiddens,
                enable_nomask=self.args.enable_nomask,
                debug_mode=self.args.debug_mode,
            )
        features = inner_states[-1].transpose(0, 1)   # T x B x C -> B x T x C
        if "window_mode" in check_mode:
            sentence_features = features[:, features.size(1)//2]
        elif check_mode=="doc_block":
            # features shape 128 * 32 * 768
            d = self.args.doc_sentences
            doc_blocks = sentence_features.size(0) // d
            pse = torch.eye(d, dtype=torch.bool)
            mask = pse.repeat(doc_blocks, 1)
            y2 = features[mask]
            sentence_features = y2
        else:
            features = torch.diagonal(features)
            sentence_features = torch.transpose(features, 0, 1)   # B x C

        return sentence_features,  {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def window_crop(self, sentence_vector):
        win = self.args.window_size
        doc_len = self.args.doc_sentences
        id_len = sentence_vector.size(0)
        id_choice = [ i for i in range(- win, win + 1) if i != 0 ]
        id_list = []
        for i in range(id_len):
            i_new = i % doc_len
            if i_new > doc_len - (win + 1):
                bias = random.choice(id_choice[: doc_len - i_new + win - 1])
            else:
                bias = random.choice(id_choice[- win - i_new:])
            id = i + bias
            id_list.append(id)
        random_ids = torch.tensor(id_list).unsqueeze(-1).expand(-1, sentence_vector.size(1)).to(device=sentence_vector.device)
        doc_output = torch.gather(sentence_vector, 0, random_ids)
        return doc_output

    def random_crop(self, sentence_vector):
        doc_len = self.args.doc_sentences
        id_len = sentence_vector.size(0)
        id_list = []
        for i in range(id_len):
            doc_id = i // doc_len
            id_choice = [ j for j in range(doc_len * doc_id, doc_len * (doc_id + 1)) if j != i ]
            id = random.choice(id_choice)
            id_list.append(id)
        random_ids = torch.tensor(id_list).unsqueeze(-1).expand(-1, sentence_vector.size(1)).to(device=sentence_vector.device)
        doc_output = torch.gather(sentence_vector, 0, random_ids)
        return doc_output


@register_model_architecture('msm', 'msm')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)

    # params about decoder
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)

    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    # shzhang whether share decoder input/output embed
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    # shzhang whether share all embed, True in Transformer
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', True)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.max_target_positions = getattr(args, 'max_target_positions', 128)

@register_model_architecture('msm', 'msm_base')
def roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture('msm', 'msm_small')
def small_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    base_architecture(args)


@register_model_architecture('msm', 'msm_large')
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)
