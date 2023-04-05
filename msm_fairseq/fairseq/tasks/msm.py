# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import random
import numpy as np
import torch
import datetime

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    ConcatDataset,
    ShiftDataset,
    MinusDataset,
    FairseqDataset,
    ConcatSentencesDataset,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    RawLabelDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    iterators,
    ShufflePairDataset,
    ConstantLikeDataset,
    TruncateDataset,
    NextSentenceDataset,
    KnowledgeDataset,
    DocumentDataset,
    ShuffleDictionaryDataset,
    MixupDataset
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.apply_bpe_int import BPE
from fairseq.data.encoders.utils import get_whole_word_mask
import json

from enum import Enum


class MSMTaskSpace(Enum):
    mlm = 0
    tlm = 1
    document_model = 2


@register_task('msm')
class MSMTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='complete',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--tlm-mask-prob', default=0.15, type=float,
                            help='For task tlm, probability of replacing a token with mask')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')

        parser.add_argument('--epoch-size', default=300000, type=int,
                            help='How many sentence per epoch per gpu')
        parser.add_argument('--multilang-sampling-alpha', type=float, default=0.3,
                            help='smoothing alpha for sample rations across multiple datasets')
        parser.add_argument('--resample', default=False, action='store_true',
                            help='resample dataset based on sentence number')
        parser.add_argument('--mlm-langs', default='en', type=str,
                            help='comma separated list of languages for which we'
                                 ' want to train XLM on')
        parser.add_argument('--tlm-langs', default='', type=str,
                            help='comma separated list of languages for which we'
                                 ' want to train TLM on')
        parser.add_argument('--enable-tlm', default=False, action='store_true',
                            help='whether use tlm')

        parser.add_argument('--enable-nsp', default=False, action='store_true',
                            help='whether use nsp task')
        
        parser.add_argument('--enable-nsp-without-cts', default=False, action='store_true',
                            help='whether use nsp task')
        
        parser.add_argument('--enable-tlm-wa', default=False, action='store_true',
                            help='whether use tlm and wa')

        parser.add_argument('--contrastive-max-sentence', default=-1, type=int,
                            help='The size of contrastive shared variable.if it is -1, it will use max-sentence')


        parser.add_argument('--layer-of-cts-loss', default=-1, type=int,
                            help='layer of cts loss')
        parser.add_argument('--tlm-mixup', default=False, action='store_true',
                            help='mixup on tlm')
        parser.add_argument('--sa-mixup', default=False, action='store_true',
                            help='mixup on sa')
        parser.add_argument('--sa-half-epoch-size', default=False, action='store_true',
                            help='half epoch size on sa')
        parser.add_argument('--tlm-mixup-2', default=False, action='store_true',
                            help='mixup on tlm')
        parser.add_argument('--sa-mixup-2', default=False, action='store_true',
                            help='mixup on sa')
        parser.add_argument('--sa-cut-2', default=False, action='store_true',
                            help='cut on sa')

        parser.add_argument('--tlm-mixup-4', default=False, action='store_true',
                            help='mixup on tlm')
        parser.add_argument('--sa-mixup-4', default=False, action='store_true',
                            help='mixup on sa')

        parser.add_argument('--enable-old-para', default=False, action='store_true',
                            help='whether use old para')
        parser.add_argument('--enable-glove-sample', default=False, action='store_true',
                            help='glove sample on memory bank')

        parser.add_argument('--enable-attentive-memory-bank', default=False, action='store_true',
                            help='attentive memory bank')
        parser.add_argument('--enable-momentum-update', default=False, action='store_true',
                            help='MOCO update method')
        parser.add_argument('--memory-bank-threshold', default=4.0, type=float,
                            help='4.0  4.5  5.0')
        
        parser.add_argument('--enable-gather-clone', default=False, action='store_true',
                            help='enable gather clone')

        parser.add_argument('--nsp-scale', default=1.0, type=float,
                            help='0.01 0.1 1.0')

        parser.add_argument('--momentum-update-rate', default=0.99, type=float,
                            help='0.9  0.99  0.999')

        parser.add_argument('--enable-sa', default=False, action='store_true',
                            help='enable sentence alignment')

        parser.add_argument('--enable-mlm-sa', default=False, action='store_true',
                            help='enable mlm in sa task')

        parser.add_argument('--align-pair-limit', default=10, type=int,
                            help='limit of align pair')

        parser.add_argument('--total-bs-multiply', default=1, type=int,
                            help='total batch size multiply')

        parser.add_argument('--mlm-dup', default=1, type=int,
                            help='duplicate the mlm data to control the train ratio between different tasks')
        parser.add_argument('--tlm-double-batch-size', default=False, action = 'store_true',
                            help='whether use double batch of tlm task.')
        parser.add_argument('--sa-double-batch-size', default=False, action = 'store_true',
                            help='whether use double batch of sa task.')
        parser.add_argument('--tlm-bs-multiply', default=1, type=int,
                            help='whether use large batch size for tlm. '
                                 'The real batch size will be max_sentence * this_parameter')
        parser.add_argument('--sa-bs-multiply', default=1, type=int,
                            help='whether use large batch size for sa. '
                                 'The real batch size will be max_sentence * this_parameter')

        parser.add_argument('--doc-double-batch-size', default=False, action = 'store_true',
                            help='whether use double batch of doc task.')
        parser.add_argument('--doc-bs-multiply', default=1, type=int,
                            help='whether use large batch size for document model task. doc_bs_multiply '
                                 'The real batch size will be max_sentence * this_parameter')

        # masking
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--enable-l2-norm', default=False, action='store_true',
                            help='l2 norm')
        parser.add_argument('--enable_nsp_without_cts', default=False, action='store_true',
                            help='remove cts loss')
        parser.add_argument('--enable-projection-head', default=False, action='store_true',
                            help='projection head')
        parser.add_argument('--enable-sa-head', default=False, action='store_true',
                            help='projection head of sa')
        parser.add_argument('--enable-multi-projection-head', default=False, action='store_true',
                            help='multi projection head')
        parser.add_argument('--enable-memory-bank', default=False, action='store_true',
                            help='enable meomry bank')
        parser.add_argument('--memory-bank-size', default=4096, type=int,
                            help='size of memory bank')
 
        parser.add_argument('--cts-temp', default=0.1, type=float,
                            help='temperature of contrastive loss')
        parser.add_argument('--enable-rm-tlm', default=False, action='store_true',
                            help='rm tlm from tlm_mlm_wa')
        parser.add_argument('--enable-sw', default=False, action='store_true',
                            help='enable setence to word prediction')
   
        parser.add_argument('--enable-sw-in-sa', default=False, action='store_true',
                            help='enable setence to word prediction in sentence alignment')

        parser.add_argument('--enable-lg-specific-memory-bank', default=False, action='store_true',
                            help='enable lg specific memory bank')

        parser.add_argument('--disable-memory-bank-push', default=False, action='store_true',
                            help='disable push to memory bank')

        parser.add_argument('--ngram-mode', default="fairseq", type=str,
                            help="could be (fairseq, fixed, spanbert), "
                                 "fairseq means the raw code of fairseq codebase, which count masked ratio on on word level"
                                 "ngram means always sample n-gram, also will count the masked ratio on token level"
                                 "spanbert means sample the n of n-gram like spanbert")
        parser.add_argument('--ngram-n', default=1, type=int,
                            help="the n for fixed ngram mode")
        parser.add_argument('--mix-task-in-same-batch', default=False, action='store_true',
                            help='whether use mixed task in same batch')
        parser.add_argument('--old-dataset', default=False, action='store_true',
                            help='whether use mixed task in same batch')
        parser.add_argument('--old-generator', default=False, action='store_true',
                            help='whether use mixed task in same batch')
        parser.add_argument('--enable-tlm-shuffle', default=False, action='store_true',
                            help='whether use mixed task in same batch')
        parser.add_argument('--enable-sa-shuffle', default=False, action='store_true',
                            help='whether use mixed task in same batch')
        parser.add_argument('--en-first', default=False, action='store_true',
                            help='whether use mixed task in same batch')
        parser.add_argument('--en-second', default=False, action='store_true',
                            help='whether use mixed task in same batch')
        parser.add_argument('--debug-verbose', default=False, action='store_true',
                            help='whether output debug information')
        parser.add_argument('--tlm-mask-one-lg', default=False, action='store_true',
                            help='whether only mask one language in tlm')

        parser.add_argument('--enable-word2doc', default=False, action='store_true',
                            help='enable word2doc loss')
        parser.add_argument('--enable-doc2word', default=False, action='store_true',
                            help='enable doc2word loss')
        parser.add_argument('--disable-mlm', default=False, action='store_true',
                            help='enable doc2word loss')

        parser.add_argument('--sa-data', default='', type=str,
                            help='the data folder of task SA. If it is empty, we will use folder of tlm' )
        # wiki definition data
        parser.add_argument('--wd-text-data', default='', type=str,
                            help='the folder for knowledge BIO data')
        parser.add_argument('--BIO-data', default='', type=str,
                            help='the folder for knowledge BIO data')
        parser.add_argument('--DE-data', default='', type=str,
                            help='the file-path for knowledge document embedding data')
        parser.add_argument('--DE-cross-data', default='', type=str,
                    help='the file-path for knowledge document embedding cross lingual index data')
        parser.add_argument('--cross-DE-label', default=False, action='store_true',
                    help='if align with knowledge document embedding cross lingual index data')
        # wiki definition parameters
        parser.add_argument('--DE-max-length', default=254, type=int,
                            help='the max length for knowledge document embedding data')
        parser.add_argument('--enable-wd-lg-specific-memory-bank', default=False, action='store_true',
                            help='enable lg specific memory bank')
        parser.add_argument('--word2doc-scale', default=1.0, type=float,
                            help='0.01 0.1 1.0')
        parser.add_argument('--doc2word-scale', default=1.0, type=float,
                            help='0.01 0.1 1.0')

        parser.add_argument('--enable-bn-queue', default=False, action='store_true',
                            help='Whether use an additional queue for bn')
        parser.add_argument('--bn-queue-size', default=0, type=int,
                            help='the size of bn queue')
        parser.add_argument('--bn-queue-shuffle', default=False, action='store_true',
                            help='shuffle the item in bn-queue before calculate bn')
        parser.add_argument('--bn-queue-push-half', default=False, action='store_true',
                            help='whether only push one item in parallel pair')
        parser.add_argument('--bn-queue-no-gather', default=False, action='store_true',
                            help='Whether bn queue only use the item in same card')

        parser.add_argument('--batch-pure-lg-sa', default=False, action='store_true',
                            help='Whether each batch only have one language in task SA')
        parser.add_argument('--batch-pure-lg-mlm', default=False, action='store_true',
                            help='Whether each batch only have one language in task MLM')
        parser.add_argument('--batch-pure-lg-tlm', default=False, action='store_true',
                            help='Whether each batch only have one language in task TLM')
        parser.add_argument('--batch-pure-lg-doc', default=False, action='store_true',
                            help='Whether each batch only have one language in task DOC')

        parser.add_argument('--sts-data', default='', type=str,
                            help='the folder for sts data')
        parser.add_argument('--doc-data', default='', type=str,
                            help='the folder for document model data')
        parser.add_argument('--enable-asyn-bn', default=False, action='store_true',
                            help='in projection head list, enable-asyn-bn')
        parser.add_argument('--enable-doc-block', default=False, action='store_true',
                            help='enable-doc-block')
        parser.add_argument('--doc-sentences', default=32, type=int,
                            help='document model: block sentences number  doc_sentences')
        parser.add_argument('--doc-sentence-tokens', default=64, type=int,
                            help='document model: block sentences max tokens')
        parser.add_argument('--doc-sentences-low', default=32, type=int,
                            help='when sentences < doc_sentences_low, this sample abandon')
        parser.add_argument('--mlm-data-scale', default=0.4, type=float,
                            help='to scale mlm_data_scale')
        parser.add_argument('--tiny-scale', default=1, type=int, help='tiny_scale')
        parser.add_argument('--enable-bert', default=False, action='store_true',
                            help='enable_bert')
        parser.add_argument('--enable-bert-hf', default=False, action='store_true',
                            help='enable_bert_hf')
        parser.add_argument('--num-segments', default=0, type=int, help='num_segments')
        parser.add_argument('--enable-noeqlen', default=False, action='store_true',
                            help='enable_noeqlen')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.size_dic = {}
        # add mask token
        if args.enable_bert:
            self.mask_idx = 103
        else:
            self.mask_idx = dictionary.add_symbol('<mask>')

        self.lg2id = {}

        paths = args.data.split(os.pathsep)
        assert len(paths) > 0

        for pair in args.tlm_langs.split(','):
            for lg in pair.split('-'):
                if lg not in self.lg2id:
                    self.lg2id[lg] = len(self.lg2id)

        for lg in args.mlm_langs.split(','):
            if lg not in self.lg2id:
                self.lg2id[lg] = len(self.lg2id)

        print('lg2id:', self.lg2id)

        self.path_cache = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(os.pathsep)
        assert len(paths) > 0
        print("msm args.enable_bert", args.enable_bert)
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'), enable_bert=args.enable_bert)
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def _get_sample_prob(self, dataset_word_lengths):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_word_lengths / dataset_word_lengths.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def get_dataset_path(self, split, data_folder, epoch, lgs, is_pair=False, is_mixup=False):
        if data_folder in self.path_cache:
            files = self.path_cache[data_folder]
        else:
            files = [path for path in os.listdir(data_folder)]
            # remove this to speed up
            # if os.path.isfile(os.path.join(data_folder, path))
            self.path_cache[data_folder] = files
        files = [path for path in files if(split in path) and (".bin" in path)]
        paths = []
        epoch_shift_list = []
        for lg_index, lg in enumerate(lgs):
            if is_pair:
                pair = lg.split('-')
                if is_mixup:
                    pair = ['a', 'b']
                split_count_0 = len([path for path in files if ".{0}.{1}.bin".format(lg, pair[0]) in path])
                split_count_1 = len([path for path in files if ".{0}.{1}.bin".format(lg, pair[1]) in path])

                if split_count_0 != split_count_1:
                    split_count = 0
                else:
                    split_count = split_count_0
            else:
                split_count = len([path for path in files if ".{0}.bin".format(lg) in path])
            if split_count == 0:
                print("Did find language {0} in {1} part of folder {2}".format(lg, split, data_folder))
                paths.append(None)
                epoch_shift_list.append(0)
                continue
            big_step = epoch // split_count
            epoch_shift_list.append(big_step % 2)
            small_step = epoch % split_count
            with data_utils.numpy_seed((self.args.seed + big_step) * 100 + lg_index):
                shuffle = np.random.permutation(split_count)
                index = shuffle[small_step]
                path = os.path.join(data_folder, "{0}.{1}.{2}".format(split, index, lg))
                paths.append(path)
        return paths, epoch_shift_list

    def load_mlm_dataset(self, split, split_path, combine, task, epoch_shift=None):
        assert split_path is not None
        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        if task == MSMTaskSpace.document_model:
            max_per_sample = self.args.tokens_per_sample_for_sentence
        else:
            max_per_sample = self.args.tokens_per_sample

        if self.args.enable_doc_block and task == MSMTaskSpace.document_model:
            print("into doc_block TokenBlockDataset")
            print("into doc_block doc_sentences_low = ", self.args.doc_sentences_low)
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.doc_sentence_tokens - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="doc_block",
                enable_doc_block=True,
                doc_sentences=self.args.doc_sentences,
                doc_sentences_low=self.args.doc_sentences_low,
                )
        else:
            # debug: manually set max_per_sample=16
            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                max_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode=self.args.sample_break_mode,
            )

        print('| loaded {} blocks from: {}'.format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        return dataset

    def create_mask(self, dataset, index, dataset_a=None, dataset_b=None,
                    task=MSMTaskSpace.mlm,
                    first_dataset_sizes=None,
                    dataset_prev=None,
                    dataset_next=None,
                    dataset_next_mask=None,
                    BIO_dataset=None,
                    document_dataset=None,
                    DE_cross_index_dataset=None,
                    lg=None,
                    src_dataset_input_map=None,
                    tgt_dataset_input_map=None,
                    ):
        lg_index = self.lg2id[lg]
        assert isinstance(task, MSMTaskSpace)
        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) if self.args.mask_whole_words else None
        print("task is {0}".format(task))

        if task in [MSMTaskSpace.mlm,
                    MSMTaskSpace.document_model]:
            mask_prob = self.args.mask_prob
            leave_unmasked_prob = self.args.leave_unmasked_prob
            random_token_prob = self.args.random_token_prob
        elif task == MSMTaskSpace.tlm:
            mask_prob = self.args.tlm_mask_prob
            leave_unmasked_prob = self.args.leave_unmasked_prob
            random_token_prob = self.args.random_token_prob
        else:
            raise ValueError("unknown task {0}".format(task))

        print("mask prob is {0}".format(mask_prob))

        mask_idx = self.mask_idx

        def do_mask(raw_dataset, **kwargs):
            dataset_input, dataset_label = MaskTokensDataset.apply_mask(
                raw_dataset,
                self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=mask_idx,
                seed=self.args.seed,
                mask_prob=mask_prob,
                leave_unmasked_prob=leave_unmasked_prob,
                random_token_prob=random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
                **kwargs,
            )
            return dataset_input, dataset_label

        if task == MSMTaskSpace.document_model:
            dataset_input = dataset
            dataset_label = dataset
            main_dataset = dataset_input
            ntokens = NumelDataset(dataset_input, reduce=True)
            sizes = dataset_input.sizes
        elif task == MSMTaskSpace.mlm or task == MSMTaskSpace.tlm:
            dataset_input, dataset_label = do_mask(dataset)
            main_dataset = dataset_input
            ntokens = NumelDataset(dataset_input, reduce=True)
            sizes = dataset_input.sizes
        else:
            raise ValueError("Unknown task: {0}".format(task.name))

        length = main_dataset.sizes.shape[0]
        default_dataset = TruncateDataset(main_dataset, 2)
        dataset_arch = {
            'net_input': {
                MSMTaskSpace.document_model.name: {
                    'src_tokens': PadDataset(
                        dataset_input if task == MSMTaskSpace.document_model else default_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'lg_idx': RawLabelDataset([lg_index] * length),
                },
                MSMTaskSpace.mlm.name: {
                    'src_tokens': PadDataset(
                        dataset_input if task == MSMTaskSpace.mlm else default_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'src_lengths': NumelDataset(
                        dataset_input if task == MSMTaskSpace.mlm else default_dataset,
                        reduce=False
                    ),
                },
                MSMTaskSpace.tlm.name: {
                    'src_tokens': PadDataset(
                        dataset_input if task == MSMTaskSpace.tlm else default_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'src_lengths': NumelDataset(
                        dataset_input if task == MSMTaskSpace.tlm else default_dataset,
                        reduce=False
                    ),
                },
                'task': RawLabelDataset([task.value] * length),
            },
            'targets': {
                MSMTaskSpace.document_model.name: {
                    'target': PadDataset(
                        dataset_label if task == MSMTaskSpace.document_model else default_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                },
                MSMTaskSpace.mlm.name: {
                    'target': PadDataset(
                        dataset_label if task == MSMTaskSpace.mlm else default_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                },
                MSMTaskSpace.tlm.name: {
                    'target': PadDataset(
                        dataset_label if task == MSMTaskSpace.tlm else default_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                },
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': ntokens,
            'index': RawLabelDataset([lg_index] * length),
        }

        lang_dataset = NestedDictionaryDataset(
            dataset_arch,
            sizes=[sizes],
        )
        return lang_dataset

    def load_all_mlm_dataset(self, split, mono_data_path, epoch, mlm_langs, combine=False):
        mono_datasets = []
        mono_langs = []
        paths, epoch_shift_list = self.get_dataset_path(split, mono_data_path, epoch, mlm_langs)
        task = MSMTaskSpace.mlm

        for index, language in enumerate(mlm_langs):
            if language == "":
                continue
            split_path = paths[index]

            dataset = self.load_mlm_dataset(
                split, split_path, combine,
                task=task,
                epoch_shift=epoch_shift_list[index])

            lang_dataset = self.create_mask(dataset, index, lg=language, task=task)
            # check the file is loadable
            last_sample = lang_dataset[len(lang_dataset) - 1]

            print("creat mask finished")
            mono_datasets.append(lang_dataset)
            mono_langs.append(language)

            if self.args.debug_verbose:
                print("=" * 60)
                print(language)
                print('mlm task')
                for i in range(0, 2):
                    sample = lang_dataset[i]
                    input_token = sample["net_input.mlm.src_tokens"]
                    print('input_token', input_token)
                    print(self.source_dictionary.string(input_token))
        return mono_datasets, mono_langs

    def load_all_document_model_dataset(self, split, mono_data_path, epoch, mlm_langs, combine=False):
        mono_datasets = []
        mono_langs = []
        paths, epoch_shift_list = self.get_dataset_path(split, mono_data_path, epoch, mlm_langs)
        task = MSMTaskSpace.document_model

        for index, language in enumerate(mlm_langs):
            if language == "":
                continue
            split_path = paths[index]

            dataset = self.load_mlm_dataset(
                split, split_path, combine,
                task=task,
                epoch_shift=epoch_shift_list[index])

            if self.args.enable_doc_block:
                dataset = TruncateDataset(dataset, self.args.doc_sentence_tokens)

            lang_dataset = self.create_mask(dataset, index, lg=language, task=task)
            # check the file is loadable
            # last_sample = lang_dataset[len(lang_dataset) - 1]

            print("creat mask finished")
            mono_datasets.append(lang_dataset)
            mono_langs.append(language)

            if self.args.debug_verbose:
                print("=" * 60)
                print(language)
                print('document_model task')
                for i in range(0, 8):
                    if i % self.args.doc_sentences == 0:
                        print("===" * 10)
                    sample = lang_dataset[i]
                    input_token = sample["net_input.document_model.src_tokens"]
                    print('document_model input_token', input_token)
                    print(self.source_dictionary.string(input_token))
        return mono_datasets, mono_langs

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(os.pathsep)
        assert len(paths) > 0
        # data_path = paths[epoch % len(paths)]
        mono_data_path = paths[0]

        mlm_langs = list(sorted(self.args.mlm_langs.split(',')))

        print("| Training on {0} mlm_langs: {1}".format(len(mlm_langs), mlm_langs))
        print("| Language to id mapping: ", {
                lang: id for id, lang in enumerate(mlm_langs)
            }
        )
        dataset_dic = {}

        if self.args.doc_data != "":
            print("into doc data")
            sts_path = self.args.doc_data.split(os.pathsep)[0]
            # loading monolingual dataset and prepare task sentence reconstruction
            mono_datasets = []
            mono_langs = []
            tmp_dataset, tmp_langs = self.load_all_document_model_dataset(split, sts_path, epoch, mlm_langs, combine)
            mono_datasets += tmp_dataset
            mono_langs += tmp_langs
            if len(mono_datasets) > 0:
                dataset_dic[MSMTaskSpace.document_model] = (mono_datasets, mono_langs)

        # loading monolingual dataset and prepare task mlm or next sentence prediction
        mono_datasets = []
        mono_langs = []

        tmp_dataset, tmp_langs = self.load_all_mlm_dataset(split, mono_data_path, epoch, mlm_langs, combine)
        mono_datasets += tmp_dataset
        mono_langs += tmp_langs

        if not self.args.disable_mlm:
            dataset_dic[MSMTaskSpace.mlm] = (mono_datasets, mono_langs)

        dataset_list = []
        for task, (datasets, dataset_langs) in dataset_dic.items():
            batch_pure = False
            if task == MSMTaskSpace.mlm and self.args.batch_pure_lg_mlm:
                batch_pure = True
            if task == MSMTaskSpace.document_model and self.args.batch_pure_lg_doc:
                batch_pure = True

            if batch_pure:
                for dataset, lang in zip(datasets, dataset_langs):
                    dataset_list.append((task.value, lang, ([dataset], [lang])))
            else:
                dataset_list.append((task.value, 'all', (datasets, dataset_langs)))

        dataset_list = list(sorted(dataset_list))

        sizes = {}
        for task_id, lang, (datasets, dataset_langs) in dataset_list:
            sizes[(task_id, lang)] = sum([len(dataset) for dataset in datasets])

        lang_datasets = []
        langs = []
        for task_id, lang, (datasets, dataset_langs) in dataset_list:
            lang_datasets += datasets
            langs += dataset_langs
            print("task = {0}, languages = {1}".format(MSMTaskSpace(task_id).name, dataset_langs))

        if split == self.args.train_subset:
            dataset = ConcatDataset(lang_datasets)
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for task_id, task_lang, (datasets, dataset_langs) in dataset_list:
                for lang, lang_dataset in zip(dataset_langs, datasets):
                    split_name = split + '_' + lang
                    lang_splits.append(split_name)
                    self.datasets[split_name] = lang_dataset
                    self.size_dic[split_name] = {task_id: len(lang_dataset)}

            # language individually. Maybe need task API changes to allow it
            # in more generic ways.
            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ','.join(lang_splits)
                )

        # All samples were shuffled, but actually the data in the Dataset were not shuffled, only the index was shuffled
        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))


        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

        self.size_dic[split] = sizes

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    def build_task_separate_iter(self, dataset, indices, max_sentences, filter_max_positions, max_tokens, required_batch_size_multiple, ignore_invalid_inputs, epoch=0):
        sizes = None
        for key, value in self.datasets.items():
            if dataset is value:
                sizes = self.size_dic[key]
        assert sizes is not None

        size_list = list(sorted(sizes.items()))
        size_range_dic = {}
        accum = 0
        for (task_id, lang), size in size_list:
            size_range_dic[(task_id, lang)] = (accum, accum+size)
            accum += size

        print("size_range_dic", size_range_dic)

        batch_number_dic = {}
        batch_sampler = []
        doc_batch_sampler = []
        seed = epoch + 1
        for (task_id, lang), (start, end) in size_range_dic.items():
            task = MSMTaskSpace(task_id)
            if start == end:
                continue
            dataset_indices = np.asarray([index for index in indices if start <= index < end])
            print("---------------------- task {0}, lg {1} ----------------------".format(task.name, lang))
            print("dataset_indices = {0}".format(dataset_indices[:2]))

            # Sort to recover the original order
            if task == MSMTaskSpace.document_model:
                dataset_indices = np.sort(dataset_indices)
                print("sorted dataset_indices = {0}".format(dataset_indices[:2]))
                if self.args.tiny_scale > 2:
                    dataset_indices = dataset_indices[ : len(dataset_indices) // self.args.tiny_scale]
                doc_sentences = self.args.doc_sentences
                end = len(dataset_indices) // doc_sentences * doc_sentences
                dataset_indices = dataset_indices[:end]

            if task == MSMTaskSpace.document_model:
                dataset_batch_sampler2 = data_utils.batch_by_size(
                    dataset_indices, dataset.num_tokens, max_tokens=max_tokens,
                    max_sentences=self.args.doc_sentences,
                    required_batch_size_multiple=required_batch_size_multiple,
                )
                with data_utils.numpy_seed(seed):
                    np.random.shuffle(dataset_batch_sampler2)
                if len(dataset_batch_sampler2) > 1:
                    dataset_indices = np.concatenate(dataset_batch_sampler2)

            dataset_max_sentences = max_sentences
            print("filter max positions:", filter_max_positions)
            dataset_multiply = 1
            if task == MSMTaskSpace.document_model:
                dataset_multiply = self.args.doc_bs_multiply
                if dataset_multiply == 1 and self.args.doc_double_batch_size:
                    dataset_multiply = 2

            if (self.args.enable_bert or self.args.enable_noeqlen) and task == MSMTaskSpace.document_model:
                print("self.args.doc_sentence_tokens", self.args.doc_sentence_tokens)
                dataset_indices = data_utils.filter_by_size(
                    dataset_indices, dataset, self.args.doc_sentence_tokens,
                    raise_exception=(not ignore_invalid_inputs),
                )
                dataset_max_sentences *= self.args.doc_bs_multiply
            else:
                print('filter_max_positions // dataset_multiply', filter_max_positions // dataset_multiply)
                dataset_indices = data_utils.filter_by_size(
                    dataset_indices, dataset, filter_max_positions // dataset_multiply, raise_exception=(not ignore_invalid_inputs),
                )
                dataset_max_sentences *= dataset_multiply

            print("sample number of task {0}, language {1}: {2}".format(task.name, lang, dataset_indices.size))
            # create mini-batches with given size constraints
            dataset_batch_sampler = data_utils.batch_by_size(
                dataset_indices, dataset.num_tokens, max_tokens=max_tokens,
                max_sentences=dataset_max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
            dataset_batch_sampler = [item for item in dataset_batch_sampler if len(item) == dataset_max_sentences]

            print("before:", len(dataset_batch_sampler))
            if task == MSMTaskSpace.mlm and self.args.mlm_data_scale < 1 :
                if self.args.doc_sentences_low == 32 or self.args.doc_sentences_low == 16 or self.args.doc_sentences_low < 9:
                    scale_number = int(self.args.mlm_data_scale * len(dataset_batch_sampler))
                    dataset_batch_sampler = dataset_batch_sampler[: scale_number]
                    if self.args.tiny_scale > 2:
                        dataset_batch_sampler = dataset_batch_sampler[ : len(dataset_batch_sampler) // self.args.tiny_scale]
            print("after:", len(dataset_batch_sampler))

            if task == MSMTaskSpace.document_model:
                # shuffle between batches
                with data_utils.numpy_seed(seed):
                    np.random.shuffle(dataset_batch_sampler)

            # batch number to int * NGPU
            batch_num2 = len(dataset_batch_sampler) // self.args.distributed_world_size * self.args.distributed_world_size
            dataset_batch_sampler = dataset_batch_sampler[:batch_num2]
            print("after int * NGPU:", len(dataset_batch_sampler))

            if task == MSMTaskSpace.document_model:
                doc_batch_sampler += dataset_batch_sampler
            else:
                batch_sampler += dataset_batch_sampler

            if dataset_batch_sampler is not None and len(dataset_batch_sampler) > 0:
                print("batch size of task {0}, language {1} is: {2}".format(task.name, lang, len(dataset_batch_sampler[0])))
            if task_id not in batch_number_dic:
                batch_number_dic[task_id] = 0
            if dataset_batch_sampler is not None and len(dataset_batch_sampler) > 0:
                batch_number_dic[task_id] += len(dataset_batch_sampler)
        for task_id, cnt in batch_number_dic.items():
            print("task={0}, batch number={1}".format(MSMTaskSpace(task_id), cnt))

        # shuffle tasks: batch_num * bs -> int * Ngpu * bs
        if self.args.enable_cross_gpu_data:
            print("msm into self.args.enable_cross_gpu_data")
            batch_sampler += doc_batch_sampler
            with data_utils.numpy_seed(seed):
                id = np.arange(len(batch_sampler)).reshape(-1, self.args.distributed_world_size)
                id_permuted = np.random.permutation(id)
            id_permuted = id_permuted.reshape(len(batch_sampler))
            batch_sampler = np.array(batch_sampler, dtype=object)
            batch_sampler = batch_sampler[id_permuted].tolist()
        else:
            batch_sampler += doc_batch_sampler
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batch_sampler)

        print("all batches number: ", len(batch_sampler))

        return batch_sampler, dataset_max_sentences

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)
        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        filter_max_positions = self.args.tokens_per_sample
        if max_positions is not None:
            filter_max_positions = min(filter_max_positions, max_positions)

        print("filter by max position {0}".format(filter_max_positions))
        indices = data_utils.filter_by_size(
            indices, dataset, filter_max_positions, raise_exception=(not ignore_invalid_inputs),
        )


        if self.args.mix_task_in_same_batch:
            batch_sampler = data_utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler, dataset_max_sentences = self.build_task_separate_iter(
                dataset, indices, max_sentences, filter_max_positions, max_tokens,
                required_batch_size_multiple, ignore_invalid_inputs, epoch)

        print("batch size test over")
        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        return epoch_iter


    def max_positions(self):
        """Return the max input length allowed by the task."""
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
