# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch

from fairseq.data import data_utils, Dictionary, plasma_utils

from . import BaseWrapperDataset, LRUCacheDataset, TokenBlockDataset


class NextSentenceDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def gen_nsp(cls, dataset: torch.utils.data.Dataset, epoch_shift: int, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""

        return (
            cls(dataset, *args, **kwargs, return_content=1, epoch_shift=epoch_shift),
            cls(dataset, *args, **kwargs, return_content=2, epoch_shift=epoch_shift),
            cls(dataset, *args, **kwargs, return_content=3, epoch_shift=epoch_shift),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        return_content: int,
        epoch_shift: int,
    ):
        self.dataset = dataset
        self.return_mask = False
        self.epoch_shift = epoch_shift
        if return_content == 1:
            self.idx_shift = 0
        elif return_content == 2:
            self.idx_shift = 1
        elif return_content == 3:
            self.idx_shift = 1
            self.return_mask = True
        
        self._sizes = [self.dataset.sizes[index * 2 + self.idx_shift + self.epoch_shift] for index in range(self.__len__())]
        
        self._sizes = plasma_utils.PlasmaArray(np.array(self._sizes))

        self._slice_indices = [self.dataset.slice_indices[index * 2 + self.idx_shift + self.epoch_shift] for index in range(self.__len__())]
        self._slice_indices = plasma_utils.PlasmaArray(np.array(self._slice_indices))


    def __getitem__(self, index):
        if not self.return_mask:
            return self.dataset[index * 2 + self.idx_shift + self.epoch_shift]
        else:
            slice_indices = self.dataset.slice_indices
            s_1, e_1 = slice_indices[index * 2 + 1 + self.epoch_shift]
            s_2, e_2 = slice_indices[index * 2 + self.epoch_shift]
            if s_1 - e_2 == 1:
                return torch.tensor([0])
            else:
                return torch.tensor([1])

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array


    def __len__(self):
        return (len(self.dataset) - self.epoch_shift) // 2

