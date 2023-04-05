# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
from . import FairseqDataset


class MinusDataset(FairseqDataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        items = self.dataset[index][:-1]
        return items - 3

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return self.dataset.collater(samples)

    @property
    def sizes(self):
        return self.dataset.sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)
        

    def prefetch(self, indices):
        if getattr(self.dataset, 'supports_prefetch', False):
            self.dataset.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
