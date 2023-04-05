# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch.utils.data.dataloader import default_collate
import random
from . import FairseqDataset
import numpy as np

class ShuffleDictionaryDataset(FairseqDataset):

    def __init__(self, defn, sizes=None):
        super().__init__()
        self.defn = defn
        self._len = len(defn)
        self.map_index = [i for i in range(0, self._len)]
        random.shuffle(self.map_index)
        self.sizes = self.defn.sizes
        self.sizes = np.array([self.sizes[self.map_index[i]] for i in range(len(self.sizes))])

    def __getitem__(self, index):
        index = self.map_index[index]
        return self.defn[index]

    def __len__(self):
        return self._len

    def collater(self, samples):
        return self.defn.collater(samples)

    def num_tokens(self, index):
        index = self.map_index[index]
        return self.defn.num_tokens(index)

    def size(self, index):
        index = self.map_index[index]
        return self.defn.size(index)

    @property
    def supports_prefetch(self):
        return self.defn.supports_prefetch()

    def prefetch(self, indices):
        self.defn.prefetch(indices)

    def set_epoch(self, epoch):
        self.defn.set_epoch(epoch)

#    def set_epoch(self, epoch):
#        super().set_epoch(epoch)
#        for ds in self.defn.values():
#            ds.set_epoch(epoch)

