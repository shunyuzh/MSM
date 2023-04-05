# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import numpy as np
from . import FairseqDataset, data_utils



class ShufflePairDataset(FairseqDataset):
    def __init__(self, dataset_a, dataset_b, shuffle):
        super().__init__()
        self.datasets = [dataset_a, dataset_b]

        self.shuffle = shuffle

        assert len(dataset_a) == len(dataset_b), \
            'datasets must have the same length'
        assert len(dataset_a) == len(shuffle), \
            'datasets and shuffle index must have the same length'
        assert all([item == 0 or item == 1 for item in self.shuffle]), \
            'the index in shuffle must be 0 or 1'

    @classmethod
    def build_dataset(cls, dataset_a, dataset_b, seed):
        with data_utils.numpy_seed(seed):
            shuffle = np.random.randint(2, size=len(dataset_a))
        return cls(dataset_a, dataset_b, shuffle), cls(dataset_b, dataset_a, shuffle)

    def __getitem__(self, index):
        return self.datasets[self.shuffle[index]][index]

    def __len__(self):
        return len(self.datasets[0])

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        s = [self.datasets[0].sizes, self.datasets[1].sizes]
        return np.asarray([s[self.shuffle[i]][i]
                            for i in range(len(self.datasets[0]))])

    def num_tokens(self, index):
        return self.datasets[self.shuffle[index]].num_tokens(index)

    def size(self, index):
        return self.datasets[self.shuffle[index]].size(index)

    def ordered_indices(self):
        return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return any(
            getattr(ds, 'supports_prefetch', False) for ds in self.datasets
        )

    def prefetch(self, indices):
        for ds in self.datasets:
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(epoch)
