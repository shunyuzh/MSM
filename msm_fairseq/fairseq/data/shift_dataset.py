# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
from . import FairseqDataset


class ShiftDataset(FairseqDataset):

    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        assert len(datasets[1]) == len(datasets[0]), \
            'datasets must have the same length'

    def __getitem__(self, index):
        items = self.datasets[0][index]
        if torch.is_tensor(self.datasets[1][index]):
            shift = torch.numel(self.datasets[1][index])
        else:
            shift = np.size(self.datasets[1][index])
        return items + shift

    def __len__(self):
        return len(self.datasets[0])

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        return self.datasets[0].sizes

    def num_tokens(self, index):
        return self.datasets[0].num_tokens(index)

    def size(self, index):
        return self.datasets[0].size(index)

    def ordered_indices(self):
        return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.datasets[0], 'supports_prefetch', False)
        

    def prefetch(self, indices):
        if getattr(self.datasets[0], 'supports_prefetch', False):
            self.datasets[0].prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.datasets[0], 'set_epoch'):
            self.datasets[0].set_epoch(epoch)
