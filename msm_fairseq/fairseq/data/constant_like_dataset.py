# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

from . import BaseWrapperDataset
import torch


class ConstantLikeDataset(BaseWrapperDataset):

    def __init__(self, dataset, value):
        super().__init__(dataset)
        self.dataset_sizes = dataset.sizes
        self.value = value

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        size = self.dataset_sizes[index]
        item = torch.ones(size, dtype=torch.int) * self.value
        return item