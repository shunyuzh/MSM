
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import torch

from . import BaseWrapperDataset


class DocumentDataset(BaseWrapperDataset):
    def __init__(self, dataset, DE_dataset, DE_cross_index_dataset, corss_index_label):
        super().__init__(dataset)
        self.DE_dataset = DE_dataset
        self.DE_cross_index_dataset = DE_cross_index_dataset
        self.cross_index_label = corss_index_label

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.cross_index_label:
            index = self.DE_cross_index_dataset[item[2]][0].long() # self.index_dataset[item[2]] tensor([11729949,        2])
        else:
            index = item[2]
        # print("cross-index",index)
        return self.DE_dataset[index] # self.DE_dataset[item[2]]

    @property
    def sizes(self):
        return self._sizes
