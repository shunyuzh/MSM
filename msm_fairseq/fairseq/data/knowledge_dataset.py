# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import numpy as np
import random
import torch

from . import BaseWrapperDataset


class KnowledgeDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        # item = self.dataset[idx]
        BIO = self.dataset[idx].tolist()  # list len

        start_token_index_list = []

        for i in range(len(BIO)):
            if BIO[i] > 4:
                start_token_index_list.append(i)
        if len(start_token_index_list) == 0:
            return torch.LongTensor([-1, -1, 2])  # , torch.LongTensor([-1]), torch.LongTensor([-2]) # main page aa

        random.seed(2020)
        index = random.sample(range(0, len(start_token_index_list)), 1)[0]  # int
        start_pos = start_token_index_list[index]  # start token pos

        end_pos = start_pos
        while end_pos+1 < len(BIO) and BIO[end_pos+1] == 1:
            end_pos += 1

        DE_emb_index = BIO[start_pos]
        return torch.LongTensor([start_pos, end_pos, DE_emb_index])  # , torch.LongTensor([end_pos]), self.DE_dataset[DE_emb_index]

    @property
    def sizes(self):
        return self._sizes

