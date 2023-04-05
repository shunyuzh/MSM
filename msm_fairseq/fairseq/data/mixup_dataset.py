# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset
import numpy as np

def _flatten(dico, prefix=None):
    """Flatten a nested dictionary."""
    new_dico = OrderedDict()
    if isinstance(dico, dict):
        prefix = prefix + '.' if prefix is not None else ''
        for k, v in dico.items():
            if v is None:
                continue
            new_dico.update(_flatten(v, prefix + k))
    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            new_dico.update(_flatten(v, prefix + '.[' + str(i) + ']'))
    else:
        new_dico = OrderedDict({prefix: dico})
    return new_dico


def _unflatten(dico):
    """Unflatten a flattened dictionary into a nested dictionary."""
    new_dico = OrderedDict()
    for full_k, v in dico.items():
        full_k = full_k.split('.')
        node = new_dico
        for k in full_k[:-1]:
            if k.startswith('[') and k.endswith(']'):
                k = int(k[1:-1])
            if k not in node:
                node[k] = OrderedDict()
            node = node[k]
        node[full_k[-1]] = v
    return new_dico


class MixupDataset(FairseqDataset):

    def __init__(self, defn, task="tlm", sizes=None):
        super().__init__()
        self.defn = defn
        self.index2range = []
        self.sizes = []
        cur_len = 0
        start = 0
        for i in range(len(defn)):
            if cur_len + defn.size(i) < 512:
                cur_len += defn.size(i)
                continue
            else:
                self.sizes.append(cur_len)
                self.index2range.append([start, i - 1])
                start = i
                cur_len = defn.size(i)
        print("inside mixup:", self.defn[0])
        self._len = len(self.sizes)
        self.sizes = np.array(self.sizes, dtype=np.int64)
    def merge_dict(self, obj1, obj2):
        res = []
        for item_1, item_2 in zip(obj1.keys(), obj2.keys()):
            if not item_1 in ["net_input.mlm.src_lengths", "net_input.tlm.src_lengths", "net_input.next_sentence_prediction.lg_idx", "net_input.task", "nsentences", "index", "ntokens"]:  
                obj1[item_1] = torch.cat([obj1[item_1], obj2[item_2]])
            elif item_1 == "ntokens":
                obj1[item_1] += obj2[item_2]
        return obj1        
    def __getitem__(self, index):
        cur_obj = self.defn[self.index2range[index][0]]
        for item in range(self.index2range[index][0] + 1, self.index2range[index][1] + 1):
            cur_obj = self.merge_dict(cur_obj, self.defn[item])
        return cur_obj

    def __len__(self):
        return self._len

    def collater(self, samples):
        return self.defn.collater(samples)

    def num_tokens(self, index):
        return np.max(self.sizes[index], 0)

    def size(self, index):
        return np.max(self.sizes[index], 0)

    @property
    def supports_prefetch(self):
        return False
#        return self.defn.supports_prefetch()

    def prefetch(self, indices):
        self.defn.prefetch(indices)

    def set_epoch(self, epoch):
        self.defn.set_epoch(epoch)