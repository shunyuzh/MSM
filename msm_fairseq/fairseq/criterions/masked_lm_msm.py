# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from fairseq.data import Dictionary

from . import FairseqCriterion, register_criterion
from fairseq.modules import ContrastiveLossWithQueue, MultiTaskGather, MemoryEfficientGather2
from fairseq.tasks.msm import MSMTaskSpace
import pickle
import os
from enum import Enum, auto
import random
import numpy
import json

class MSMLossSpace(Enum):
    mlm = auto()
    tlm = auto()
    doc_model = auto()
    doc_acc = auto()


@register_criterion('masked_lm_msm')
class MaskedLmMsmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        paths = args.data.split(os.pathsep)
        assert len(paths) > 0
        print("args.enable_bert", args.enable_bert)
        self.dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'), enable_bert=args.enable_bert)

        self.contrastive_loss_dic = {}
        if self.args.enable_docencoder:
            self.contrastive_loss_dic[MSMTaskSpace.document_model] = ContrastiveLossWithQueue(
                queue_size=self.args.memory_bank_size,
                dim=self.args.encoder_embed_dim * self.args.head_dim_multiples,
                args=self.args,
                multi_queue=self.args.enable_lg_specific_memory_bank,    # Set True == 一个queue一种语言
            )
        if self.args.enable_cross_gpu:
            self.gather_list_variable0 = MemoryEfficientGather2(self.args.distributed_world_size)
            self.gather_list_variable1 = MemoryEfficientGather2(self.args.distributed_world_size)

        if len(self.contrastive_loss_dic) > 0:
            contrastive_max_sentence = self.args.max_sentences if self.args.contrastive_max_sentence == -1 \
                else self.args.contrastive_max_sentence
            self.variable_0_gather = MultiTaskGather(
                batch_size=contrastive_max_sentence,
                hidden_size=self.args.encoder_embed_dim,
                distributed_world_size=self.args.distributed_world_size,
                enable_bn_queue=self.args.enable_bn_queue,
            )
            self.variable_1_gather = MultiTaskGather(
                batch_size=contrastive_max_sentence,
                hidden_size=self.args.encoder_embed_dim,
                distributed_world_size=self.args.distributed_world_size,
                enable_bn_queue=self.args.enable_bn_queue,
            )
        self.src_queue = None
        self.tgt_queue = None
        self.eval_state = 0

        self.moco_model = [] # use an ordinary list to avoid it's been iterator by function parameter()

    def push(self, src, tgt):
        src = src.detach()
        tgt = tgt.detach()
        if self.src_queue is None:
            self.src_queue = src
        else:
            self.src_queue = torch.cat([self.src_queue, src], dim=0)
        if self.tgt_queue is None:
            self.tgt_queue = tgt
        else:
            self.tgt_queue = torch.cat([self.tgt_queue, tgt], dim=0)

        if self.src_queue.size()[0] > 1000:
            self.get_acc()

    def get_acc(self):
        if self.src_queue is None:
            return 0
        src_queue = F.normalize(self.src_queue, dim=-1, eps=1e-4)
        tgt_queue = F.normalize(self.tgt_queue, dim=-1, eps=1e-4)
        logits = torch.matmul(src_queue, tgt_queue.T)

        prediction = torch.argmax(logits, dim=1)
        print("number items:", prediction.size()[0])
        label = torch.arange(prediction.size()[0], dtype=torch.long).cuda()
        acc = torch.sum(prediction == label).item() / label.size()[0]
        print("accuracy src2tgt", acc)

        prediction = torch.argmax(logits.T, dim=1)
        print("number items:", prediction.size()[0])
        label = torch.arange(prediction.size()[0], dtype=torch.long).cuda()
        acc = torch.sum(prediction == label).item() / label.size()[0]
        print("accuracy tgt2src", acc)

        self.src_queue = None
        self.tgt_queue = None
        return acc

    def update_variable(self, extra_src, extra_tgt, src_lg):
        self.variable_0_gather.queue.data = extra_src['contrastive_state'].data
        self.variable_1_gather.queue.data = extra_tgt['contrastive_state'].data

        if src_lg is not None:
            self.variable_0_gather.queue_id.data = src_lg.data
            self.variable_1_gather.queue_id.data = src_lg.data


    def forward(self, model, sample, reduce=True):

        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss_dic = {}
        sample_size_dic = {}
        for task in list(MSMLossSpace):
            loss_dic[task] = torch.tensor(0.0)
            sample_size_dic[task] = 0

        def update_loss_dic(loss_enum, loss, sample_size):
            loss_dic[loss_enum] = loss
            sample_size_dic[loss_enum] = sample_size

        task = MSMTaskSpace(sample['net_input']['task'][0].item())
        current_batch_task = task
        all_task = sample['net_input']['task'].tolist()
        assert all([item == task.value for item in all_task]), str(all_task)
        if len(self.contrastive_loss_dic) > 0:
            self.variable_1_gather.clear()
            self.variable_0_gather.clear()
        if task == MSMTaskSpace.mlm or task == MSMTaskSpace.tlm:
            targets = sample['targets'][task.name]['target']
            masked_tokens = targets.ne(self.padding_idx)

            sample_size = masked_tokens.int().sum().item()
            # (Rare case) When all tokens are masked, the model results in empty
            # tensor and gives CUDA error.
            if sample_size == 0:
                masked_tokens = None
            assert sample['net_input'][task.name]['src_tokens'].shape[1] < 513    
            _, logits, extra = model(src_tokens=sample['net_input'][task.name]['src_tokens'],
                                  task=task,
                                  masked_tokens=masked_tokens,
                                  return_all_hiddens=True)
            if sample_size != 0:
                targets = targets[masked_tokens]
            loss = F.nll_loss(
                F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            if task == MSMTaskSpace.mlm:
                update_loss_dic(MSMLossSpace.mlm, loss, sample_size)
            if task == MSMTaskSpace.tlm:
                update_loss_dic(MSMLossSpace.tlm, loss, sample_size)
            # loss scale
        elif task == MSMTaskSpace.document_model:
            assert sample['net_input'][task.name]['src_tokens'].shape[1] < 513

            src_lg = sample['net_input'][task.name]['lg_idx']
            variable_index = src_lg[0].item()

            _, logits_src, extra_src = model(
                src_tokens=sample['net_input'][task.name]['src_tokens'],
                task=task,
                return_all_hiddens=True,
                variable_index=variable_index,
            )

            # gather tensor from other GPUs
            if self.args.enable_cross_gpu:
                variable0_list = self.gather_list_variable0.collect_all_gather(extra_src['doc_output'])
                variable1_list = self.gather_list_variable1.collect_all_gather(extra_src['contrastive_state'])
                local_q_vector = extra_src['doc_output']
                local_k_vector = extra_src['contrastive_state']

                global_q_vector = []
                global_k_vector = []

                for i, item in enumerate(zip(variable0_list, variable1_list)):
                    q_vector, k_vectors = item
                    if i != self.args.device_id:
                        global_q_vector.append(q_vector.to(local_q_vector.device))
                        global_k_vector.append(k_vectors.to(local_q_vector.device))
                    else:
                        global_q_vector.append(local_q_vector)
                        global_k_vector.append(local_k_vector)

                global_q_vector = global_q_vector[self.args.device_id: ] + global_q_vector[:self.args.device_id]
                global_k_vector = global_k_vector[self.args.device_id: ] + global_k_vector[:self.args.device_id]
                nega_gpu = self.args.nega_gpu
                global_q_vector = global_q_vector[:nega_gpu]
                global_k_vector = global_k_vector[:nega_gpu]
                src_lg2 = src_lg.repeat(nega_gpu)

                global_q_vector = torch.cat(global_q_vector, dim=0)
                global_k_vector = torch.cat(global_k_vector, dim=0)

                cts_mask = None
                loss_cts, sample_size_cts = self.contrastive_loss_dic[task].get_loss(
                    tensor_query=global_q_vector,
                    tensor_key=global_k_vector,
                    query_id=src_lg2,
                    key_id=src_lg2,
                    mask=cts_mask,
                    args=self.args,
                )
            else:
                cts_mask = None
                loss_cts, sample_size_cts = self.contrastive_loss_dic[task].get_loss(
                    tensor_query=extra_src['doc_output'],
                    tensor_key=extra_src['contrastive_state'],
                    query_id=src_lg,
                    key_id=src_lg,
                    mask=cts_mask,
                    args=self.args,
                )
            update_loss_dic(MSMLossSpace.doc_acc,
                            torch.tensor(self.contrastive_loss_dic[task].accuracy + 0.0),
                            sample_size_cts)

            self.update_variable(extra_src, extra_src, src_lg)
            self.eval_state = 1 - self.eval_state

            # The masked token size of monolingual mlm batch.
            # Most of sample can't reach max sequence length. The average length is 0.9 * max squence length
            sample_size = self.args.tokens_per_sample * self.args.max_sentences * self.args.mask_prob * 0.9
            cts_ratio = sample_size / (self.args.max_sentences * self.args.doc_bs_multiply) if sample_size_cts > 0 else 0
            loss = loss_cts * cts_ratio * self.args.nsp_scale

            update_loss_dic(MSMLossSpace.doc_model, loss_cts, sample_size_cts)
        else:
            raise ValueError("unknown task", task)

        if len(self.contrastive_loss_dic) > 0:
            self.variable_0_gather.task_id.data.fill_(task.value)
            self.variable_1_gather.task_id.data.fill_(task.value)
            variable_0_task2tensors = self.variable_0_gather.gather()
            variable_1_task2tensors = self.variable_1_gather.gather()

            for task_id in variable_0_task2tensors.keys():
                if task_id < 0:
                    continue
                task = MSMTaskSpace(task_id)
                if task not in self.contrastive_loss_dic:
                    continue
                contrastive_losses = self.contrastive_loss_dic[task]

                # negative queue
                if task == MSMTaskSpace.document_model:
                    contrastive_losses.queue.push(variable_1_task2tensors[task_id]['queue'],
                                                  variable_1_task2tensors[task_id]['queue_id'])
                else:
                    contrastive_losses.queue.push(variable_0_task2tensors[task_id]['queue'],
                                                  variable_0_task2tensors[task_id]['queue_id'])
                    contrastive_losses.queue.push(variable_1_task2tensors[task_id]['queue'],
                                                  variable_1_task2tensors[task_id]['queue_id'])

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        for task in MSMLossSpace:
            logging_output['loss_' + task.name] = utils.item(loss_dic[task].data)
            logging_output['sample_size_' + task.name] = sample_size_dic[task]
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        for task in MSMLossSpace:
            sample_size_task = sum(log.get('sample_size_' + task.name, 0) for log in logging_outputs)
            loss_task = sum(log.get('loss_' + task.name, 0) for log in logging_outputs) / sample_size_task / math.log(2) \
                if sample_size_task > 0 else 0.

            if 'acc' in task.name:
                out = [ log.get('loss_doc_acc') for log in logging_outputs if log.get('loss_doc_model') > 0 ]
                # print("At this time task is: {}, len: {}, mean: {}".format(task, len(out), sum(out) / len(out)))
                loss_task = sum(out) / len(out) if len(out) > 0 else 0
                loss_task *= 100

            agg_output['loss_' + task.name] = loss_task
        agg_output['loss_tsa_acc'] = 0
        return agg_output

