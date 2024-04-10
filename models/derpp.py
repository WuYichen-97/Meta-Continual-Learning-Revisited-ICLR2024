# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import numpy as np
np.random.seed(0)
import torch.nn as nn
import os


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser

def label_smooth(label, n_class=3, alpha=0.1):
    '''
    label: true label
    n_class: # of class
    alpha: smooth factor
    '''
    k = alpha / (n_class - 1)
    temp = torch.full((label.shape[0], n_class), k)
    temp = temp.scatter_(1, label.unsqueeze(1), (1-alpha))
    return temp

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.loss = nn.KLDivLoss()

    def forward(self, pred, target):
        pred = pred.log_softmax(dim = self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing/(self.cls-1))
            true_dist.scatter_(1, target.data.unsqueeze(1),self.confidence)
        loss = torch.mean(torch.sum(-true_dist*pred, dim=self.dim))
        loss = self.loss(pred, true_dist)
        return loss


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.label_smooth_loss = LabelSmoothingLoss(classes=10, smoothing=0.05)


    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK


    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 




    def observe(self, inputs, labels, not_aug_inputs,t, test=False):
        # M = 10
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs,labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits, buf_task = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)


            '——— ++ part: Add the label supervision to avoid the logits bias when task index suddenly change. (In this situation, ' \
            'the logits are more incline to the previous class)'
            buf_inputs, buf_labels, buf_logits, buf_task  = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data,
                             task_labels=t*torch.ones_like(labels))
        return loss.item()
