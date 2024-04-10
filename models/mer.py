# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via Meta-Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # it worth to denote this operation
    # remove batch_size from parser
    for i in range(len(parser._actions)):
        if parser._actions[i].dest == 'batch_size':
            del parser._actions[i]
            break

    parser.add_argument('--beta', type=float, required=True,
                        help='Within-batch update beta parameter.')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Across-batch update gamma parameter.')
    parser.add_argument('--batch_num', type=int, required=True,
                        help='Number of batches extracted from the buffer.')

    return parser


class Mer(ContinualModel):
    NAME = 'mer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Mer, self).__init__(backbone, loss, args, transform) # continual_model.py
        self.buffer = Buffer(self.args.buffer_size, self.device)
        assert args.batch_size == 1, 'Mer only works with batch_size=1' 

    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK


    # Cifar10/100 version
    def draw_batches(self, inp, lab ,t):
        batches = []
        # for i in range(self.args.batch_num):
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((buf_inputs, inp))
            labels = torch.cat((buf_labels, torch.tensor([lab]).to(self.device)))
            task_id = torch.cat((buf_id, torch.tensor([t]).to(self.device)))
            batches.append((inputs.unsqueeze(0), labels.unsqueeze(0), task_id.unsqueeze(0)))
        else:
            batches.append((inp.unsqueeze(0), torch.tensor([lab]).unsqueeze(0).to(self.device), torch.tensor([lab]).unsqueeze(0).to(self.device)))
        return batches


    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK
        return int(offset1), int(offset2)    

    def take_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_offsets(t)
        offset1 = 0
        loss = self.loss(logits[:, offset1:offset2], y-offset1)
        return loss

    def observe(self, inputs, labels, not_aug_inputs, t):
        theta_A0 = self.net.get_params().data.clone()
        for i in range(self.args.batch_num):          
            theta_Wi0 = self.net.get_params().data.clone()
            batches = self.draw_batches(inputs, labels, t) 
            batch_inputs, batch_labels, batch_id = batches[i]
            batch_inputs = batch_inputs.squeeze(0)
            batch_labels = batch_labels.squeeze(0)
            batch_id = batch_id.squeeze(0)
            loss = 0.0
            for idx in range(len(batch_inputs)):
                self.opt.zero_grad()
                bx = batch_inputs[idx]
                if len(bx.shape) == 3:
                    bx = bx.unsqueeze(0)
                by = batch_labels[idx].unsqueeze(0).long()
                bt = batch_id[idx]
                prediction = self.net(bx)
                loss = self.loss(prediction, by)
                loss.backward()
                self.opt.step()
            # within batch reptile meta-update
            new_params = theta_Wi0 + self.args.beta * (self.net.get_params() - theta_Wi0)
            self.net.set_params(new_params)
        # across batch reptile meta-update
        new_new_params = theta_A0 + self.args.gamma * (self.net.get_params() - theta_A0)
        self.net.set_params(new_new_params)
        self.buffer.add_data(examples=not_aug_inputs, labels=labels, task_labels=t*torch.ones_like(labels))
        return loss.item()