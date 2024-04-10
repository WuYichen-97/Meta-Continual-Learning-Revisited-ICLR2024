
from functools import reduce
import torch
from utils.buffer import Buffer
from utils.ring_buffer import RingBuffer

from utils.args import *
from models.utils.continual_model import ContinualModel
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import os
np.random.seed(0)
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--grad_clip_norm', type=float, help='learning rate', default=1.0)
    return parser

class ErOBC(ContinualModel):
    NAME = 'er_obc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErOBC, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.iter = 0

    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        self.N_TASKS = dataset.N_TASKS
        self.Total_classes = self.N_CLASSES_PER_TASK*self.N_TASKS


    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK 
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK 
        # print('offset1', offset1)
        return int(offset1), int(offset2)                  

    def take_multitask_loss(self, bt, logits, y, task_id):
        loss = 0.0
        for i, ti in enumerate(bt):
            if i < 10:
                offset1, offset2 = self.compute_offsets(ti)
                offset2 = 10
            else:
                _, offset2 = self.compute_offsets(task_id)
                # task simiarity exp2
                offset1 = 0
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)


    def observe(self, inputs, labels, not_aug_inputs, task_id):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        self.opt.zero_grad()
        outputs = self.net(inputs, mode = 'surrogage')
        loss = self.loss(outputs, labels)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            loss += self.loss(self.net(buf_inputs), buf_labels)
        else:
            buf_id = task_id*torch.ones_like(labels)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        # update the balance classifier
        self.opt1.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                50, transform=self.transform)           
            outputs = self.net(buf_inputs)
            loss = self.loss(outputs,buf_labels)
            loss.backward()
            self.opt1.step()
            self.opt1.zero_grad()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             task_labels=task_id*torch.ones_like(labels))

        return loss.item()


