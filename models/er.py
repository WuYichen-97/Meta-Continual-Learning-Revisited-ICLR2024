
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
np.random.seed(0)
import os

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)


    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
   

 
    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 

 

    def observe(self, inputs, labels, not_aug_inputs, task_id):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        self.opt.zero_grad()
        outputs = self.net(inputs)
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

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             task_labels=task_id*torch.ones_like(labels))

        return loss.item()


