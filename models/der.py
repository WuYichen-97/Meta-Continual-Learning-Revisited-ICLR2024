from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *   # introduce the parameters
from models.utils.continual_model import ContinualModel
import numpy as np
import torch

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    # args three types parameters
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, 
                        help='Penalty weight.')
    return parser

class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)


    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK

    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        task_id = dataset.i//dataset.N_CLASSES_PER_TASK 


    def observe(self, inputs, labels, not_aug_inputs, t):
        # M = 10
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty(): 
            buf_inputs, buf_logits = self.buffer.get_data(  
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)  # distillation loss according to the DER paper

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()

