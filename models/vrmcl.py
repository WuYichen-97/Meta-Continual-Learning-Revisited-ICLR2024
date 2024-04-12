import math
import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from datasets import NAMES as DATASET_NAMES
# from utils.args import *
from argparse import ArgumentParser
from models import get_all_models
from models.utils.continual_model import ContinualModel
# from src.utils.training import mask_classes_in_k
from collections import OrderedDict
import numpy as np
from torch.distributions.beta import Beta
from copy import deepcopy
np.random.seed(0)
def get_parser():
    parser = ArgumentParser(description='Continual learning via La-MAML')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--seed', type=int, required=True, help='seed')

    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate.')

    parser.add_argument('--s_momentum', type=float, required=True, default=0.8, help='momentum_ratio of STORM')
    parser.add_argument('--s_lr', type=float, required=True, default = 0.1, help='learning rate of STORM')
    


    #parser.add_argument('--optim_wd', type=float, default=0., help='optimizer weight decay.')
    #parser.add_argument('--optim_mom', type=float, default=0., help='optimizer momentum.')
    #parser.add_argument('--optim_nesterov', type=int, default=0, help='optimizer nesterov momentum.')

    # train
    parser.add_argument('--n_epochs', type=int, help='Batch size.')
    parser.add_argument('--batch_size', type=int, help='Batch size.')
    parser.add_argument('--replay_batch_size', type=int, help='Batch size.')
    parser.add_argument('--buffer_size', type=int, required=True, help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, help='The batch size of the memory buffer.')

    # optimizer
    parser.add_argument('--optim_wd', type=float, default= 0, help='Weight_decay')
    parser.add_argument('--optim_mom', type=float, default=0.9, help='Weight_decay') # for cifar100 dataset

    # parser.add_argument('--optim_lr', type=float, help='learning rate')
    parser.add_argument('--grad_clip_norm', type=float, help='learning rate', default=2.0)
    #parser.add_argument('beta1',type=float)#, default = 0.9)
    #parser.add_argument('beta2',type=float)#, default = 0.999)

    # meta
    parser.add_argument('--alpha_initial', type=float, help='inner_loop learning rate', default = 0.15)
    parser.add_argument('--second_order', default=False, action='store_true')
    parser.add_argument('--asyn_update', default=False, action='store_true')
    parser.add_argument('--inner_batch_size', type=int, help='inner loop update using minibatch', default = 1)
    parser.add_argument('--meta_update_per_batch', type=int, default = 5)

    # other
    parser.add_argument('--csv_log', action='store_true', help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true', help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true', help='Test on the validation set')
    return parser



class VRMCL(ContinualModel):
    NAME = 'vrmcl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(VRMCL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task_lr = OrderedDict(
            [ (name+'lr',torch.nn.Parameter(
            self.args.alpha_initial * torch.ones_like(p, requires_grad=True, device=self.device))) for name,p in
                        self.net.named_parameters()]
        )
        self.opt_lr = torch.optim.SGD(self.task_lr.values(), lr=self.args.lr)
        self.meta_loss = None
        self.meta_loss_last_step = None
        self.fast_weight = None
        self.observed_batch = None
        self.max_batch_in_minitask = None
        # self.momentum = [] #None
        self.momentum = [[None]]*63
        self.g1 = []
        self.fast_weight_last_step = None
        self.tag = False
        self.task_id = -1
        self.total = -1
        self.accumu_grad = [[None]]*12
        self.grad = []#[None]*12
        self.grad1 = []
        self.var = []
        self.var1 = []
        self.lgrad = [[None]]*12
        self.outer_loss = []
        self.distance = []
        self.t = 0
        self.loss_mean =0
        self.save = 0

    def begin_task(self, dataset):
        # clear meta-loss
        self.outer_loss = []
        self.meta_loss, self.meta_loss1 = [],[]
        self.meta_loss_last_step = []
        # clone the model
        self.fast_weight = self.net.get_fast_weight()
        self.fast_weight_last_step = self.net.get_fast_weight(mode='ema')
        # zero grad
        self.opt.zero_grad()
        self.opt_lr.zero_grad()
        self.observed_batch = 0
        self.max_batch_in_minitask = len(dataset.train_loader)
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        self.iter = 0
        self.task_id = 0

    def end_task(self, dataset):
        task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # torch.save(self.outer_loss, 'loss'+str(int(task_id))+'.pth') 


    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK
        return int(offset1), int(offset2)       

    def take_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y-offset1)
        return loss
    
    
    def take_multitask_loss(self, bt, logits, y, task_id):
        loss = 0.0
        for i, ti in enumerate(bt):
            if i < 32:
                offset1, offset2 = self.compute_offsets(ti)
                offset2 = 200
                loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
            else:
                _, offset2 = self.compute_offsets(task_id)
                offset1 = 0
                loss += 0.4*self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)

    def meta_loss_compute(self, x, y, bt, t, j,y_logits=None):
        """
        differentiate the loss through the network updates wrt alpha
        """
        if j ==0:
            _ = self.net(x) # update the BN part
        offset1, offset2 = self.compute_offsets(t)
        logits = self.net.functional_forward(x, fast_weight=self.fast_weight)#[:,:offset2]
        loss_q = self.take_multitask_loss(bt, logits, y, t)            
        return loss_q    

    def meta_loss_compute_last_step(self, x, y, bt, t, j,y_logits=None):
        """
        differentiate the loss through the network updates wrt alpha
        """
        if j ==0:
            _ = self.net(x) # update the BN part
        offset1, offset2 = self.compute_offsets(t)
        logits = self.net.functional_forward(x, fast_weight=self.fast_weight_last_step, mode='ema')#[:,:offset2]
        loss_q = self.take_multitask_loss(bt, logits, y, t)            
        return loss_q   

    def inner_loop(self, x, y, t):       
        logits = self.net.functional_forward(x, fast_weight=self.fast_weight)
        loss = self.take_loss(t,logits,y)
        grad = list(torch.autograd.grad(loss, self.fast_weight.values(), create_graph = self.args.second_order, retain_graph=self.args.second_order))
        if self.args.grad_clip_norm:
            for i in range(len(grad)):
                grad[i] = torch.clamp(grad[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)
        self.fast_weight = OrderedDict(
           (name, p - lr * g)  for (name, p), g, lr in zip(self.fast_weight.items(), grad, self.task_lr.values())
        )  
        _, pred = torch.max(logits.data, 1)
        correct = torch.eq(pred, y).sum()
        return correct, logits

    def inner_loop_last_step(self, x, y, t):       
        logits = self.net.functional_forward(x, fast_weight=self.fast_weight_last_step, mode='ema')
        loss = self.take_loss(t,logits,y)
        grad = list(torch.autograd.grad(loss, self.fast_weight_last_step.values(), create_graph = self.args.second_order, retain_graph=self.args.second_order))
        if self.args.grad_clip_norm:
            for i in range(len(grad)):
                grad[i] = torch.clamp(grad[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)
        self.fast_weight_last_step = OrderedDict(
           (name, p - lr * g)  for (name, p), g, lr in zip(self.fast_weight_last_step.items(), grad, self.task_lr.values())
        )  
        _, pred = torch.max(logits.data, 1)
        correct = torch.eq(pred, y).sum()
        return correct, logits

    def outer_loop(self, task_id):           
        # update the parameters of backbone
        meta_loss = sum(self.meta_loss) / len(self.meta_loss)
        self.outer_loss.append(meta_loss)
        meta_loss.backward()
        if self.args.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.net.main.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.task_lr.values(), self.args.grad_clip_norm)
        

        meta_loss_last_step = sum(self.meta_loss_last_step) / len(self.meta_loss_last_step)
        meta_loss_last_step.backward()
        if self.args.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.net.ema.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.task_lr.values(), self.args.grad_clip_norm)
        # update the task_lr
        self.fast_weight_last_step = self.net.get_fast_weight('ema')
        self.opt_lr.step()

        # STORM
        if self.args.asyn_update:  
            cnt = 0
            with torch.no_grad():
                        
                # vr-mcl
                for g, p, lr in zip(self.net.ema.parameters(), self.net.main.parameters(), self.task_lr.values()):
                        if self.momentum[cnt][0] == None:
                            p.data = p.data - (p.grad) * self.args.s_lr
                            self.momentum[cnt] = [p.grad]
                        else:
                            self.momentum[cnt] = [p.grad + self.args.s_momentum*(self.momentum[cnt][0]-g.grad)]
                            p.data = p.data - (self.momentum[cnt][0]) * self.args.s_lr
                        cnt += 1

        else:
            self.opt.step()
        # zero grad and lossl  
        self.opt.zero_grad()
        self.opt_lr.zero_grad()
        self.fast_weight = self.net.get_fast_weight()
        self.meta_loss, self.meta_loss1 = [], []
        self.meta_loss_last_step = []
        self.iter+=1 
        return meta_loss


    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs, task_id):
        correct = 0
        inputs = torch.tensor(inputs, dtype=torch.float)
        real_batch_size = inputs.shape[0]
        num_inner_steps = math.ceil(real_batch_size / self.args.inner_batch_size)
        logits_save = None
        for i in range(self.args.meta_update_per_batch):
            perm = torch.randperm(real_batch_size)
            x_s, y_s = inputs[perm], torch.tensor(labels[perm], dtype=torch.long)
            if self.buffer.is_empty():
                x_q, y_q = inputs, torch.tensor(labels, dtype=torch.long)
                buf_id = task_id*torch.ones_like(y_q)
            else:
                buf_inputs, buf_labels, buf_logits, buf_id = self.buffer.get_data(
                    self.args.replay_batch_size, transform=self.transform)
                x_q = torch.cat((inputs, buf_inputs))
                y_q = torch.cat((torch.tensor(labels,dtype=torch.long), buf_labels))
                buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
                # one step adaptation
            for j in range(num_inner_steps):
                x = x_s[j * self.args.inner_batch_size:(j + 1) * self.args.inner_batch_size].detach()
                y = y_s[j * self.args.inner_batch_size:(j + 1) * self.args.inner_batch_size].detach()
                correct_inner, logits = self.inner_loop(x, y, task_id)
                _, _ = self.inner_loop_last_step(x, y,task_id)
                
                if logits_save == None:
                    logits_save = logits
                else:
                    logits_save = torch.cat((logits_save,logits),dim=0)
                self.meta_loss.append(self.meta_loss_compute(x_q, y_q, buf_id, task_id, j))
                self.meta_loss_last_step.append(self.meta_loss_compute_last_step(x_q, y_q, buf_id, task_id,j ))

                if not j and not i:
                    correct += correct_inner
            # meta_update
            meta_loss = self.outer_loop(task_id)

        self.observed_batch += 1
        if self.observed_batch <= self.max_batch_in_minitask:
            self.buffer.add_data(examples=not_aug_inputs,
                        labels=labels, logits=logits_save.detach(), task_labels=task_id*torch.ones_like(labels))

        return meta_loss.item()


