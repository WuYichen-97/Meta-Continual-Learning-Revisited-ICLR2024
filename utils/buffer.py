# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from copy import deepcopy
import random

def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """
        
    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)
        
        data_concatenate = torch.cat if type(dataset.train_loader.dataset.data) == torch.Tensor else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            refold_transform = lambda x: x.cpu()
        else:    
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                refold_transform = lambda x: (x.cpu()*255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                refold_transform = lambda x: (x.cpu()*255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
            ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
            ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
                ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
                ])

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                # print('attr',attr)
                if attr_str != 'task_labels':
                    typ = torch.int64 if attr_str.endswith('els') else torch.float32
                    setattr(self, attr_str, torch.zeros((self.buffer_size,
                            *attr.shape[1:]), dtype=typ, device=self.device))
                if attr_str == 'task_labels':
                    typ = torch.int64 if attr_str.endswith('els') else torch.float32
                    setattr(self, attr_str, torch.zeros((self.buffer_size), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network  (这个部分存入的之前模型的logits,用于计算distillation loss)
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)
            # print('examples', self.examples.shape)
            # print('labels', self.labels.shape)
            # print('logits', self.logits.shape)
            # print('task_labels', self.task_labels.shape)

        # wrong
        # for i in range(examples.shape[0]):
        #     index = random.randint(self.labels[i]*100,self.labels[i]*100+100 )
        #     # print('index',index)
        #     self.examples[index] = examples[i].to(self.device)
        #     if labels is not None:
        #         self.labels[index] = labels[i].to(self.device)
        #         # print('labels1',labels)
        #     if logits is not None:
        #         self.logits[index] = logits[i].to(self.device)
        #     if task_labels is not None:
        #         # self.task_labels[index] = task_labels[i].to(self.device)
        #         self.task_labels[index] = task_labels[i].to(self.device)            

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            # if buffer is not full, then push the examples into the buffer; else random choose one position to replace the element
            # (这里可以看到exemplar加入buffer的情况, 是随机的, 并不是每个类维护的很好的那种memory buffer)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                    # print('labels1',labels)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    # self.task_labels[index] = task_labels[i].to(self.device)
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        # dele_index = []
        # if self.num_seen_examples>1000:
        #     _, counts = torch.unique(self.labels, return_counts=True) 
        #     for index, cc in enumerate(choice):
        #         ll = self.labels[cc]
        #         lcount = counts[ll]
        #         exp_prob = lcount/100
        #         prob = random.random()
        #         if prob < exp_prob and prob<0.15:
        #             dele_index.append(index)
        # np.delete(choice, dele_index)

        # print('choice',choice)
        # print('num',min(self.num_seen_examples, self.examples.shape[0]))
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        
        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple
        # return ret_tuple


    # def get_all_saved_data(self, sample_size, transform: transforms=None) -> Tuple:
    def get_all_saved_data(self, exclude_task,transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        size = min(self.num_seen_examples, self.examples.shape[0])
        # if sample_size > size:
        #     sample_size = size
        # choice = np.random.choice(size, size, replace=False)
        # choice = torch.arange(size)
        choice = self.labels

        offset1 = exclude_task*2
        offset2 = offset1 + 1
        # print('offset', offset1, offset2)
        # print('choice0', choice)
        choice1 = choice != offset1
        choice2 = choice != offset2
        # print('choice1', choice1)
        # print('choice2', choice2)
        choice =  (choice1 & choice2).nonzero().squeeze()
        # print('!!!!!!!!!!!!!!!!')
        # print('choice',choice)
        # choice = np.random.choice(size, sample_size, replace=False)

        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        return ret_tuple




    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple


    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False





    # def get_MIR_data(self,transform,fastweights):
    #     if transform is None: transform = lambda x: x
    #     ret_tuple = (torch.stack([transform(ee.cpu())
    #                         for ee in self.examples[indexes]]).to(self.device),)
    #     for attr_str in self.attributes[1:]:
    #         if hasattr(self, attr_str):
    #             attr = getattr(self, attr_str).to(self.device)
    #             ret_tuple += (attr[indexes],)
    #     return ret_tuple






    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
