# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from random import seed
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from backbone.ResNet18_MAML import resnet18_maml
from backbone.ResNet18 import resnet18
from backbone.Pretrained_ResNet18 import ResNet18
import PIL

from backbone.pc_cnn import PC_CNN
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
import numpy as np
import os
from torchvision.datasets.utils import download_url, check_integrity
import sys
import pickle
import copy
import torchvision.transforms as transforms

import torchvision.models as models
import torch.nn as nn

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)
class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        return img, target, not_aug_img


def uniform_corruption(corruption_ratio, num_classes):
    eye = np.eye(num_classes)
    noise = np.full((num_classes, num_classes), 1/num_classes)
    corruption_matrix = eye * (1 - corruption_ratio) + noise * corruption_ratio
    return corruption_matrix


def flip1_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix


def flip2_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_ratio / 2
    return corruption_matrix

class My_Noise_CIFAR10():
    def __init__(self, seed, dataset='cifar10', imbalanced_factor=None, Reverse=False, corruption_type=None, corruption_ratio=0.) -> None:
        self.train_dataset, _= self.build_dataloader(seed, dataset, imbalanced_factor, Reverse ,corruption_type, corruption_ratio)

        self.data = self.train_dataset.data#.reshape((50000, 3, 32, 32))
        # self.data = self.data.transpose((0,2,3,1))
        # print('self.data',self.data[0])
        # print('shape', self.data.shape)
        self.targets = self.train_dataset.targets
        self.train_transforms = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2615))])

        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def build_dataloader(self, seed=1, dataset='cifar10', imbalanced_factor=None, Reverse=False, corruption_type=None, corruption_ratio=0.,):
        np.random.seed(seed)
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615)),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615)),
        ])
        dataset_list = {
            'cifar10': torchvision.datasets.CIFAR10,
            'cifar100': torchvision.datasets.CIFAR100,
        }
        corruption_list = {
            'uniform': uniform_corruption,
            'flip1': flip1_corruption,
            'flip2': flip2_corruption,
        }

        train_dataset = dataset_list[dataset](root='../data', train=True, download=True, transform=train_transforms)
        test_dataset = dataset_list[dataset](root='../data', train=False, transform=test_transforms)

        num_classes = len(train_dataset.classes)

        # index_to_meta = []
        index_to_train = []
        num_meta_total = 0

        if imbalanced_factor is not None:
            imbalanced_num_list = []
            sample_num = int((len(train_dataset.targets) - num_meta_total) / num_classes)
            for class_index in range(num_classes):
                imbalanced_num = sample_num / (imbalanced_factor ** (class_index / (num_classes - 1)))
                imbalanced_num_list.append(int(imbalanced_num))
            # np.random.shuffle(imbalanced_num_list)  
            if Reverse:
                imbalanced_num_list.reverse()
                print(imbalanced_num_list)
            else:
                print(imbalanced_num_list)
                print('imbalance_factor', imbalanced_factor)
        else:
            imbalanced_num_list = None

        for class_index in range(num_classes):
            index_to_class = [index for index, label in enumerate(train_dataset.targets) if label == class_index]
            np.random.shuffle(index_to_class)
            index_to_class_for_train = index_to_class

            if imbalanced_num_list is not None:
                index_to_class_for_train = index_to_class_for_train[:imbalanced_num_list[class_index]]
            index_to_train.extend(index_to_class_for_train)

        train_dataset.data = train_dataset.data[index_to_train]
        train_dataset.targets = list(np.array(train_dataset.targets)[index_to_train])

        if corruption_type is not None:
            corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
            print(corruption_matrix)
            for index in range(len(train_dataset.targets)):
                p = corruption_matrix[train_dataset.targets[index]]
                train_dataset.targets[index] = np.random.choice(num_classes, p=p)     
        return train_dataset, test_dataset   

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # to return a PIL Image
        # not_aug_img = img
        img = Image.fromarray(img, mode='RGB')
        # original_img = img.copy()
        not_aug_img = self.not_aug_transform(img)

        # if self.train_transforms is not None:
        img = self.train_transforms(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        return img, target, not_aug_img


class SequentialCIFAR10(ContinualDataset):
    # 继承自ContinualDataset, 因此有 self.i = 0
    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    task_id = 0
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))])


    def get_data_loaders(self, tag=False):
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)


        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=test_transform)
        train, test = store_masked_loaders(train_dataset, test_dataset, self, tag)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(eman1):
        return resnet18_maml(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS, eman=eman1) #PC_CNN
    # def get_backbone():
    #     return PC_CNN(3*32*32,SequentialCIFAR10.N_CLASSES_PER_TASK * SequentialCIFAR10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy
        # return nn.CrossEntropyLoss(reduction='none')
    
    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None
