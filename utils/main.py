# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy # needed (don't change it)
import importlib
import os
import sys
import socket
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train#, evaluate0
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime
import wandb

import time
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args(): #
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    # choose the model want to run
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models()) # duplicate introduce in add_experiment_args 注释掉了args里面的model
    torch.set_num_threads(4)
    # add_management_args(parser)
    args = parser.parse_known_args()[0]  # parser.parse_args()[0] (运行报错) 这样在sh文件里面加入其他命令行的时候, 这里还没添加，会报错. 所以必须用parse_known_args()
    mod = importlib.import_module('models.' + args.model)
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    # other configurations are shown in args.py
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args

def main(args=None):
    # lecun_fix()
    # print('arg',args.model)
    if args is None:
        args = parse_args()
    dataset = get_dataset(args)
    if args.model =='vrmcl':
        backbone = dataset.get_backbone(eman1=True) 
    else:
        backbone = dataset.get_backbone() 
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    # print parameters
    print("Total trainable parameters: ", count_parameters(model))

    # wandb.watch(model,log='all')
    if isinstance(dataset, ContinualDataset):
        start_time = time.time()
        train(model, dataset, args)
        end_time = time.time()
        total_time = end_time - start_time
        print('running time is: {:.2f}s'.format(total_time)) 
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


if __name__ == '__main__':
    main()
