# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from backbone import MammothBackbone, xavier, num_flat_features
from collections import OrderedDict
import torch.nn.functional as F


def functional_conv_block(x, weights, bias, is_training, stride=2, padding=1):
        x = F.conv2d(x, weights, bias=bias, padding=padding,stride=stride)
        return x

class PC_CNN_base(MammothBackbone):
    def __init__(self, input_size: int, output_size:int)-> None:
        super(PC_CNN_base, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.channels = 160
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size = 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.conv3 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.linear1 = nn.Linear(16*self.channels,320)
        self.linear2 = nn.Linear(320,320)
        
        nn.init.zeros_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)


        self._features = nn.Sequential(
             self.conv1,
             nn.ReLU(),
             self.conv2,
             nn.ReLU(), 
             self.conv3,
             nn.ReLU(),
             nn.Flatten(),
             self.linear1,
             nn.ReLU(),
             self.linear2,
             nn.ReLU(),
        )
        self.classifier = nn.Linear(320, self.output_size)
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)


    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        feats = self._features(x)

        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)
        raise NotImplementedError("Unknown return type")
    def functional_forward(self,x:torch.Tensor,fast_weight:OrderedDict,returnt:str ='out')-> torch.Tensor:
        x = F.relu(functional_conv_block(x, weights=fast_weight['conv1.weight'], bias = fast_weight['conv1.bias'], is_training=True))
        x = F.relu(functional_conv_block(x, weights=fast_weight['conv2.weight'], bias = fast_weight['conv2.bias'],is_training=True))
        x = F.relu(functional_conv_block(x, weights=fast_weight['conv3.weight'], bias = fast_weight['conv3.bias'],is_training=True))
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, fast_weight['linear1.weight'], fast_weight['linear1.bias']))
        feats = F.relu(F.linear(x, fast_weight['linear2.weight'], fast_weight['linear2.bias']))
        if returnt == 'features':
            return feats
        out = F.linear(feats, fast_weight['classifier.weight'], fast_weight['classifier.bias'])
        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)
        raise NotImplementedError("Unknown return type")

    def get_fast_weight(self) -> OrderedDict:
        # print('self.named', self.named_parameters())
        return self.named_parameters()

class PC_CNN(MammothBackbone):
    # def __init__(self, eman= False, momentum=0.99, input_size, output_size):
    def __init__(self, input_size: int, output_size:int, eman =False, momentum=0)-> None:
        super(PC_CNN, self).__init__()
        self.eman = eman
        self.momentum = momentum
        self.main = PC_CNN_base(input_size, output_size)

        # build ema model
        if eman:
            print("using EMAN as teacher model")
            self.ema = PC_CNN_base(input_size, output_size)
            for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_main.data)
                # param_ema.requires_grad = False
        else:
            self.ema = None

    def momentum_update_ema(self):
        state_dict_main = self.main.state_dict()
        state_dict_ema = self.ema.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            # v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                # v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)
                v_ema.copy_(v_main)

    def reset_ema(self):
            for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_main.data)

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.main.net.apply(xavier)

    def forward(self, x: torch.Tensor, returnt='out', mode ='main') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        if mode == 'main':
            if returnt == 'features':
                return self.main.forward(x, returnt='features')
            elif returnt == 'out':
                return self.main.forward(x, returnt='out')
            elif returnt == 'all':
                return self.main.forward(x, returnt='all')
            else:
                raise NotImplementedError("Unknown return type")

        elif mode == 'ema':
            if returnt == 'features':
                return self.ema.forward(x, returnt='features')
            elif returnt == 'out':
                return self.ema.forward(x, returnt='out')
            elif returnt == 'all':
                return self.ema.forward(x, returnt='all')
            else:
                raise NotImplementedError("Unknown return type")

    def functional_forward(self,x:torch.Tensor,fast_weight:OrderedDict,returnt:str ='out', mode='main')-> torch.Tensor:
        # print('fast_weight', fast_weight)
        if mode == 'main':
            if returnt == 'features':
                return self.main.functional_forward(x, fast_weight, returnt = 'features')
            elif returnt == 'out':
                return self.main.functional_forward(x, fast_weight, returnt = 'out')
            elif returnt == 'all':
                return self.main.functional_forward(x,fast_weight, returnt = 'all')
            raise NotImplementedError("Unknown return type")

        elif mode == 'ema':
            if returnt == 'features':
                return self.ema.functional_forward(x, fast_weight, returnt = 'features')
            elif returnt == 'out':
                return self.ema.functional_forward(x, fast_weight, returnt = 'out')
            elif returnt == 'all':
                return self.ema.functional_forward(x,fast_weight, returnt = 'all')
            raise NotImplementedError("Unknown return type")

    def get_fast_weight(self, mode='main') -> OrderedDict:
        if mode == 'main':
            param = self.main.get_fast_weight()
            return OrderedDict([[p[0], p[1].clone()] for p in param])
        elif mode == 'ema':
            state_dict_main = self.main.state_dict()
            state_dict_ema = self.ema.state_dict()
            for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
                assert k_main == k_ema, "state_dict names are different!"
                assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
                if 'num_batches_tracked' in k_ema:
                    v_ema.copy_(v_main)
                else:
                    v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)
            param = self.ema.get_fast_weight()
            return OrderedDict([[p[0], p[1].clone()] for p in param])


