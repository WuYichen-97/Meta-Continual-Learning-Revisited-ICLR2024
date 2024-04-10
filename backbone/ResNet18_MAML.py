# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
from backbone import MammothBackbone
from collections import OrderedDict


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def functional_conv_block(x, weights, bn_weights, bn_biases, is_training, stride=1, padding=1, momentum=0.1):
        x = F.conv2d(x, weights, bias=None, padding=padding,stride=stride)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training, momentum = momentum)
        return x

class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

    def functional_forward(self, x, fast_weight, layer_num, momentum): # 因为restnet18的basic_block是[2,2,2,2]
        out =F.relu(functional_conv_block(x,weights=fast_weight[layer_num+'conv1.weight'], bn_weights=fast_weight[layer_num+'bn1.weight'],
                                   bn_biases=fast_weight[layer_num+'bn1.bias'], is_training=True, stride=self.stride, momentum=momentum))
        out = functional_conv_block(out,weights=fast_weight[layer_num+'conv2.weight'], bn_weights=fast_weight[layer_num+'bn2.weight'],
                                   bn_biases=fast_weight[layer_num+'bn2.bias'],is_training=True, momentum=momentum)
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            out += functional_conv_block(x,weights=fast_weight[layer_num+'shortcut.0.weight'], bn_weights=fast_weight[layer_num+'shortcut.1.weight'],
                                   bn_biases=fast_weight[layer_num+'shortcut.1.bias'], is_training=True,stride=self.stride,padding=0, momentum=momentum)

        else:
            out += x
        out = F.relu(out)
        return out



class ResNet_Base(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """
    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet_Base, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """
        out = relu(self.bn1(self.conv1(x)))  # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4
        out = avg_pool2d(out, out.shape[2])  # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512
        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feature)

        raise NotImplementedError("Unknown return type")

    def functional_forward(self, x: torch.Tensor, fast_weight:OrderedDict, returnt='logits', momentum=0.1) -> torch.Tensor:

        out = F.relu(functional_conv_block(x,weights=fast_weight['conv1.weight'], bn_weights=fast_weight['bn1.weight'],
                                   bn_biases=fast_weight['bn1.bias'], is_training=True), momentum) # corresponding to line 147
        # if hasattr(self, 'maxpool'):
        #     raise Exception('functinol maxpooling layer not yet implemented')

        for i, block in enumerate(self.layer1.children()): # corresponding to line 150
            out = block.functional_forward(out, fast_weight,f'layer1.{i}.', momentum)
        for i, block in enumerate(self.layer2.children()): # corresponding to line 151
            out = block.functional_forward(out, fast_weight,f'layer2.{i}.', momentum)
        for i, block in enumerate(self.layer3.children()): # corresponding to line 152
            out = block.functional_forward(out, fast_weight,f'layer3.{i}.', momentum)
        for i, block in enumerate(self.layer4.children()): # corresponding to line 153
            out = block.functional_forward(out, fast_weight,f'layer4.{i}.', momentum)
        out =F.avg_pool2d(out,out.shape[-2])
        feature = out.view(out.size(0), -1)

        if returnt == 'features':
            return feature

        out = F.linear(feature,weight=fast_weight['linear.weight'],bias=fast_weight['linear.bias'])

        if returnt == 'logits':
            return out
        elif returnt == 'all':
            return (out, feature)
        raise NotImplementedError("Unkonwn output!")

    def get_fast_weight(self) -> OrderedDict:
        # return OrderedDict([[p[0],p[1].clone()] for p in self.named_parameters()])
        return self.named_parameters()



class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """
    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, eman=True) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.main = ResNet_Base(BasicBlock, [2, 2, 2, 2], num_classes, nf)
        if eman:
            print("using EMAN as teacher model")
            self.ema = ResNet_Base(BasicBlock, [2, 2, 2, 2], num_classes, nf)
            for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_main.data)
                # param_ema.requires_grad = False
        else:
            self.ema = None
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

    def functional_forward(self,x:torch.Tensor,fast_weight:OrderedDict,returnt:str ='logits', mode='main')-> torch.Tensor:
        if mode == 'main':
            if returnt == 'features':
                return self.main.functional_forward(x, fast_weight, returnt = 'features')
            elif returnt == 'logits':
                return self.main.functional_forward(x, fast_weight, returnt = 'logits')
            elif returnt == 'all':
                return self.main.functional_forward(x,fast_weight, returnt = 'all')
            raise NotImplementedError("Unknown return type")

        elif mode == 'ema':
            if returnt == 'features':
                return self.ema.functional_forward(x, fast_weight, returnt = 'features')
            elif returnt == 'logits':
                return self.ema.functional_forward(x, fast_weight, returnt = 'logits')
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
                v_ema.copy_(v_main)
            param = self.ema.get_fast_weight()
            return OrderedDict([[p[0], p[1].clone()] for p in param])


def resnet18_maml(nclasses: int, nf: int = 20, eman=False) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, eman=eman)
