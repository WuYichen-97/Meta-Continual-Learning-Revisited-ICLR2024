import torchvision.models as models
import torch

class StandardResNet(torch.nn.Module):
    def __init__(self, n_classes, pretrained=False, rn=18, dim_in=512, ckpt_path=None):
        super(StandardResNet, self).__init__()
        if rn == 18:
            classifier = models.resnet18(pretrained=False)
            classifier.load_state_dict(torch.load('resnet18-pretrained.pth'))
        elif rn == 50:
            classifier = models.resnet50(pretrained=pretrained)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)['state_dict']
            classifier.load_state_dict(ckpt)

        self.encoder = torch.nn.Sequential(*list(classifier.children())[:-1])
        dim_in = classifier.fc.in_features
        self.linear = torch.nn.Linear(dim_in, n_classes)

    def features(self, x):
        '''Features before FC layers'''
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits

def ResNet18(n_classes, pretrained=True, rn=18, dim_in=512, ckpt_path=None):
    classifier = StandardResNet(n_classes, pretrained=pretrained, rn=rn, dim_in=dim_in, ckpt_path=ckpt_path)
    return classifier