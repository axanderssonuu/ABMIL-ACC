import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
from miltrick.batchnorm import BatchNorm2dIdentity


class QMNISTFeatureModel(ResNet):
    def __init__(self, without_batchnorm=False, *args, **kwargs):
        if without_batchnorm:
            norm_layers = BatchNorm2dIdentity
        else:
            norm_layers = None

        super(QMNISTFeatureModel, self).__init__(
            Bottleneck, [2, 2, 2, 2], num_classes=2, norm_layer=norm_layers, *args, **kwargs)  
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1
        )
        
        self.seq2 = nn.Sequential(
            self.layer2
        )

        self.seq3 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool
        )

    def forward(self, x):
        x = self.seq2(self.seq1(x))
        x = self.seq3(x)
        return x.view(x.size(0), -1)


class QMNISTAttentionModel(torch.nn.Module):
    def __init__(self):
        super(QMNISTAttentionModel, self).__init__()
        self.L = 2048 
        self.D = 524  
        self.K = 1
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, H):
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        return Y_prob, A


def QMNISTLoss(Y_pred, Y):
    Y_pred = torch.clamp(Y_pred, min=1e-5, max=1. - 1e-5)
    neg_log_likelihood = -1. * (Y * torch.log(Y_pred) +  (1. - Y) * torch.log(1. - Y_pred))
    return neg_log_likelihood