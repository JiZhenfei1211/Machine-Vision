import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable as V
import torchvision.models as models
import numpy as np
import sys
import math

class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        densenet = models.densenet161(pretrained=True)
        self.densenet = nn.Sequential(*list(densenet.children())[:-1])
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(2208, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, momentum=0.01),
            nn.Linear(512, args.num_cls),
            nn.Sigmoid()
        )
        
        
    def forward(self, images):
        with torch.no_grad():
            out = self.densenet(images)
        out = F.relu(out, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0),-1)
        out = self.classifier(out)
        return out


def comp_bce_loss_weights(target):
    weights = np.copy(target.cpu().numpy())
    batch_size = weights.shape[0]
    class_num = weights.shape[1]
    
    pos_weight = 1 - np.sum(weights, axis=0) / batch_size
    neg_weight = 1 - pos_weight
    
    for i in range(batch_size):
        for j in range(class_num):
            if weights[i, j] == 1:
                weights[i, j] = pos_weight[j] + 1
            else:
                weights[i, j] = neg_weight[j] + 1
    weights = torch.Tensor(weights)
    return weights
