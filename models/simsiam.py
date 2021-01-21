import torch
import torch.nn as nn
import math
# from models.resnet import resnet50
from torchvision.models import resnet50
import torch.nn.functional as F
def D(p, z, version='original'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def negcos(p, z):
    # z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p*z.detach()).sum(dim=1).mean()
    # return - nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.is_cifar_flag = False

    def is_cifar(self, is_cifar=True):
        self.is_cifar_flag = is_cifar
        print('cifar settings')

    def forward(self, x):
        x = self.l1(x)
        if not self.is_cifar_flag:
            x = self.l2(x)
        x = self.l3(x)

        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x


class SimSiam(nn.Module):

    def __init__(self, backbone='resnet50', d=2048, is_cifar=False):
        super(SimSiam, self).__init__()

        # if backbone == 'resnet50':
        #     net = resnet50()
        # else:
        #     raise NotImplementedError('Backbone model not implemented.')
        net=backbone
        num_ftrs = net.fc.in_features
        net = list(net.children())
        net = net[:3] + net[4:]
        self.features = nn.Sequential(*net[:-1])
        # num_ftrs = net.fc.out_features
        # self.features = net


        # projection MLP
        self.projection = ProjectionMLP(num_ftrs, 2048, 2048)
        # prediction MLP
        self.prediction = PredictionMLP(2048, 512, 2048)
        if is_cifar:
            self.projection.is_cifar()

        self.reset_parameters()

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     # projection
    #     z = self.projection(x)
    #     # prediction
    #     p = self.prediction(z)
    #     return z, p

    # def forward(self, x, x1):
    #     x, x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     # projection
    #     z = self.projection(x)
    #     # prediction
    #     p = self.prediction(z)
    #     return z, p
    def forward(self, x1, x2):
        x1, x2 = self.features(x1), self.features(x2)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        f, h = self.projection, self.prediction
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L
    def reset_parameters(self):
        # reset conv initialization to default uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)