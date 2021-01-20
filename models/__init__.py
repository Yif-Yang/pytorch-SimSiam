from .simsiam import SimSiam
from .simsiam_hua import SimSiam as Simsiamhua
# from .byol import BYOL
# from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import torch
from .backbones import *

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(name, backbone):
    if name == 'simsiam':
        model = SimSiam(get_backbone(backbone))
    elif name == 'simsiamhua':
        model = Simsiamhua(get_backbone(backbone))
    else:
        raise NotImplementedError
    return model






