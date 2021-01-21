from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from .resnet_cifar import resnet20
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet18_cifar(**kwargs):
    return resnet20()

def resnet18_cifar_new(**kwargs):
    from .resnet_cifar_new import resnet18 as resnet18_new
    return resnet18_new()

def resnet18_cifar_new_no_maxpool(**kwargs):
    from .resnet_cifar_new_no_maxpool import resnet18 as resnet18_new_no_maxpool
    return resnet18_new_no_maxpool()

def resnet_cifar_from_taoyang(**kwargs):
    from .resnet_from_taoyang1122 import ResNet as resnet_cifar_from_taoyang
    return resnet_cifar_from_taoyang('cifar10', depth=18, num_classes=1000, bottleneck=False)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet50w2(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


# def resnet50w4(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


# def resnet50w5(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3])