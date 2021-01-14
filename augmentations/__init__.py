from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
def get_aug(name, image_size, train, train_classifier=True, aug_type=None):

    if train==True:
        if name == 'simsiam':

            if aug_type is None:
                augmentation = SimSiamTransform(image_size)
            elif aug_type == 'add_2_hard':
                augmentation = SimSiamTransform(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        else:
            raise NotImplementedError
    elif train==False:
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








