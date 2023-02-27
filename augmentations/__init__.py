from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
from .moco_aug import MoCoTransform
from .barlowtwins_aug import BarlowTwinsTransform
from .clip_aug import CLIPTransform, CLIPTransform_Single

mnist_mean_std = [[0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]]
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
small_imagenet_mean_std = [[0.4810, 0.4574, 0.4078], [0.2146, 0.2104, 0.2138]]
cifar10_mean_std = [[0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2615]]
cinic10_mean_std = [[0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]]
fer2013_mean_std = [[0.507771, 0.507771, 0.507771], [0.24553601, 0.24553601, 0.24553601]] # 32 * 32
intel_mean_std = [[0.43020985, 0.45750308, 0.4538848], [0.2386186, 0.23767951, 0.27331892]] # 32 * 32

clip_mean_std = [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]]

def get_aug(name='simsiam', mean_std_name='cifar10', image_size=224, train=True, train_classifier=None):

    if mean_std_name == "mnist":
        mean_std = mnist_mean_std
    elif mean_std_name == "cifar10":
        mean_std = cifar10_mean_std
    elif mean_std_name == "imagenet":
        mean_std = imagenet_mean_std
    elif mean_std_name == "small_imagenet":
        mean_std = small_imagenet_mean_std
    elif mean_std_name == "fer2013":
        mean_std = fer2013_mean_std
    elif mean_std_name == "intel":
        mean_std = intel_mean_std
    elif mean_std_name == "clip":
        mean_std = clip_mean_std
    else:
        raise NotImplementedError

    if train==True:
        if name in ['simsiam', 'nnclr']:
            augmentation = SimSiamTransform(image_size, mean_std)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size, mean_std)
        elif name == 'barlowtwins':
            augmentation = BarlowTwinsTransform(image_size, mean_std)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size, mean_std)
        elif name == 'moco':
            augmentation = MoCoTransform(image_size, mean_std)
        elif 'clip' in name or "moco_v3" in name:
            augmentation = CLIPTransform(image_size, mean_std)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        if 'clip' in name:
            augmentation = CLIPTransform_Single(image_size, mean_std)
        else:
            augmentation = Transform_single(image_size, train=train_classifier, mean_std=mean_std)
    else:
        raise Exception
    
    return augmentation








