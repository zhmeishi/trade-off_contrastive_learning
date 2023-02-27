####################################################################################
### Code from original implementation: #############################################
### https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py ###
####################################################################################

import random

from PIL import ImageFilter
import numpy as np
import torchvision.transforms as T

class SvAVTransform():
    def __init__(self, size_crops, nmb_crops, min_scale_crops, max_scale_crops):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.single = False
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = T.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([T.Compose([randomresizedcrop,
                                              T.RandomHorizontalFlip(p=0.5),
                                              T.Compose(color_transform),
                                              T.ToTensor(),
                                              T.Normalize(mean=mean, std=std)
                                             ])] * nmb_crops[i])
        self.transform = trans
    
    def __call__(self, x):
        return tuple(map(lambda trans: trans(x), self.transform)) 
    
class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
