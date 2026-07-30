# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the swav directory of this source tree.
#

##############################################################
'''
Changes made:
- Adapted to grayscale images instead of RGB 
- Norms changed from default to calculated from the CEM1.5M dataset 
- Transforms changed to match the MitoNet paper




'''

import random
from logging import getLogger

from PIL import ImageFilter
from PIL import Image
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pathlib import Path
from typing import Union

logger = getLogger()

def grayscale_loader(path: Union[str, Path]) -> Image.Image:
    with open(path,'rb') as f:
        img = Image.open(f)

    return img.convert('L')

class MultiCropDataset(datasets.ImageFolder):
    def __init__(
            self,
            data_path,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            size_dataset=-1,
            return_index=False,
    ):
        super().__init__(data_path, loader = grayscale_loader)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        # calculated off of CEM1.5M 
        mean = [0.574]
        std = [0.176]

        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i],max_scale_crops[i])
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0,360), fill =(0,)),
                get_distortion(),
                PILRandomGaussianBlur(),
                transforms.ToTensor(),
                RandomGaussianNoise(p=0.5,mean=0.0,std=0.05),
                transforms.Normalize(mean=mean,std=std),

            ])] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self,index):
        path,_ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops
    
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


class RandomGaussianNoise(object):
    """
    Apply Gaussian Noise to a PyTorch Tensor with probability p.
    Expects tensor values in [0.0, 1.0].
    """

    def __init__(self, p=0.5, mean=0.0, std=0.05):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if random.random() <= self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

def get_distortion(s=1.0):
    # s is the strength of brightness and constrast jitter
    color_jitter = transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

    return rnd_color_jitter
