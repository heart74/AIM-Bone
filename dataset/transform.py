"""transform.py - Genaral transformation methods for training set and test set.
   These transformations include resize, normalization, toTensor for all images, and others pseudorandomly for training data,
   including HorizontalFlip, ImageCompression, GaussNoise etc..
"""

# Author: Yutong Yao

from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Normalize, RandomBrightnessContrast,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize, ImageCompression, Rotate,
    ShiftScaleRotate, GridDistortion, GridDropout,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip, GaussianBlur, CoarseDropout,
    PadIfNeeded, ToGray, IAASharpen, IAAEmboss, IAAPiecewiseAffine,ColorJitter)
from dataset.transform_RandomErase import RandomErase

import warnings
warnings.filterwarnings("ignore")


def dfdc_train_transforms(size=224):
    return Compose([
        Resize(height=size, width=size), # resize

        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        # OneOf([
        #     # 非空间性转换
        #     ImageCompression(quality_lower=10, quality_upper=40),
        #     # 随机改变图片的 HUE、饱和度和值
        #     ImageCompression(quality_lower=6, quality_upper=10),], p=0.1),
        GaussNoise(p=0.3),
        # GaussNoise(var_limit=(250,500),p=0.1),
        GaussianBlur(blur_limit=(1,3), p=0.05),
        HorizontalFlip(),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            # 非空间性转换
            RandomBrightnessContrast(), 
            # 随机改变图片的 HUE、饱和度和值
            HueSaturationValue()], p=0.5),
        OneOf([
            CoarseDropout(), 
            GridDropout(),
            RandomErase(height=size, width=size)
            ], p=0.2),
        ToGray(p=0.2),
        # 对图片进行平移（translate）、缩放（scale）和旋转（roatate）
        # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ShiftScaleRotate(shift_limit=0.02, scale_limit=0.04, rotate_limit=2, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        # new added
        # OneOf([
        #     # 畸变相关操作
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        OneOf([
            # 锐化、浮雕等操作
            # CLAHE(clip_limit=2),
            IAASharpen(),
            # IAAEmboss(),
        ], p=0.3),
        
        # Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), #normalize
        # ToTensorV2(), # totensor
        
    ])

def dfdc_test_transforms(size=224):
    return Compose([
        Resize(height=size, width=size), # resize
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), #normalize
        ToTensorV2(), # totensor
    ])

def get_transform_aug(backbone, loader='cv'):
    if loader == 'cv':
        backbone = backbone.lower()
        if backbone in ['mesonet', 'hrnet_w18_small_v2', 'skresnet-18', 'resnest-14d']:
            dsize_default = 256
        elif backbone in ['xception','f3net']:
            dsize_default = 299
        elif backbone in ['efficientnet-b0', 'iresnet_se18','efficientnet-video']:
            dsize_default = 224
        elif backbone in ['efficientnet-b2']:
            dsize_default = 260
        elif backbone in ['efficientnet-b7']:
            dsize_default = 600
        # elif backbone == "":
        #     augment_trans = aug_transform(dsize_default)
        else:
            print('check your spell')
            raise NotImplementedError(backbone)
        default_data_transforms = {
            'train': dfdc_train_transforms(dsize_default),
            'val': dfdc_test_transforms(dsize_default),
            'test': dfdc_test_transforms(dsize_default),
        }
        # print(default_data_transforms)
        return default_data_transforms
        
    else:
        raise NotImplementedError(loader)

