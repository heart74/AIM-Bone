from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Normalize, RandomBrightnessContrast,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize, ImageCompression, Rotate,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, Cutout, GridDropout,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip, GaussianBlur, CoarseDropout,
    PadIfNeeded, ToGray, Sharpen, IAAEmboss, PiecewiseAffine, ColorJitter, IAAPerspective, IAAPiecewiseAffine, Downscale, GlassBlur)

import cv2
import numpy as np
import time
import os
from dataset.AIM_util import AugmentInsideMask, AugmentInsideMaskv2

# transform = ColorJitter(brightness=[0.5,1.0], ,p=1.)
img = cv2.imread('example_faces/id3.jpg')
out_path = 'example_AIM_faces/demo_v2'
if not os.path.exists(out_path):
    os.makedirs(out_path)
# for i in np.arange(0,1,0.05):
for i in np.arange(0.05,1,0.05):
    
    # # transform = ColorJitter(brightness=[1,1], contrast=[i,i], saturation=[1,1], hue=[0,0],p=1.)
    # # transform = OneOf([
    # #     RGBShift(r_shift_limit=[10,50],g_shift_limit=[10,50],b_shift_limit=[10,50]),
    # #     RGBShift(r_shift_limit=[-50,-10],g_shift_limit=[-50,-10],b_shift_limit=[-50,-10])],p=1.0)
    # # transform = ImageCompression(quality_lower=10, quality_upper=60,compression_type=ImageCompression.ImageCompressionType.WEBP, p=0.5)
    transform = Downscale(scale_min=0.3,scale_max=0.95,p=1.)
    # # transform = GlassBlur(sigma=0.3,iterations=int(i*20),p=1.)
    transform2 = Compose([
        # Image quality
        OneOf([
            ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
            ImageCompression(quality_lower=50, quality_upper=90, compression_type=ImageCompression.ImageCompressionType.WEBP, p=0.5)],p=.6),
        Downscale(scale_min=0.6,scale_max=0.95,p=.2),
        GaussNoise(p=0.1),
        OneOf([
            GaussianBlur(blur_limit=(1,5), p=0.5), # fwa
            Sharpen(alpha=[0.3,0.7],lightness=[1.,1.],p=.5),
        ], p=1.),


        # Color Transfer
        OneOf([
            # change of Brightness(Dark level)
            ColorJitter(brightness=[0.3, 0.6], contrast=[1,1], saturation=[1,1], hue=[0,0],p=.05),
            # change of Brightness(Grayscale level)
            ColorJitter(brightness=[1, 1], contrast=[1,1], saturation=[0.3,0.6], hue=[0,0],p=.05),
            # change of Color(Yellow to Red)
            OneOf([
                ColorJitter(brightness=[1, 1], contrast=[1,1], saturation=[1,1], hue=[-0.06,-0.02],p=0.5),
                ColorJitter(brightness=[1, 1], contrast=[1,1], saturation=[1,1], hue=[0.02,0.06],p=0.5)
            ],p=.3),
            # random color shift
            OneOf([
                RGBShift(r_shift_limit=[10,50],g_shift_limit=[10,50],b_shift_limit=[10,50]),
                RGBShift(r_shift_limit=[-50,-10],g_shift_limit=[-50,-10],b_shift_limit=[-50,-10])],p=.3)
        ], p=.3),
        # Warping Artifact
        ElasticTransform(alpha_affine=15,p=0.2),
    ],p=1.)
    result = transform(image=img)["image"]
    AIMed = AugmentInsideMask(img, result)
    # time.sleep(3)
    conc = np.hstack((img, result, AIMed))
    cv2.imwrite(os.path.join(out_path,'demo'+str(np.round(i,2))+'.png'),conc,[int(cv2.IMWRITE_WEBP_QUALITY), 0])
