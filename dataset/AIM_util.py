
"""AIM_util.py - Helper functions for building AIM-dataset (w/ data augmentation) and loading.
   These helper functions includes image file loading, image preprocessing, image augmentations, fake face generation ,etc..
"""

# Author: Yutong Yao, Junrui Huang

import torch
# import torchvision.transforms as transforms
# import torch.utils.data as torch_data
import random
import numpy as np
import cv2
import os
import json
from PIL import Image
from scipy import signal
import csv
from alignment.align_trans import get_reference_facial_points, warp_and_crop_face
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Normalize, RandomBrightnessContrast,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize, ImageCompression, Rotate,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, Cutout, GridDropout,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    OpticalDistortion, RandomSizedCrop, VerticalFlip, GaussianBlur, CoarseDropout,
    PadIfNeeded, ToGray, Sharpen, Emboss, PiecewiseAffine, ColorJitter, Downscale)

reference = get_reference_facial_points(default_square=True)

def normalize(img):
    i_min, i_max = img.min(), img.max()
    return (img - i_min) / (i_max-i_min)

# Jerry YYDS
def cal_diff_jpg_v3_1(img):
    """
    Grayscale + edge noise extraction
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 1.0

    w_k = np.array([[1, -1]],
                   dtype='float')
    h_k = np.array([[1], [-1]],
                   dtype='float')
    f_w, f_h = img * 1.0, img * 1.0
    for i in range(1):
        f_w = signal.convolve2d(f_w, w_k, 'same')
        f_h = signal.convolve2d(f_h, h_k, 'same')
    f_res = f_h*0.5 + f_w*0.5
    res_avg = f_res.mean()
    f_res[0,:] = res_avg
    f_res[:,0] = res_avg
    f_res = normalize(f_res)
    f_res = np.dstack([f_res]*3)
    f_res = np.uint8(f_res*255)
    return f_res

def cv_loader(path):
    try:
        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        # if img.shape[:2] != img_size:
        #     img = cv2.resize(img, img_size)
        return img
    except IOError:
        print('Cannot load image ' + path)
    except AttributeError:
        print('AttributeError ' + path)
        exit(0)

def pil_loader(img_path):
    img = Image.open(img_path).convert('RGB')
    return img

def parse_list_text(rootPath, listName):
    paths, labels = [], []
    with open(os.path.join(rootPath, listName), 'r') as f:
        print(os.path.join(rootPath, listName)) 
        lines = [line.rstrip() for line in f]
        # 10003887.mp4 0
        for line in lines:
            relativePath, label = line.split(" ")
            paths.append(os.path.join(rootPath, relativePath))
            labels.append(int(label))
    return paths, labels

def get_rate(labels):
    classNum = len(np.unique(labels))
    rate = [0 for i in range(classNum)]
    for label in labels:
        rate[label] += 1
    # [2,4]
    maxNum = max(rate)
    rate = [maxNum / i  for i in rate]
    # [2,1]
    maxNum = max(rate)
    rate = [i / maxNum  for i in rate]
    # [1,0.5]
    return rate

def read_crop_face(img, box, blocksize=1):
    # img_name = os.path.splitext(img_name)[0]  # exclude image file extension (e.g. .png)
    # landms = info[img_name]['landms']
    # box = info['box']
    height, width = img.shape[:2]
    # enlarge the bbox by 1.3 and crop
    scale = 1.3
    # if len(box) == 2:
    #     box = box[0]
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0) # Check for out of bounds, x-y top left corner
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    y1 = (y1//blocksize)*blocksize
    x1 = (x1//blocksize)*blocksize
    size_bb = (size_bb//blocksize)*blocksize

    cropped_face = img[y1:y1 + size_bb, x1:x1 + size_bb]
    return cropped_face

def read_align_face(img, landms):
    facial5points = [[landms[j*2],landms[j*2+1]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(256,256),
                                        return_trans_inv=False)
    return warped_face



# borrowed from FaceSwap Kowalski
def random_erode_dilate(mask,p=0.5, ksize=None):
    mask = PiecewiseAffine(scale=(0.01, 0.03), p=1.0)(image=mask)['image']
    if random.random()>p:
        if ksize is  None:
            ksize = random.randint(1,10)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.erode(mask,kernel,1)/255
    else:
        if ksize is  None:
            ksize = random.randint(1,5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.dilate(mask,kernel,1)/255
    return mask

#  AIM core function v1.0
def AugmentInsideMaskv1(img,edgeJitter=0.2,ratio=2):
    # img = cv2.imread('example2.png')
    height,width = np.array(img).shape[:2]
    mask = cv2.imread('HolyMask2.1.png',0)
    mask = np.array(cv2.resize(mask,(height,width)))/255
    mask = cv2.resize(mask,(height,width))
    
    # test by using mask as source ↓
    # src = np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255
    # FAKE source ↑ ANNOTATE

    # resized height and width REAL source ↓
    r_h,r_w = int(height*(0.5+random.random())),int(width*(0.5+random.random()))
    src = cv2.resize(img,(r_h,r_w))
    kernel = (5,5)
    noise_trans = Compose([
        
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.2),
        OneOf([
            GaussianBlur(blur_limit=(1,5), p=0.8), # fwa
            Sharpen(),
        ], p=1.),

        ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01, p=0.2),
        
    ])
    src = noise_trans(image=src)["image"]
    src = cv2.resize(src,(height,width))
    # target
    tgt = np.copy(img)
    # output image
    composedImg = np.copy(tgt)
    
    mask = random_erode_dilate(mask)
    maskIndices = np.where(mask!=0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # maskEdge = np.dstack([maskEdge]*3)
    composedImg[maskIndices[0], maskIndices[1]] = mask[maskIndices[0], maskIndices[1]] * src[maskIndices[0], maskIndices[1]] + (1 - mask[maskIndices[0], maskIndices[1]]) * tgt[maskIndices[0], maskIndices[1]]
    if random.random()<edgeJitter:
        mask = mask[:,:,0]
        mask = np.clip(mask*ratio,0,1)
        maskEdge = 4*mask*(1-mask)
        maskEdgeIndices = np.where(maskEdge>0.1)
        maskEdge = np.repeat(maskEdge[:, :, np.newaxis], 3, axis=2)
        jittered = MotionBlur(blur_limit=31,p=1.0)(image=composedImg)['image']
        composedImg[maskEdgeIndices[0], maskEdgeIndices[1]] = maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]] * jittered[maskEdgeIndices[0], maskEdgeIndices[1]] + (1 - maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]]) * composedImg[maskEdgeIndices[0], maskEdgeIndices[1]]
    # cv2.imwrite('composed_mask2{:d}.png'.format(0), composedImg)
    return composedImg


#  AIM core function v2.0
def AugmentInsideMaskv2(img,edgeJitter=0.2,ratio=2):
    # img = cv2.imread('example2.png')
    height,width = np.array(img).shape[:2]
    mask = cv2.imread('HolyMask2.1.png',0)
    mask = np.array(cv2.resize(mask,(height,width)))/255
    mask = cv2.resize(mask,(height,width))
    
    # test by using mask as source ↓
    # src = np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255
    # FAKE source ↑ ANNOTATE

    # resized height and width REAL source ↓
    r_h,r_w = int(height*(0.5+random.random())),int(width*(0.5+random.random()))
    # import pdb; pdb.set_trace()
    src = cv2.resize(img,(r_h,r_w))

    noise_trans = Compose([
        # Image quality
        OneOf([
            ImageCompression(quality_lower=10, quality_upper=80, p=0.5),
            ImageCompression(quality_lower=10, quality_upper=80, compression_type=ImageCompression.ImageCompressionType.WEBP, p=0.5)],p=.6),
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
    src = noise_trans(image=src)["image"]
    src = cv2.resize(src,(height,width))

    cv2.imwrite('visualization\\Rec-Aug.png', src)
    # target
    tgt = np.copy(img)
    # output image
    composedImg = np.copy(tgt)
    
    mask = random_erode_dilate(mask)
    cv2.imwrite('visualization\\Mask_rand.png', np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255)
    cv2.imwrite('visualization\\1-Mask_rand.png', 255-np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255)
    maskIndices = np.where(mask!=0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # import pdb; pdb.set_trace()
    # maskEdge = np.dstack([maskEdge]*3)
    # composedImg[maskIndices[0], maskIndices[1]] = mask[maskIndices[0], maskIndices[1]] * src[maskIndices[0], maskIndices[1]] + (1 - mask[maskIndices[0], maskIndices[1]]) * tgt[maskIndices[0], maskIndices[1]]
    composedImg = mask*src + (1-mask)*tgt
    if random.random()<edgeJitter:
        mask = mask[:,:,0]
        mask = np.clip(mask*ratio,0,1)
        maskEdge = 4*mask*(1-mask)
        maskEdgeIndices = np.where(maskEdge>0.1)
        maskEdge = np.repeat(maskEdge[:, :, np.newaxis], 3, axis=2)
        jittered = MotionBlur(blur_limit=31,p=1.0)(image=composedImg)['image']
        composedImg[maskEdgeIndices[0], maskEdgeIndices[1]] = maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]] * jittered[maskEdgeIndices[0], maskEdgeIndices[1]] + (1 - maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]]) * composedImg[maskEdgeIndices[0], maskEdgeIndices[1]]
    # cv2.imwrite('composed_mask2{:d}.png'.format(0), composedImg)
    cv2.imwrite('visualization\\Inside.png', mask*src)
    cv2.imwrite('visualization\\Outside.png', (1-mask)*tgt)
    cv2.imwrite('visualization\\Composed.png', composedImg)
    return composedImg

#  AIM core function v3.0
def AugmentInsideMaskv3(img,edgeJitter=0.0,ratio=2):
    # img = cv2.imread('example2.png')
    height,width = np.array(img).shape[:2]
    mask = cv2.imread('HolyMask2.1.png',0)
    mask = np.array(cv2.resize(mask,(height,width)))/255
    mask = cv2.resize(mask,(height,width))
    
    # test by using mask as source ↓
    # src = np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255
    # FAKE source ↑ ANNOTATE

    # resized height and width REAL source ↓
    r_h,r_w = int(height*(0.5+random.random())),int(width*(0.5+random.random()))
    src = cv2.resize(img,(r_h,r_w))

    noise_trans = OneOf([
            ImageCompression(quality_lower=10, quality_upper=80, p=0.5),
            ImageCompression(quality_lower=10, quality_upper=80, compression_type=ImageCompression.ImageCompressionType.WEBP, p=0.5)],p=1.)
    src = noise_trans(image=src)["image"]
    src = cv2.resize(src,(height,width))
    # target
    tgt = np.copy(img)
    # output image
    composedImg = np.copy(tgt)
    
    mask = random_erode_dilate(mask)
    maskIndices = np.where(mask!=0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # maskEdge = np.dstack([maskEdge]*3)
    composedImg[maskIndices[0], maskIndices[1]] = mask[maskIndices[0], maskIndices[1]] * src[maskIndices[0], maskIndices[1]] + (1 - mask[maskIndices[0], maskIndices[1]]) * tgt[maskIndices[0], maskIndices[1]]
    if random.random()<edgeJitter:
        mask = mask[:,:,0]
        mask = np.clip(mask*ratio,0,1)
        maskEdge = 4*mask*(1-mask)
        maskEdgeIndices = np.where(maskEdge>0.1)
        maskEdge = np.repeat(maskEdge[:, :, np.newaxis], 3, axis=2)
        jittered = MotionBlur(blur_limit=31,p=1.0)(image=composedImg)['image']
        composedImg[maskEdgeIndices[0], maskEdgeIndices[1]] = maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]] * jittered[maskEdgeIndices[0], maskEdgeIndices[1]] + (1 - maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]]) * composedImg[maskEdgeIndices[0], maskEdgeIndices[1]]
    # cv2.imwrite('composed_mask2{:d}.png'.format(0), composedImg)
    return composedImg

#  AIM core function v4.0
def AugmentInsideMaskv4(img,edgeJitter=0.0,ratio=2):
    # img = cv2.imread('example2.png')
    height,width = np.array(img).shape[:2]
    mask = cv2.imread('HolyMask2.1.png',0)
    mask = np.array(cv2.resize(mask,(height,width)))/255
    mask = cv2.resize(mask,(height,width))
    
    # test by using mask as source ↓
    # src = np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255
    # FAKE source ↑ ANNOTATE

    # resized height and width REAL source ↓
    r_h,r_w = int(height*(0.5+random.random())),int(width*(0.5+random.random()))
    src = cv2.resize(img,(r_h,r_w))

    noise_trans = Compose([
        # Image quality
        Downscale(scale_min=0.6,scale_max=0.95,p=.2),
        GaussNoise(p=0.1),
        OneOf([
            GaussianBlur(blur_limit=(1,5), p=0.5), # fwa
            Sharpen(alpha=[0.3,0.7],lightness=[1.,1.],p=.5),
        ], p=1.),
    ],p=1.)
    src = noise_trans(image=src)["image"]
    src = cv2.resize(src,(height,width))
    # target
    tgt = np.copy(img)
    # output image
    composedImg = np.copy(tgt)
    
    mask = random_erode_dilate(mask)
    maskIndices = np.where(mask!=0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # maskEdge = np.dstack([maskEdge]*3)
    composedImg[maskIndices[0], maskIndices[1]] = mask[maskIndices[0], maskIndices[1]] * src[maskIndices[0], maskIndices[1]] + (1 - mask[maskIndices[0], maskIndices[1]]) * tgt[maskIndices[0], maskIndices[1]]
    if random.random()<edgeJitter:
        mask = mask[:,:,0]
        mask = np.clip(mask*ratio,0,1)
        maskEdge = 4*mask*(1-mask)
        maskEdgeIndices = np.where(maskEdge>0.1)
        maskEdge = np.repeat(maskEdge[:, :, np.newaxis], 3, axis=2)
        jittered = MotionBlur(blur_limit=31,p=1.0)(image=composedImg)['image']
        composedImg[maskEdgeIndices[0], maskEdgeIndices[1]] = maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]] * jittered[maskEdgeIndices[0], maskEdgeIndices[1]] + (1 - maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]]) * composedImg[maskEdgeIndices[0], maskEdgeIndices[1]]
    # cv2.imwrite('composed_mask2{:d}.png'.format(0), composedImg)
    return composedImg

#  AIM core function v5.0
def AugmentInsideMaskv5(img,edgeJitter=0.0,ratio=2):
    # img = cv2.imread('example2.png')
    height,width = np.array(img).shape[:2]
    mask = cv2.imread('HolyMask2.1.png',0)
    mask = np.array(cv2.resize(mask,(height,width)))/255
    mask = cv2.resize(mask,(height,width))
    
    # test by using mask as source ↓
    # src = np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255
    # FAKE source ↑ ANNOTATE

    # resized height and width REAL source ↓
    r_h,r_w = int(height*(0.5+random.random())),int(width*(0.5+random.random()))
    src = cv2.resize(img,(r_h,r_w))

    noise_trans = Compose([
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
        ], p=1.),
    ],p=1.)
    src = noise_trans(image=src)["image"]
    src = cv2.resize(src,(height,width))
    # target
    tgt = np.copy(img)
    # output image
    composedImg = np.copy(tgt)
    
    mask = random_erode_dilate(mask)
    maskIndices = np.where(mask!=0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # maskEdge = np.dstack([maskEdge]*3)
    composedImg[maskIndices[0], maskIndices[1]] = mask[maskIndices[0], maskIndices[1]] * src[maskIndices[0], maskIndices[1]] + (1 - mask[maskIndices[0], maskIndices[1]]) * tgt[maskIndices[0], maskIndices[1]]
    if random.random()<edgeJitter:
        mask = mask[:,:,0]
        mask = np.clip(mask*ratio,0,1)
        maskEdge = 4*mask*(1-mask)
        maskEdgeIndices = np.where(maskEdge>0.1)
        maskEdge = np.repeat(maskEdge[:, :, np.newaxis], 3, axis=2)
        jittered = MotionBlur(blur_limit=31,p=1.0)(image=composedImg)['image']
        composedImg[maskEdgeIndices[0], maskEdgeIndices[1]] = maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]] * jittered[maskEdgeIndices[0], maskEdgeIndices[1]] + (1 - maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]]) * composedImg[maskEdgeIndices[0], maskEdgeIndices[1]]
    # cv2.imwrite('composed_mask2{:d}.png'.format(0), composedImg)
    return composedImg

#  AIM core function v6.0
def AugmentInsideMaskv6(img,edgeJitter=0.0,ratio=2):
    # img = cv2.imread('example2.png')
    height,width = np.array(img).shape[:2]
    mask = cv2.imread('HolyMask2.1.png',0)
    mask = np.array(cv2.resize(mask,(height,width)))/255
    mask = cv2.resize(mask,(height,width))
    
    # test by using mask as source ↓
    # src = np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255
    # FAKE source ↑ ANNOTATE

    # resized height and width REAL source ↓
    r_h,r_w = int(height*(0.5+random.random())),int(width*(0.5+random.random()))
    src = cv2.resize(img,(r_h,r_w))

    noise_trans = ElasticTransform(alpha_affine=15,p=1.)

    src = noise_trans(image=src)["image"]
    src = cv2.resize(src,(height,width))
    # target
    tgt = np.copy(img)
    # output image
    composedImg = np.copy(tgt)
    
    mask = random_erode_dilate(mask)
    maskIndices = np.where(mask!=0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # maskEdge = np.dstack([maskEdge]*3)
    composedImg[maskIndices[0], maskIndices[1]] = mask[maskIndices[0], maskIndices[1]] * src[maskIndices[0], maskIndices[1]] + (1 - mask[maskIndices[0], maskIndices[1]]) * tgt[maskIndices[0], maskIndices[1]]
    if random.random()<edgeJitter:
        mask = mask[:,:,0]
        mask = np.clip(mask*ratio,0,1)
        maskEdge = 4*mask*(1-mask)
        maskEdgeIndices = np.where(maskEdge>0.1)
        maskEdge = np.repeat(maskEdge[:, :, np.newaxis], 3, axis=2)
        jittered = MotionBlur(blur_limit=31,p=1.0)(image=composedImg)['image']
        composedImg[maskEdgeIndices[0], maskEdgeIndices[1]] = maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]] * jittered[maskEdgeIndices[0], maskEdgeIndices[1]] + (1 - maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]]) * composedImg[maskEdgeIndices[0], maskEdgeIndices[1]]
    # cv2.imwrite('composed_mask2{:d}.png'.format(0), composedImg)
    return composedImg





def AugmentInsideMask(img, edgeJitter=1.0,ratio=2):
    # img = cv2.imread('example2.png')
    height,width = np.array(img).shape[:2]
    mask = cv2.imread('HolyMask2.1.png',0)
    mask = np.array(cv2.resize(mask,(height,width)))/255
    
    # test by using mask as source ↓
    src = np.repeat(mask[:, :, np.newaxis], 3, axis=2)*255
    # FAKE source ↑ ANNOTATE

    # resized height and width REAL source ↓
    # r_h,r_w = int(height*(0.5+random.random())),int(width*(0.5+random.random()))
    # src = cv2.resize(img,(r_h,r_w))
    # src = inside
    # src = cv2.resize(src,(height,width))
    # target
    tgt = np.copy(img)
    # output image
    composedImg = np.copy(tgt)
    
    mask = random_erode_dilate(mask)
    maskIndices = np.where(mask!=0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # maskEdge = np.dstack([maskEdge]*3)
    composedImg[maskIndices[0], maskIndices[1]] = mask[maskIndices[0], maskIndices[1]] * src[maskIndices[0], maskIndices[1]] + (1 - mask[maskIndices[0], maskIndices[1]]) * tgt[maskIndices[0], maskIndices[1]]
    if random.random()<edgeJitter:
        mask = mask[:,:,0]
        mask = np.clip(mask*ratio,0,1)
        maskEdge = 4*mask*(1-mask)
        maskEdgeIndices = np.where(maskEdge>0.1)
        maskEdge = np.repeat(maskEdge[:, :, np.newaxis], 3, axis=2)
        jittered = MotionBlur(blur_limit=31,p=1.0)(image=composedImg)['image']
        composedImg[maskEdgeIndices[0], maskEdgeIndices[1]] = maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]] * jittered[maskEdgeIndices[0], maskEdgeIndices[1]] + (1 - maskEdge[maskEdgeIndices[0], maskEdgeIndices[1]]) * composedImg[maskEdgeIndices[0], maskEdgeIndices[1]]
    # cv2.imwrite('composed_mask2{:d}.png'.format(0), composedImg)
    return composedImg



