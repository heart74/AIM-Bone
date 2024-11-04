"""transform_RandomErase.py - Transformation methods that pseudorandomly removes landmarks for training set.
"""

# Author: Yutong Yao

import torch
import torch.nn.functional as F

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

import random
import math
from scipy.ndimage import binary_erosion, binary_dilation
from albumentations import ImageOnlyTransform

REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]

class RandomErase(ImageOnlyTransform):
    def __init__(self, height, width, always_apply=False, p=0.5):
        super(RandomErase, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def remove_eyes(self, image, landmarks):
        image = image.copy()
        (x1, y1), (x2, y2) = landmarks[:2]
        mask = np.zeros_like(image[..., 0])
        line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
        w = self.dist((x1, y1), (x2, y2))
        dilation = int(w // 4)
        line = binary_dilation(line, iterations=dilation)
        image[line, :] = 0
        return image

    def remove_nose(self, image, landmarks):
        image = image.copy()
        (x1, y1), (x2, y2) = landmarks[:2]
        x3, y3 = landmarks[2]
        mask = np.zeros_like(image[..., 0])
        x4 = int((x1 + x2) / 2)
        y4 = int((y1 + y2) / 2)
        line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
        w = self.dist((x1, y1), (x2, y2))
        dilation = int(w // 4)
        line = binary_dilation(line, iterations=dilation)
        image[line, :] = 0
        return image

    def remove_mouth(self, image, landmarks):
        image = image.copy()
        (x1, y1), (x2, y2) = landmarks[-2:]
        mask = np.zeros_like(image[..., 0])
        line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
        w = self.dist((x1, y1), (x2, y2))
        dilation = int(w // 3)
        line = binary_dilation(line, iterations=dilation)
        image[line, :] = 0
        return image

    def remove_landmark(self, image, model_landmark=REFERENCE_FACIAL_POINTS, height=224, width=224):

        landmarks = np.array(model_landmark) # [:, [1,0]]
        # 模板反映射
        landmarks[:, [0]] = landmarks[:, [0]] * height/96 
        landmarks[:, [1]] = landmarks[:, [1]] * width/112
        landmarks = landmarks.astype(np.int16)

        if random.random() > 0.5:
            image = self.remove_eyes(image, landmarks)
        elif random.random() > 0.5:
            image = self.remove_mouth(image, landmarks)
        elif random.random() > 0.5:
            image = self.remove_nose(image, landmarks)
        return np.uint8(image)


    def apply(self, img, **params):
        return self.remove_landmark(img, height=self.height, width=self.width)