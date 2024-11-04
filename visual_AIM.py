from dataset.AIM_util import AugmentInsideMaskv1, AugmentInsideMaskv2, AugmentInsideMask
import cv2
import os
import numpy as np


image_root = '.\\code_pack\\00008c1864c83093a95bc52638baf72e_0.jpg'
output_path = '.\\out.png'
mask = '.\\HolyMask2.1.png'
image = cv2.imread(image_root)
AIMed = AugmentInsideMaskv2(image, edgeJitter=0.0)
AIMed = np.hstack((image, AIMed))
cv2.imwrite(output_path, AIMed)  