

"""AIM_loader.py - Dataset building and augmentation of dataset.
   Different sorts of pseudorandom image augmentation (w/ seed set) and real-time fake face generation method are applied for generalization.
   
   Image augmentation includes trasformations defined in transform.py where trasformation differs between training set and test/validation set.
   Besides,
   Three types of Real-time fake face generation method are applied in this file

    2. FaceWarpingArtifacts(fwa_api) is applied when triggered as we select one real images from the training set and apply diffenent noise pattern inside
   an affined general mask, besides, motion blur is pseudorandomly applied on the edge of the mask to create artifacts as a simulation of some
   common fake images , thus create a negative sample.

"""

# Author: Boyuan Liu

import torch
import torchvision.transforms as transforms
import torch.utils.data as torch_data
import random
import numpy as np
import cv2
import os
import json
from PIL import Image
import csv
from .AIM_util import *
from .transform import dfdc_test_transforms, dfdc_train_transforms
import albumentations
import pdb

class AIMDataset(torch.utils.data.Dataset):
    def __init__(self, rootPaths, img_size=224, mode="train", sampleNum=-1, rate=None, transform=False, loader="cv", Use_AIM=False, AIM_ver=1, real_prob=0.5, AIM_prob=0.3, batch_equ=1, cal_res=0, level="c23", Is_align=False):

        # why not super
        self.rootPaths = rootPaths
        self.sampleNum = sampleNum
        if "TRAIN" in str.upper(mode):
            listName = "train-list.txt"
        elif "VAL" in str.upper(mode):
            listName = "val-list.txt"
            # listName = "train-list.txt"
        elif "TEST" in str.upper(mode):
            listName = "test-list.txt"
        else:
            raise NotImplementedError("mode, {}".format(mode))
        self.mode = mode
        self.level = level
        self.paths, self.labels = [], []

        if 'v' in self.level:
            cut = listName.index('.txt')
            listName = listName[:cut]+'-'+self.level+'.txt'
        for rootPath in self.rootPaths:
            if self.level in ['c23', 'c40']:
                rootPath = os.path.join(rootPath,self.level,'frames')

            paths, labels = parse_list_text(rootPath, listName)
            # pdb.set_trace()
            self.paths += paths
            self.labels += labels

        
        self.batch_equ = batch_equ

        self.Use_AIM = Use_AIM
        
        if AIM_ver <= 6:
            self.AIM = eval('AugmentInsideMaskv'+str(AIM_ver))
        else:
            # pdb.set_trace()
            raise NotImplementedError() 
        self.AIM_prob = AIM_prob
        self.real_prob = real_prob

        self.rate = rate
        print("[Selected Dataset] rate:", self.rate)
        self.generate()

        self.transform = transform

        self.loader = pil_loader


        if "CV" in str.upper(loader):
            self.loader = cv_loader


        self.f_id = 0
        self.a_id = 0
        self.r_id = 0
        # self.realList.sort()
        self.cal_res = cal_diff_jpg_v3_1 if cal_res>0 else None
        self.Is_align = Is_align
        self.img_size = img_size
        # pdb.set_trace()
        if self.cal_res:
            print("cal_res:", self.cal_res.__name__)
        else:
            print("cal_res: False")
        print("Is_align:", self.Is_align)
    def sample(self, imageNames, decayRate=1.0):
        """
        decayRate: 基于self.sampleNum的采样浮动比例，用于类别均衡
        """
        sampleNumNow = int(decayRate*self.sampleNum)
        if self.sampleNum == -1 or len(imageNames) <= sampleNumNow:
            return imageNames
        return random.sample(imageNames, sampleNumNow)
        
    def generate(self):

        
        realList = []
        fakeList = []
        # self.labelList = []
        print("[Selected Dataset] loading frameList")
        for path, label in zip(self.paths, self.labels):
            
            decayRate = get_rate(self.labels)[label] if self.rate else 1.
            folder_path = path
            # json_path = folder_path + ".json"
            if not os.path.exists(folder_path):
                print("[Dataset] warning: not exist: ", folder_path, " skipped.")
                continue
            # with open(json_path, 'r') as load_f:
            #     json_info = json.load(load_f)
            # import pdb; pdb.set_trace()
            # for faceName in self.sample(json_info.keys(), decayRate):
            for faceName in self.sample(os.listdir(folder_path), decayRate):
                imgPath = os.path.join(folder_path, faceName)
                # hjr
                if label == 1:
                    realList.append(imgPath)
                else:
                    fakeList.append(imgPath)
        if self.Use_AIM:
            aimList = random.sample(realList, int(self.AIM_prob*len(realList)))
            if self.batch_equ==0:
                fakeList = random.sample(fakeList, len(fakeList)-int(self.AIM_prob*len(realList)))
        else:
            aimList = []
        self.imageList  = realList+fakeList+aimList
        self.labelList = [1]*len(realList)+[0]*len(fakeList)+[2]*len(aimList)
        # import pdb; pdb.set_trace()
        # random.shuffle(self.realList)
        # random.shuffle(self.fakeList)

        print("[Selected Dataset] len(VideoList):", len(self.paths))
        print("[Selected Dataset] len(FakeList):", len(fakeList))
        print("[Selected Dataset] len(RealList) :", len(realList))
        print("[Selected Dataset] len(AIMList) :", len(aimList))
        print("[Selected Dataset] total image nums :", len(self.imageList))
        # pdb.set_trace()
        
    def __getitem__(self, index):

        frame_path = self.imageList[index]
        frame = self.loader(frame_path)
        # frame = np.uint8(frame)

        face = frame
        # if self.Is_align:
        #     face = read_align_face(frame, face_info['landms'])
        # else:
        #     face = read_crop_face(frame, face_info["box"], blocksize=8)

        label = self.labelList[index]
        if self.transform:
            face = dfdc_train_transforms(self.img_size)(image=face)['image']
        if label==2:
            face = self.AIM(face)
            label = 0
        face = np.uint8(face)
        # face = GaussianBlur(blur_limit=(1,3), p=0.05)(image=face)['image']

        # legacy transform
        if self.cal_res is not None:
            face = self.cal_res(face)
        if index % 1000 == 0:
            cv2.imwrite("face.png", face)
            # import pdb; pdb.set_trace()
        
        face = dfdc_test_transforms(self.img_size)(image=face)['image']

        return face, label

    def __len__(self):
        return len(self.imageList)