import os
import os.path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import torchvision.transforms as Transforms
from torch.utils.data import dataset, dataloader
import torch.nn as nn
import torch.nn.functional as F
from backbone.model_selection import model_selection # baseline choose
from dataset.transform import get_transform
from alignment.align_trans import get_reference_facial_points, warp_and_crop_face
import json
from scipy import signal

backbone_ensembles = ['efficientnet-b0', 'efficientnet-b0', 'efficient-attention-b0']
backbone_weight_paths = ['res3.1_0.2shitter_xray_effb0_seed111_auc98.58_Iter5.5k.ckpt', 'res3.2_0.2shitter_xray_effb0_seed2423_auc98.82_iter6k.ckpt', 'effb0_atten_rbg_align_AUC96.17_Iter_005000.ckpt']
base_path = os.path.join(os.path.dirname(__file__), 'pretrained')
backbone_weight_paths = [os.path.join(base_path, i) for i in backbone_weight_paths]
print(backbone_ensembles)
Is_prob_ensemble = False


def cal_norm_img(img_diff):
    
    diff_max, diff_min = img_diff.max(), img_diff.min()
    img_diff = (img_diff - diff_min) / (diff_max-diff_min)
    return np.uint8(255*img_diff)

def cal_diff_jpg(img, compression_rate=90, norm=True):
    """
    Params
      img, np.uint8
    Return
      img_diff, np.uint8
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_rate]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img_compressed = cv2.imdecode(encimg, 1)
    img_diff = img*1.0 - img_compressed*1.0
    if norm:
        img_diff = cal_norm_img(img_diff)
    return img_diff

def cal_diff_jpg_v2(img, compression_rates=[85,90,95]):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_concat = []
    for rate in compression_rates:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rate]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img_compressed = cv2.imdecode(encimg, 0)
        img_diff = img*1.0 - img_compressed*1.0
        # normalize
        diff_max, diff_min = img_diff.max(), img_diff.min()
        img_diff = (img_diff - diff_min) / (diff_max-diff_min)
        img_concat.append(img_diff)
    img_diff_v2 = np.dstack(img_concat)
    return np.uint8(img_diff_v2*255)

def normalize(img):
    i_min, i_max = img.min(), img.max()
    return (img - i_min) / (i_max-i_min)

def cal_laplacian(img):
    if img.max() < 1.1:
        img = np.uint8(img*255)
    else:
        img = np.uint8(img)
    Ksize = 3
    laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=Ksize)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

def cal_diff_jpg_v3(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 1.0
    h, w = img.shape[0], img.shape[1]

    w_k = np.array([[1, -1]],
                   dtype='float')
    h_k = np.array([[1], [-1]],
                   dtype='float')
    f_w, f_h = img * 1.0, img * 1.0
    for i in range(1):
        f_w = signal.convolve2d(f_w, w_k, 'same')
        f_h = signal.convolve2d(f_h, h_k, 'same')
    f_f = f_h*0.5 + f_w*0.5
    f_lap = cal_laplacian(normalize(f_f))
    f_lap[1,:] = f_lap[0,0]
    f_lap[:,1] = f_lap[0,0]
    f_mean = normalize(f_lap)*255
    f_mean = np.dstack([f_mean]*3)
    f_mean = np.uint8(f_mean)
    return f_mean

def cal_diff_jpg_v3_1(img):
    """
    灰度+提取边缘噪声
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

def cal_diff_jpg_v3_2(img):
    """
    灰度
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 1.0

    f_res = normalize(img)
    f_res = np.dstack([f_res]*3)
    f_res = np.uint8(f_res*255)
    return f_res

def cal_diff_jpg_v3_3(img):
    """
    灰度+提取边缘噪声(三个方向三个维度)
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
    f_w_avg = f_w.mean()
    f_h_avg = f_h.mean()
    f_w[0,:] = f_w_avg
    f_w[:,0] = f_w_avg
    f_h[0,:] = f_h_avg
    f_h[:,0] = f_h_avg
    f_res = f_h*0.5 + f_w*0.5
    f_w   = normalize(f_w)
    f_h   = normalize(f_h)
    f_res = normalize(f_res)
    f_res = np.dstack([f_res, f_w, f_h])
    f_res = np.uint8(f_res*255)
    return f_res

def cal_model_fliped_tensor(image, model):
    """
    Param:
      image: torch.tensor [N,C,H,W]
    Return:
      logit_ensemble: torch.tensor [N, class_num]
    """
    image_fliped = torch.flip(image, dims=(3,))
    image_ensemble = torch.cat([image, image_fliped], dim=0)  # [2N, C, H, W]
    logit = model(image_ensemble)
    N = len(logit) // 2
    logit_ensemble = (logit[:N,:] + logit[N:,:])/2
    return logit_ensemble

class FolderDataset(dataset.Dataset):
    def __init__(self, img_folder, face_info, transform=None):
        self.img_folder = img_folder
        self.imgNames = sorted(os.listdir(img_folder))
        # REMEMBER to use sorted() to ensure correct match between imgNames and predictions
        # do NOT change the above two lines

        self.face_info = face_info
        self.transform = transform

        self.refrence = get_reference_facial_points(default_square=True)

    def __len__(self):
        return len(self.imgNames)

    def read_crop_face(self, img_name, img_folder, info, blocksize=1):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        img_name = os.path.splitext(img_name)[0]  # exclude image file extension (e.g. .png)
        # landms = info[img_name]['landms']
        box = info[img_name]['box']
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

    def read_align_face(self, img_name, img_folder, info):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        img_name = os.path.splitext(img_name)[0]  # exclude image file extension (e.g. .png)
        landms = info[img_name]['landms']
        facial5points = [[landms[j*2],landms[j*2+1]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(256,256),
                                         return_trans_inv=False)
        return warped_face

    def __getitem__(self, idx):
        img_name = self.imgNames[idx]
        # Read-in images are full frames, maybe you need a face cropping.
        img = self.read_crop_face(img_name, self.img_folder, self.face_info, blocksize=8)
        # img_align = self.read_align_face(img_name, self.img_folder, self.face_info)

        img_3_1 = cal_diff_jpg_v3_1(img)
        img_3_2 = cal_diff_jpg_v3_2(img)
        
        img_3_1 = self.transform(img_3_1)
        img_3_2 = self.transform(img_3_2)
        img_att = self.transform(img)
        return img_3_1, img_3_2, img_att


class Model():
    def __init__(self, is_prob_ensemble=Is_prob_ensemble):
        # init and load your model here
        
        self.is_prob_ensemble = is_prob_ensemble
        self.load_models(model_names=backbone_ensembles, model_weight_paths=backbone_weight_paths)
        # determine your own batchsize based on your model size. The GPU memory is 16GB
        # relatively larger batchsize leads to faster execution.
        self.batchsize = 32

    def load_models(self, model_names, model_weight_paths=None):
        self.models = []
        self.sizes  = []
        if model_weight_paths is None:
            model_weight_paths = [None]*len(model_names)
        assert len(model_weight_paths) == len(model_names)
        for name, weight_path in zip(model_names, model_weight_paths):
            model, size, *_ = model_selection(name, num_out_classes=2, weight_path=weight_path)
            model.eval()
            model.cuda(0)
            self.models.append(model)
            self.sizes.append(size)

    def run(self, input_dir, json_file, post_function=nn.Softmax(dim=1)):
        with open(json_file, 'r') as load_f:
            json_info = json.load(load_f)

        transform = get_transform(backbone_ensembles[0], loader='cv')

        dataset_eval = FolderDataset(input_dir, json_info, 
                                        transform=transform["test"]
                                    )
        dataloader_eval = dataloader.DataLoader(dataset_eval, batch_size=self.batchsize,
                                                shuffle=False, num_workers=4)
        # USE shuffle=False in the above dataloader to ensure correct match between imgNames and predictions
        # Do set drop_last=False (default) in the above dataloader to ensure all images processed

        print('Detection model inferring ...')
        prediction = []
        with torch.no_grad():  # Do USE torch.no_grad()
            for imgs_31, imgs_32, img_att in tqdm(dataloader_eval):
                imgs_31 = imgs_31.to('cuda:0')
                imgs_32 = imgs_32.to('cuda:0')
                img_att = img_att.to('cuda:0')
                imgs = [imgs_31, imgs_32, img_att]

                predOneModel = []
                for imgs_m, model, _ in zip(imgs, self.models, self.sizes):
                    # logits = model(imgs_m)
                    logits = cal_model_fliped_tensor(imgs_m, model)
                    if self.is_prob_ensemble:
                        logits = post_function(logits)
                    
                    predOneModel.append(logits)
                preds = sum(predOneModel) / len(predOneModel)
                if not self.is_prob_ensemble:
                    preds = post_function(preds)
                prediction.append(preds[:,1])
        
        prediction = torch.cat(prediction, dim=0)
        prediction = prediction.cpu().numpy()
        prediction = prediction.squeeze().tolist()
        assert isinstance(prediction, list)
        assert isinstance(dataset_eval.imgNames, list)
        assert len(prediction) == len(dataset_eval.imgNames)
        # print("preds:", prediction)

        return dataset_eval.imgNames, prediction


if __name__ == "__main__":
    model = Model()
    print(model)
    input_dir, json_file = './frames/10000001', './frames/10000001.json'
    names,predictions = model.run(input_dir, json_file)
    print(names,predictions)