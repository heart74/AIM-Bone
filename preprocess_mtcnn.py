import torch
from mtcnn import MTCNN
import cv2
import numpy as np
import argparse
import PIL.Image as Image

from torchvision import transforms as trans
import os
import json
from tqdm import tqdm

FRAME_PER_SPLIT = 1
SKIP_EXISTED = True
DEVICE = "cuda:1"

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


class videoProcessor:
    def __init__(self, device=DEVICE):
        device = torch.device(device)
        mtcnn = MTCNN(device=device)
        mtcnn.min_face_size = 20.0
        # mtcnn.thresholds = [0.4, 0.7, 0.8]
        mtcnn.thresholds = [0.4, 0.7, 0.6]

        self.mtcnn = mtcnn

    def _rotate_img(self, frame):
        frame=cv2.transpose(frame)
        frame = cv2.flip(frame, 1)
        return frame

    def process_img(self, frame):
        try:
            bounding_boxes, landmarks = self.mtcnn.detect_faces(Image.fromarray(frame[:, :, ::-1]))
        except:
            # print('no face found')
            return None
        if len(bounding_boxes) == 0:
            return None
        bbox = bounding_boxes[0]
        landmark = landmarks[0]
        box = [float(i) for i in bbox]
        landms = []
        for i in range(5):
            landms.append(float(landmark[i]))
            landms.append(float(landmark[5+i]))
        return box, landms

    def __call__(self, video_path, frame_path, json_path):
        frames = []
        idxes = []
        # sample frame
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % FRAME_PER_SPLIT == 0:
                frames.append(frame)
                idxes.append(idx)
            idx += 1
        cap.release()

        # detect and save
        results = {}
        results_exist = {}
        if os.path.isfile(json_path):
            with open(json_path, 'r') as load_f:
                results_exist = json.load(load_f)
        not_detected_num = 0
        total_num = len(idxes)
        for frame, idx in zip(frames, idxes):
            idx_str = "{:05d}".format(idx)
            if SKIP_EXISTED and idx_str in results_exist:
                results[idx_str] = results_exist[idx_str]
                continue
            # rotate if face not detected
            for i in range(5):
                det_result = self.process_img(frame)
                if det_result is None:
                    frame = self._rotate_img(frame)
                if i==3:
                    frame = cv2.flip(frame, 1)
                else:
                    break
            if det_result is None:
                not_detected_num += 1
                continue
            box, landms = det_result
            result = {
                "box": box,
                "landms": landms
            }
            results[idx_str] = result
            face = read_crop_face(frame, box)
            face = cv2.resize(face,(224,224)) # Have to resize to keep face on the same size
            cv2.imwrite(os.path.join(frame_path, idx_str+".jpg"), face) # Don't have to save as jpg, png can be a lot better
            # cv2.imwrite(os.path.join(frame_path,idx_str+".png"),face, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(video_path, "{} / {}".format(total_num-not_detected_num, total_num))

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        # import pdb; pdb.set_trace()

path = './DFDC-testset/videos'
def run(video_paths):
    processor = videoProcessor(device=DEVICE)
    tpaths = './DFDC-testset/frames'
    vpaths = path
    for video_name in tqdm(video_paths):
        if video_name[-4:] != ".mp4":
            continue
        # import pdb; pdb.set_trace()
        vname = video_name[:-4]
        vpath = os.path.join(vpaths, video_name)
        # vpath = video_name
        tpath = os.path.join(tpaths, vname)
        jpath = os.path.join(tpaths, vname+".json")
        os.makedirs(tpath, exist_ok=True)
        processor(vpath, tpath, jpath)

from multiprocessing import Pool
if __name__ == '__main__':
    # vid_paths = sorted(os.listdir(path))
    
    with open('dfdc-test_hard.txt', 'r') as f:
        vid_paths = f.readlines()
    vid_paths = [vid_path.strip()+'.mp4' for vid_path in vid_paths]
    # for root, dirs, files in os.walk(path, topdown=False):
        # for file in files:
        #     f_name = os.path.join(root, file)
        #     if '.mp4' in f_name:
        #         vid_paths.append(f_name)

    poo_size = 10
    4
    # import pdb; pdb.set_trace()
    vid_paths = vid_paths + vid_paths[:(poo_size-len(vid_paths)%poo_size)%poo_size]
    vid_paths = np.reshape(vid_paths, (poo_size,-1)).tolist()
    print(len(vid_paths), len(vid_paths[0]))
    with Pool(poo_size) as p:
        p.map(run, vid_paths)
    # print(vid_paths[0])

# 先生成文件夹，每一帧如果有对应Json，直接读取并且跳过
# 如果没有Json，Json加一行并写图片，完成文件夹后写Json文件

# 所以说有json文件就一定是完整的
# 有文件夹图片没json的就是没处理完的