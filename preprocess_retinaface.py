from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from Retinaface.data import cfg_mnet, cfg_re50
from Retinaface.layers.functions.prior_box import PriorBox
from Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from Retinaface.models.retinaface import RetinaFace
from Retinaface.utils.box_utils import decode, decode_landm
import time
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./Retinaface/weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=1, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu, device="cuda:1"):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    else:
        device = torch.cuda.current_device()
        # import pdb; pdb.set_trace()
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

FRAME_PER_SPLIT = 1
SKIP_EXISTED = True


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
    def __init__(self):
        torch.set_grad_enabled(False)
        self.cfg = None
        if args.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif args.network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        self.device = torch.device("cuda:1")
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.net = load_model(self.net, args.trained_model, args.cpu, device=self.device)
        self.net.eval()
        print('Finished loading model!')
        # print(net)
        cudnn.benchmark = True
        
        self.net = self.net.to(self.device)
        self.resize = 1

    def _rotate_img(self, frame):
        frame=cv2.transpose(frame)
        frame = cv2.flip(frame, 1)
        return frame
        
    def process_img(self, img_raw):
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        # tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        if len(dets)==0:
            return None
        box = dets[0][:5]
        if box[4]<0.5:
            return None
        # import pdb; pdb.set_trace()
        dets = np.concatenate((dets, landms), axis=1)
        return box, landms[0]

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
            # 若未检测到 旋转重新检测
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
                "box": box.tolist(),
                "landms": landms.tolist()
            }
            results[idx_str] = result
            face = read_crop_face(frame, box)
            face = cv2.resize(face,(224,224))
            cv2.imwrite(os.path.join(frame_path, idx_str+".jpg"), face)
            # cv2.imwrite(os.path.join(frame_path,idx_str+".png"),face, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print(video_path, "{} / {}".format(total_num-not_detected_num, total_num))

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        # import pdb; pdb.set_trace()


path = './DFDC-testset/videos'
def run(video_paths):
    processor = videoProcessor()
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
    
    with open('dfdc-test_hard6.txt', 'r') as f:
        vid_paths = f.readlines()
    vid_paths = [vid_path.strip()+'.mp4' for vid_path in vid_paths]
    # for root, dirs, files in os.walk(path, topdown=False):
        # for file in files:
        #     f_name = os.path.join(root, file)
        #     if '.mp4' in f_name:
        #         vid_paths.append(f_name)

    poo_size = 3
    
    # import pdb; pdb.set_trace()
    vid_paths = vid_paths + vid_paths[:(poo_size-len(vid_paths)%poo_size)%poo_size]
    vid_paths = np.reshape(vid_paths, (poo_size,-1)).tolist()
    print(len(vid_paths), len(vid_paths[0]))
    with Pool(poo_size) as p:
        p.map(run, vid_paths)
    print(vid_paths[0])


# path = './DFDC-preview/dfdc_preview_set'
# def run(video_paths):
#     processor = videoProcessor()
#     tpaths = './DFDC-preview/frames'
#     vpaths = path
#     for video_name in tqdm(video_paths):

#         # import pdb; pdb.set_trace()
#         vname = video_name
#         vpath = os.path.join(vpaths, video_name+'.mp4')
#         # vpath = video_name
#         tpath = os.path.join(tpaths, vname)
#         jpath = os.path.join(tpaths, vname+".json")
#         os.makedirs(tpath, exist_ok=True)
#         processor(vpath, tpath, jpath)

# from multiprocessing import Pool
# if __name__ == '__main__':
#     # vid_paths = sorted(os.listdir(path))[:]
#     # vid_paths = []
#     # for root, dirs, files in os.walk(path, topdown=False):
#     #     for file in files:
#     #         f_name = os.path.join(root, file)
#     #         if '.mp4' in f_name:
#     #             vid_paths.append(f_name)
    
#     with open('dfdc-p_hard2.txt', 'r') as f:
#         vid_paths = f.readlines()

#     vid_paths = [vid_path.strip() for vid_path in vid_paths]
#     poo_size = 3
    
#     # import pdb; pdb.set_trace()
#     vid_paths = vid_paths + vid_paths[:(poo_size-len(vid_paths)%poo_size)%poo_size]
#     vid_paths = np.reshape(vid_paths, (poo_size,-1)).tolist()
#     print(len(vid_paths), len(vid_paths[0]))
#     with Pool(poo_size) as p:
#         p.map(run, vid_paths)