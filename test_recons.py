import argparse
import os
from datetime import datetime
import torch
from dataset.transform import get_transform_aug
from dataset.AIM_loader import AIMDataset
from torchvision.utils import make_grid
from torch.nn import DataParallel
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from backbone.model_selection import model_selection
from backbone.decoder import Efficientdecoder1
from utils.metrics import evaluate_ROC, get_confusion_matrix, score_thre_confusionmatrix
import numpy as np
import pdb
from config import config
from config import update_config
# from utils.loss import CrossEntropyLabelSmooth
# from sam import SAM
import itertools
import cv2
from train_recons import *

date_str = datetime.now().strftime('%m%d_%H%M%S')

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

def parse_args():
    parser = argparse.ArgumentParser(description='experiment for AIM2 by LBY Image reconstruction effect test')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='./cfgs/recons/baseline.yaml')
    
    parser.add_argument('--train_worker', type=int, default=0)
    parser.add_argument('--valid_worker', type=int, default=2)
    parser.add_argument('--model_dir', type=str,  default='')
    parser.add_argument('--models', type=str,nargs='+',  default=['./model/encoder/baseline__1231_134805/Encoder_Iter_036000_loss0.04.ckpt'])
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    update_config(config, args)
    return args

def main(args):
    # gpu init
    multi_gpus = False
    if len(config.GPUS) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config.GPUS])
    # assert torch.cuda.is_available(), 'CUDA not available.'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device: ', device)

    # log init
    if args.model_dir:
        f1 = [os.path.join(args.model_dir, path) for path in sorted(os.listdir(args.model_dir))]
        f2 = [a.replace('enc','dec').replace('Enc', 'Dec') for a in f1]
    else:
        f1 = args.models
        f2 = [a.replace('enc','dec').replace('Enc', 'Dec') for a in f1]
    for encoder_model, decoder_model in zip(f1,f2):
        encoder, img_size, *_ = model_selection(config.MODEL.BACKBONE,
                                            num_out_classes=2, device=device, use_fc=False)
        print("Loading Encoder:", encoder_model)
        state_dict = torch.load(encoder_model, map_location="cpu")['net_state_dict']
        encoder.load_state_dict(state_dict, strict=False)
        # net_state_dict = net.state_dict()
        # torch.save(net_state_dict,
        #     os.path.join('model2,ckpt')
        # )
        # exit(0)
        decoder = Efficientdecoder1()
        print("Loading Decoder:", decoder_model)
        state_dict = torch.load(decoder_model, map_location="cpu")['net_state_dict']
        decoder.load_state_dict(state_dict, strict=False)

        # hjr
        print("==> Dataset loading...")
        trainset = AIMDataset(config.DATASET.TRAIN_ROOT, mode='train',
                                sampleNum=config.DATASET.SAMPLE,
                                rate=False,
                                transform=False,
                                loader='cv',
                                Use_AIM=False,
                                AIM_ver=config.AIM.VERSION,
                                real_prob=config.TRAIN.REAL_PROB,
                                AIM_prob=config.AIM.PROB,
                                batch_equ=config.BATCH_EQU,
                                cal_res=config.CAL_RES,
                                level=config.DATASET.LEVEL,
                                Is_align=config.IS_ALIGN)
        valset = AIMDataset(config.DATASET.TRAIN_ROOT, mode='val',
                                sampleNum=config.DATASET.SAMPLE,
                                transform=False,
                                loader='cv',
                                Use_AIM=False,
                                cal_res=config.CAL_RES,
                                level=config.DATASET.LEVEL,
                                Is_align=config.IS_ALIGN)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=config.BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=config.TRAIN.TRAIN_WORKER,
                                                drop_last=False)
        valloader = torch.utils.data.DataLoader(valset,
                                                batch_size=config.BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=config.VALID.VALID_WORKER,
                                                drop_last=False)
        print("len of trainloader: {}".format(len(trainloader)))
        print("len of valloader: {}".format(len(valloader)))

        # loss
        # criterion = CrossEntropyLabelSmooth(num_classes=2).to(device)
        # optimizer = SAM(net.parameters(), optimizer, lr=config.TRAIN.LR)
        if multi_gpus:
            encoder = DataParallel(encoder).to(device)
            decoder = DataParallel(decoder).to(device)
        else:
            encoder = encoder.to(device)
            decoder = decoder.to(device)
        print("Model Name: " + config.MODEL.BACKBONE)
        print("Number of parameter: Encoder: %.2fM  Decoder:  %.2fM" % (sum([param.nelement() for param in encoder.parameters()])/1e6, sum([param.nelement() for param in decoder.parameters()])/1e6))
        with torch.no_grad():
            # train model
            loader = trainloader if args.mode=='train' else valloader
            for data in loader:
                img, _ = data[0].to(device), data[1].to(device)
                logit = encoder(img)
                recons = decoder(logit)
                # pdb.set_trace()
                print_stack_imgs(img, recons)
            # if total_iters>30000:
            #     break


if __name__ == "__main__":

    args = parse_args()
    set_seed(config.SEED)
    assert args.mode in ['train', 'val']
    # print(args.cfg[args.cfg.index('cfgs/')+5:args.cfg.index('.yaml')])
    main(args)
    # print(parse_args().cfg)



