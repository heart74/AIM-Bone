import argparse
import os
from datetime import datetime
import torch
from dataset.transform import get_transform_aug
from dataset.AIM_loader import AIMDataset
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from backbone.model_selection import model_selection

from utils.metrics import evaluate_ROC, get_confusion_matrix, score_thre_confusionmatrix
import numpy as np

from config import config
from config import update_config
import json
from train import set_seed
import pdb
import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

date_str = datetime.now().strftime('%m%d_%H%M%S')

def parse_args():
    parser = argparse.ArgumentParser(description='backbone feature extract and saved to pickle')

    parser.add_argument('--cfg',
                        help='define the dataset, preprocess, model, sample_num',
                        type=str,
                        default='./cfgs/feature/FFppv1.yaml')
    parser.add_argument('--max_num', type=int, default=-1,help='if set to -1 (default), no limit is set for total sample_num, else the model extract max_num images')
    parser.add_argument('--modelpath', type=str, default='./model/baseline_64__1028_223513/Iter_010000_AUC98.13.ckpt')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batchsize', type=int, default=512)
    parser.add_argument('--comments', type=str, default='')
    parser.add_argument('--extra_comments', type=str, default='')

    args = parser.parse_args()
    update_config(config, args)
    # pdb.set_trace()
    return args

def test(net, loader, args):
    net.eval()
    for p in net.parameters():
        device = p.device
        break
    y_label = []
    y_feat = []
    count = 0
    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            inputs, classes = data[0].to(device), data[1].to(device)

            feature = net(inputs)
            # pdb.set_trace()
            # roc curve
            y_label += classes.cpu().numpy().tolist()
            y_feat += feature.cpu().numpy().tolist()
            count+=args.batchsize
            if args.max_num>0 and count>=args.max_num:
                y_feat=y_feat[:args.max_num]
                y_label=y_label[:args.max_num]
                break
    # import pdb; pdb.set_trace()
    return y_feat, y_label



def main(args):
    # gpu init
    multi_gpus = False
    if len(config.GPUS) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config.GPUS])
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.gpu])
    # assert torch.cuda.is_available(), 'CUDA not available.'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device: ', device)

    net, img_size, *_ = model_selection(config.MODEL.BACKBONE,
                                        num_out_classes=2, device=device, use_fc=False)
    # net_state_dict = net.state_dict()
    # torch.save(net_state_dict,
    #     os.path.join('model2,ckpt')
    # )
    # exit(0)
    model_path = args.modelpath
    print("model loading", model_path)
    state_dict = torch.load(model_path, map_location="cpu")['net_state_dict']
    net.load_state_dict(state_dict, strict=False)
    transform = get_transform_aug(config.MODEL.BACKBONE, loader='cv')
    trans_test = transform["test"]
    print("==> Dataset loading...")
    valset = AIMDataset(config.DATASET.TRAIN_ROOT, mode=args.mode,
                            sampleNum=config.DATASET.SAMPLE,
                            transform=trans_test,
                            Use_AIM=config.AIM.USE,
                            AIM_ver=config.AIM.VERSION,
                            AIM_prob=config.AIM.PROB,
                            loader='cv',
                            cal_res=config.CAL_RES,
                            level=config.DATASET.LEVEL,
                            Is_align=config.IS_ALIGN)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batchsize,
                                            shuffle=True,
                                            num_workers=config.VALID.VALID_WORKER,
                                            drop_last=False)
    print("len of "+args.mode+"loader: {}".format(len(valloader)))
    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    print("Model Name: " + config.MODEL.BACKBONE)
    print("Number of parameter: %.2fM" % (sum([param.nelement() for param in net.parameters()])/1e6))
    # import pdb; pdb.set_trace()
    feats, labels = test(net, valloader, args)
    feat_dict = {}
    assert len(feats)==len(labels)
    feat_dict['comments'] = args.comments+' '+args.extra_comments
    feat_dict['length'] = len(feats)
    feat_dict['feat_shape']=len(feats[0])
    feat_dict['config_file'] = args.cfg
    feat_dict['model'] = args.modelpath
    feat_dict['labels'] = args.cfg.split('/')[-1][:-5]
    feat_dict['feats'] = feats
    
    with open('./json_features/'+date_str+'_'+args.comments+'.json', 'w', encoding='utf-8') as f:
        json.dump(feat_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    args = parse_args()
    set_seed(config.SEED)
    assert args.mode in ['val', 'test']
    main(args)

