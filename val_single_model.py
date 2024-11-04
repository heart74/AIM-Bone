 
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
# from utils.loss import CrossEntropyLabelSmooth
# from sam import SAM
from train import set_seed
import pdb
date_str = datetime.now().strftime('%m%d_%H%M%S')

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

def bprint(*content):
    print(*content)
    with open('./logs/'+args.mode+'_log/'+date_str+'_'+config.NAME+'.txt','a') as f:
        print(*content,file=f)

# 测试时也需加载cfg文件，获取训练配置保持一致(backbone, 预处理>{对齐、res等}，HQ/LQ，)，名称、测试使用的数据集、
def parse_args():
    parser = argparse.ArgumentParser(description='experiment for AIM by LBY')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='./cfgs/base_32.yaml')
    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--val_iter', type=int, default=500)
    parser.add_argument('--train_worker', type=int, default=0)
    parser.add_argument('--valid_worker', type=int, default=2)

    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--mode', type=str, default='val')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    update_config(config, args)
    pdb.set_trace()
    return args



def outcomes(total_loss, label, score, mode):
    acc, _, thre, auc, ap = evaluate_ROC(label, score, pos_label=1)
    eer = 1. - acc
    confusion_matrix = score_thre_confusionmatrix(label, score, 0.5)
    accuracy = np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)

    #     prob = np.random.rand()
    # if prob > 0.995:
    bprint(confusion_matrix)
    # import pdb; pdb.set_trace()
    bprint(
        mode+" loss: {:.4f} EER: {:.4f}, acc: {:.4f}, AUC: {:.4f}, AP: {:.4f}".format(
            total_loss.item(), eer, accuracy, auc, ap))
    return auc


def test(net, criterion, loader, mode):
    net.eval()
    for p in net.parameters():
        device = p.device
        break
    y_label = []
    y_score = []
    logging_num = len(loader) // 5
    with torch.no_grad():
        eval_losses = 0
        for i, data in enumerate(loader):
            inputs, classes = data[0].to(device), data[1].to(device)

            logit = net(inputs)
            eval_loss = criterion(logit, classes)
            eval_losses += eval_loss

            # roc curve
            y_label += list(classes.cpu().numpy())
            prob = torch.softmax(logit, dim=1)
            y_score += list(prob[:, 1].cpu().numpy())

            if i % logging_num == 0:
                bprint("validating: {}%...".format(i / len(loader)*100))
    mean_loss = eval_losses / len(loader)
    # import pdb; pdb.set_trace()
    auc = outcomes(mean_loss, y_label, y_score, mode)
    return auc

def main(args):
    # gpu init
    multi_gpus = False
    if len(config.GPUS) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config.GPUS])
    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.gpu])
    # assert torch.cuda.is_available(), 'CUDA not available.'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    bprint('device: ', device)

    net, img_size, *_ = model_selection(config.MODEL.BACKBONE,
                                        num_out_classes=2, device=device)
    # net_state_dict = net.state_dict()
    # torch.save(net_state_dict,
    #     os.path.join('model2,ckpt')
    # )
    # exit(0)
    bprint("model loading", args.path)
    state_dict = torch.load(args.path, map_location="cpu")['net_state_dict']
    net.load_state_dict(state_dict, strict=False)

    # hjr
    transform = get_transform_aug(config.MODEL.BACKBONE, loader='cv')
    trans_test = transform["test"]

    bprint("==> Dataset loading...")
    valset = AIMDataset(config.DATASET.TRAIN_ROOT, mode=args.mode,
                              sampleNum=config.DATASET.SAMPLE,
                              transform=trans_test,
                              loader='cv',
                              cal_res=config.CAL_RES,
                              level=config.DATASET.LEVEL,
                              Is_align=config.IS_ALIGN)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=256,
                                            shuffle=False,
                                            num_workers=config.VALID.VALID_WORKER,
                                            drop_last=False)
    bprint("len of "+args.mode+"loader: {}".format(len(valloader)))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    bprint("Model Name: " + config.MODEL.BACKBONE)
    bprint("Number of parameter: %.2fM" % (sum([param.nelement() for param in net.parameters()])/1e6))
    auc = test(net, criterion, valloader, args.mode)



if __name__ == "__main__":

    args = parse_args()

    bprint(config)
    set_seed(config.SEED)
    # print(args.cfg[args.cfg.index('cfgs/')+5:args.cfg.index('.yaml')])
    assert args.mode in ['val', 'test']
    main(args)
    # print(parse_args().cfg)



