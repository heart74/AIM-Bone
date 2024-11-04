
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
    with open('./logs/'+args.mode+'_log/'+date_str+'_'+args.comments+'_'+config.NAME+'.txt','a') as f:
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

    parser.add_argument('--dirpath', type=str, default='')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batchsize', type=int, default=512)
    parser.add_argument('--comments', type=str, default='')
    # parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()
    update_config(config, args)
    # pdb.set_trace()
    return args

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

def outcomes(total_loss, label, score, mode, sizes):
    acc, _, thre, auc, ap = evaluate_ROC(label, score, pos_label=1)
    eer = 1. - acc
    confusion_matrix = score_thre_confusionmatrix(label, score, 0.5)
    accuracy = np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
    # import pdb; pdb.set_trace()
    vlabel, vscore = [],[]
    ind = 0
    assert sum(sizes)==len(score)==len(label)
    try:
        for size in sizes:
            if size==0:
                vlabel.append(0)
                vscore.append(0.5)
                continue
            assert np.mean(label[ind:ind+size])==label[ind]
            vlabel.append(label[ind])
            vscore.append(np.mean(score[ind:ind+size]))
            ind+=size
    except:
        import pdb; pdb.set_trace()

    
    vacc, _, vthre, vauc, vap = evaluate_ROC(vlabel, vscore, pos_label=1)
    veer = 1. - vacc
    vconfusion_matrix = score_thre_confusionmatrix(vlabel, vscore, 0.5)
    vaccuracy = np.sum(np.diag(vconfusion_matrix))/np.sum(vconfusion_matrix)

    #     prob = np.random.rand()
    # if prob > 0.995:
    bprint(confusion_matrix)
    # import pdb; pdb.set_trace()
    bprint(
        mode+" loss: {:.4f} EER: {:.4f}, acc: {:.4f}, AUC: {:.4f}, AP: {:.4f} vEER: {:.4f}, vacc: {:.4f}, vAUC: {:.4f}, vAP: {:.4f}".format(
            total_loss.item(), eer, accuracy, auc, ap, veer, vaccuracy, vauc, vap))
    return auc


def test(net, criterion, loader, mode, sizes):
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
            # logit = cal_model_fliped_tensor(inputs, net)
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
    auc = outcomes(mean_loss, y_label, y_score, mode, sizes)
    return auc

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
    bprint('device: ', device)

    net, img_size, *_ = model_selection(config.MODEL.BACKBONE,
                                        num_out_classes=2, device=device)
    # net_state_dict = net.state_dict()
    # torch.save(net_state_dict,
    #     os.path.join('model2,ckpt')
    # )
    # exit(0)
    model_dir = args.dirpath

    
    for model_path in sorted(os.listdir(model_dir)):
        if 'AUC' not in model_path:
            continue
        model_path = os.path.join(model_dir, model_path)
        bprint("model loading", model_path)
        state_dict = torch.load(model_path, map_location="cpu")['net_state_dict']
        net.load_state_dict(state_dict, strict=False)

        # hjr
        transform = get_transform_aug(config.MODEL.BACKBONE, loader='cv')
        trans_test = transform["test"]

        bprint("==> Dataset loading...")
        valset = AIMDataset(config.DATASET.TRAIN_ROOT, mode=args.mode,
                                sampleNum=config.DATASET.SAMPLE,
                                transform=False,
                                loader='cv',
                                cal_res=config.CAL_RES,
                                level=config.DATASET.LEVEL,
                                Is_align=config.IS_ALIGN)
        valloader = torch.utils.data.DataLoader(valset,
                                                batch_size=args.batchsize,
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
        
        # print(valset.paths)
        sizes1, sizes2 = [],[]
        sizes = []
        for path,label in zip(valset.paths, valset.labels):
            if label == 1:
                sizes1.append(min(config.DATASET.SAMPLE, len(os.listdir(path))))
            else:
                sizes2.append(min(config.DATASET.SAMPLE, len(os.listdir(path))))
        sizes = sizes1+sizes2
        # import pdb; pdb.set_trace()
        auc = test(net, criterion, valloader, args.mode, sizes)



if __name__ == "__main__":

    args = parse_args()

    bprint(config)
    set_seed(config.SEED)
    # print(args.cfg[args.cfg.index('cfgs/')+5:args.cfg.index('.yaml')])
    assert args.mode in ['val', 'test']
    main(args)
    # print(parse_args().cfg)



