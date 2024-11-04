 
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

date_str = datetime.now().strftime('%m%d_%H%M%S')
def bprint(*content):
    print(*content)
    with open('./logs/train_log/'+date_str+'_'+config.NAME+'_'+args.comments+'.txt','a') as f:
        print(*content,file=f)

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
def parse_args():
    parser = argparse.ArgumentParser(description='experiment for AIM by LBY')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='./cfgs/base_32.yaml')
    
    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--val_iter', type=int, default=250)
    parser.add_argument('--train_worker', type=int, default=0)
    parser.add_argument('--valid_worker', type=int, default=2)
    parser.add_argument('--comments', type=str, default='')
    args = parser.parse_args()
    update_config(config, args)
    return args


def set_seed(seed):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 为了禁止hash随机化，使得实验可复现
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # if you are using multi-GPU.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def confusionmatrix_writer(confusion_matrix,
                           mode='Train', total_iters=-1):
    assert mode in ['Train', 'Val']
    true_mask = np.zeros_like(confusion_matrix)
    true_mask[0, 0] = confusion_matrix[0, 0]
    true_mask[1:, 1] = confusion_matrix[1:, 1]
    # true_mask = np.eye(confusion_matrix.shape[0]) * confusion_matrix
    # true_mask[1, 1] = confusion_matrix[1, 1]
    per_class_acc = true_mask.sum(1) / confusion_matrix.sum(1)
    total_acc = true_mask.sum() / confusion_matrix.sum()
    min_acc = min(per_class_acc)
    # prob = 1.0
    if mode == 'Val':
    #     prob = np.random.rand()
    # if prob > 0.995:
        
        bprint(confusion_matrix)
    return total_acc, min_acc

def outputs_writer(total_loss, label, score, epoch,
                   total_iters, mode, lr=None):
    assert mode in ['Train', 'Val', 'test']
    acc, _, thre, auc, ap = evaluate_ROC(label, score, pos_label=1)
    eer = 1. - acc
    confusion_matrix = score_thre_confusionmatrix(label, score, 0.5)
    accuracy = np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
    confusionmatrix_writer(confusion_matrix, mode, total_iters=total_iters)
    if mode == 'Train':
        bprint(
            "Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, EER: {:.4f}, acc: {:.4f}, AUC: {:.4f}, AP: {:.4f}, lr: {}".format(
                total_iters, epoch, total_loss.item(), eer, accuracy, auc, ap, lr))
    else:
        # import pdb; pdb.set_trace()
        bprint(
            mode + " Iters: {:0>6d}/[{:0>2d}], Val loss: {:.4f} EER: {:.4f}, acc: {:.4f}, AUC: {:.4f}, AP: {:.4f}".format(
                total_iters, epoch, total_loss.item(), eer, accuracy, auc, ap))
    return auc


def test(net, criterion, loader, epoch, total_iters, mode):
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
    auc = outputs_writer(mean_loss, y_label, y_score, epoch, total_iters, mode)
    net.train()
    return auc

def main(args):
    # gpu init
    multi_gpus = False
    if len(config.GPUS) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config.GPUS])
    # assert torch.cuda.is_available(), 'CUDA not available.'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    bprint('device: ', device)

    # log init
    
    save_dir = os.path.join('./model', args.cfg[args.cfg.index('cfgs/')+5:args.cfg.index('.yaml')]+ '_' +config.SAVE_PATH + '_' + date_str)
    

    net, img_size, *_ = model_selection(config.MODEL.BACKBONE,
                                        num_out_classes=2, device=device)
    # net_state_dict = net.state_dict()
    # torch.save(net_state_dict,
    #     os.path.join('model2,ckpt')
    # )
    # exit(0)

    if config.MODEL.PRETRAINED:
        bprint("[train] loading", config.MODEL.PRETRAINED)
        state_dict = torch.load(config.MODEL.PRETRAINED, map_location="cpu")['net_state_dict']
        net.load_state_dict(state_dict, strict=False)

    # hjr
    transform = get_transform_aug(config.MODEL.BACKBONE, loader='cv')
    trans_train = transform["train"]
    trans_test = transform["test"]

    bprint("==> Dataset loading...")
    trainset = AIMDataset(config.DATASET.TRAIN_ROOT, mode='train',
                              sampleNum=config.DATASET.SAMPLE,
                              rate=True,
                              transform=True,
                              loader='cv',
                              Use_AIM=config.AIM.USE,
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
    bprint("len of trainloader: {}".format(len(trainloader)))
    bprint("len of valloader: {}".format(len(valloader)))

    # loss
    # criterion = CrossEntropyLabelSmooth(num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)


    optimizer = optim.Adam(
        net.parameters(),
        lr=config.TRAIN.LR, weight_decay=config.TRAIN.WD
    )
    # optimizer = SAM(net.parameters(), optimizer, lr=config.TRAIN.LR)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                milestones=config.TRAIN.MILESTONES,
                                                gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    #################################
    #  TRAIN  #######################
    #################################
    total_iters = 1
    bprint("Model Name: " + config.MODEL.BACKBONE)
    bprint("Number of parameter: %.2fM" % (sum([param.nelement() for param in net.parameters()])/1e6))
    os.makedirs(save_dir, exist_ok=True)
    net.train()
    for epoch in range(1, config.TRAIN.TOTAL_EPOCH + 1):

        # train model
        bprint('Train Epoch: {}/{} ...'.format(epoch, config.TRAIN.TOTAL_EPOCH))

        for data in trainloader:
            img, label = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            
            logit = net(img)
            total_loss = criterion(logit, label)

            total_loss.backward()
            optimizer.step()

            total_iters += 1
            if total_iters % config.TRAIN.LOG_ITER == 0:
                
                y_label = list(label.cpu().numpy())
                prob = torch.softmax(logit, dim=1)
                y_score = list(prob[:, 1].detach().cpu().numpy())
                lr = exp_lr_scheduler.get_last_lr()[0]
                outputs_writer(total_loss, y_label, y_score, epoch, total_iters, "Train", lr=lr)
            if total_iters % config.VALID.VALID_ITER == 0:
                # import pdb; pdb.set_trace()
                auc = test(net, criterion, valloader, epoch, total_iters, 'Val')
                valloader.dataset.generate()

                #################################
                #  Saving   #####################
                #################################
                if auc > 0.7:
                    print('Saving checkpoint: {}'.format(total_iters))
                    if multi_gpus:
                        net_state_dict = net.module.state_dict()
                    else:
                        net_state_dict = net.state_dict()
                    torch.save({
                        'iters': total_iters,
                        'net_state_dict': net_state_dict
                        },
                        os.path.join(save_dir, 'Iter_%06d_AUC%.02f.ckpt' % (total_iters, auc*100))
                    )

        exp_lr_scheduler.step()
        trainloader.dataset.generate()
        # if total_iters>30000:
        #     break


if __name__ == "__main__":

    args = parse_args()

    bprint(config)
    set_seed(config.SEED)
    # print(args.cfg[args.cfg.index('cfgs/')+5:args.cfg.index('.yaml')])
    main(args)
    # print(parse_args().cfg)



