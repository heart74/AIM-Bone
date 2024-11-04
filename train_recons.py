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

date_str = datetime.now().strftime('%m%d_%H%M%S')
def bprint(*content):
    print(*content)
    with open('./recons_logs/train_log/'+date_str+'_'+config.NAME+'_'+args.comments+'.txt','a') as f:
        print(*content,file=f)

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
def parse_args():
    parser = argparse.ArgumentParser(description='experiment for AIM2 by LBY')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='./cfgs/recons/baseline.yaml')
    
    parser.add_argument('--log_iter', type=int, default=10)
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

def outputs_writer(total_loss, label, score, epoch,
                   total_iters, mode, lr=None):
    assert mode in ['Train', 'Val', 'test']
    acc, _, thre, auc, ap = evaluate_ROC(label, score, pos_label=1)
    eer = 1. - acc
    confusion_matrix = score_thre_confusionmatrix(label, score, 0.5)
    accuracy = np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
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

def get_numpy_image(X):
    X = X[:8]
    X = make_grid(X.detach().cpu(), nrow=X.shape[0]).numpy() * 0.5 + 0.5
    # cv2 的RGB与BGR
    X = X.transpose([1,2,0])*255
    np.clip(X, 0, 255).astype(np.uint8)
    return X

def print_stack_imgs(img, recons, save_path='./reconstruction.png'):
    
    img = get_numpy_image(img.cpu())
    recons = get_numpy_image(recons.cpu().detach())
    
    cv2.imwrite(save_path, np.vstack((img, recons)))
    pdb.set_trace()
    return

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
    
    enc_save_dir = os.path.join('./model/encoder', args.cfg[args.cfg.index('cons/')+5:args.cfg.index('.yaml')]+ '_' +config.SAVE_PATH + '_' + date_str)
    dec_save_dir = os.path.join('./model/decoder', args.cfg[args.cfg.index('cons/')+5:args.cfg.index('.yaml')]+ '_' +config.SAVE_PATH + '_' + date_str)

    encoder, img_size, *_ = model_selection(config.MODEL.BACKBONE,
                                        num_out_classes=2, device=device, use_fc=False)
    # net_state_dict = net.state_dict()
    # torch.save(net_state_dict,
    #     os.path.join('model2,ckpt')
    # )
    # exit(0)
    decoder = Efficientdecoder1()
    if config.MODEL.PRETRAINED:
        bprint("[train] loading", config.MODEL.PRETRAINED)
        state_dict = torch.load(config.MODEL.PRETRAINED, map_location="cpu")['net_state_dict']
        encoder.load_state_dict(state_dict, strict=False)

    # hjr
    transform = get_transform_aug(config.MODEL.BACKBONE, loader='cv')
    trans_train = transform["train"]
    trans_test = transform["test"]

    bprint("==> Dataset loading...")
    trainset = AIMDataset(config.DATASET.TRAIN_ROOT, mode='train',
                              sampleNum=config.DATASET.SAMPLE,
                              rate=False,
                              transform=True,
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
    bprint("len of trainloader: {}".format(len(trainloader)))
    bprint("len of valloader: {}".format(len(valloader)))

    # loss
    # criterion = CrossEntropyLabelSmooth(num_classes=2).to(device)
    criterion = F.mse_loss

    
    optimizer = optim.Adam(
        # encoder.parameters(),
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=config.TRAIN.LR, weight_decay=config.TRAIN.WD
    )
    # optimizer = SAM(net.parameters(), optimizer, lr=config.TRAIN.LR)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                milestones=config.TRAIN.MILESTONES,
                                                gamma=0.1)
    if multi_gpus:
        encoder = DataParallel(encoder).to(device)
        decoder = DataParallel(decoder).to(device)
    else:
        encoder = encoder.to(device)
        decoder = decoder.to(device)
    #################################
    #  TRAIN  #######################
    #################################
    total_iters = 1
    bprint("Model Name: " + config.MODEL.BACKBONE)
    bprint("Number of parameter: Encoder: %.2fM  Decoder:  %.2fM" % (sum([param.nelement() for param in encoder.parameters()])/1e6, sum([param.nelement() for param in decoder.parameters()])/1e6))
    os.makedirs(enc_save_dir, exist_ok=True)
    os.makedirs(dec_save_dir, exist_ok=True)
    encoder.train()
    decoder.train()
    

    for epoch in range(1, config.TRAIN.TOTAL_EPOCH + 1):

        # train model
        bprint('Train Epoch: {}/{} ...'.format(epoch, config.TRAIN.TOTAL_EPOCH))

        for data in trainloader:
            img, label = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            logit = encoder(img)
            recons = decoder(logit)
            
            total_loss = criterion(recons, img)

            total_loss.backward()
            optimizer.step()
            # pdb.set_trace()
            total_iters += 1
            if total_iters % config.TRAIN.LOG_ITER == 0:
                # pdb.set_trace()
                bprint('Reconstruct loss: '+ str(total_loss))
                print_stack_imgs(img, recons)
                

            if total_iters % config.VALID.VALID_ITER == 0:
                # import pdb; pdb.set_trace()
                # valloader.dataset.generate()

                #################################
                #  Saving   #####################
                #################################
                if multi_gpus:
                    enc_state_dict = encoder.module.state_dict()
                    dec_state_dict = decoder.module.state_dict()
                else:
                    enc_state_dict = encoder.state_dict()
                    dec_state_dict = decoder.state_dict()
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': enc_state_dict
                    },
                    os.path.join(enc_save_dir, 'Encoder_Iter_%06d_loss%.02f.ckpt' % (total_iters, total_loss))
                )
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': dec_state_dict
                    },
                    os.path.join(dec_save_dir, 'Decoder_Iter_%06d_loss%.02f.ckpt' % (total_iters, total_loss))
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



