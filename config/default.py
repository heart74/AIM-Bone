"""default.py - Default choices for model training configuration.
   This file set the parameters and the default choices of configuration in case not defined in the yaml file
"""

# Author: Boyuan Liu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.NAME = "base"
_C.DESCRIPTION = ""
_C.GPUS = [0]
_C.BATCH_SIZE = 32
_C.SEED = 423
_C.SAVE_PATH = ""
_C.BATCH_EQU = 1
_C.CAL_RES = 0 # 0 indicates RGB input, 1 indicates res input, 2 indicates concat{RGB, res}
_C.IS_ALIGN = False

# MODEL
_C.MODEL = CN()
_C.MODEL.BACKBONE = "efficientnet-b0"
_C.MODEL.PRETRAINED = ""

# TRAIN
_C.TRAIN = CN()
_C.TRAIN.TRAIN_WORKER = 0
_C.TRAIN.LOG_ITER = 10
_C.TRAIN.LR = 5e-4
_C.TRAIN.WD = 1e-4
_C.TRAIN.MILESTONES = [12]
_C.TRAIN.TOTAL_EPOCH = 15
_C.TRAIN.REAL_PROB = 0.5
 

# DATASET
_C.DATASET = CN()
_C.DATASET.TRAIN_ROOT = [""]
_C.DATASET.LEVEL = "c23"
_C.DATASET.SAMPLE = 20

# VALID
_C.VALID = CN()
_C.VALID.VALID_WORKER = 2
_C.VALID.VALID_ITER = 500

# AIM
_C.AIM = CN()
_C.AIM.USE = False
_C.AIM.VERSION = 2
_C.AIM.PROB = 0.3

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # import pdb; pdb.set_trace()
    namespace = vars(args).keys()
    if 'log_iter' in namespace:
        cfg.TRAIN.LOG_ITER = args.log_iter
    if 'val_iter' in namespace:
        cfg.VALID.VALID_ITER = args.val_iter
    
    if 'train_worker' in namespace:
        cfg.TRAIN.TRAIN_WORKER = args.train_worker
    if 'valid_worker' in namespace:
        cfg.VALID.VALID_WORKER = args.valid_worker

    cfg.freeze()


if __name__ == '__main__':
    print(_C)

