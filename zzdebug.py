import json


def check_split_info():
    f_name = 'face-forensics/test.json'
    with open(f_name, 'r') as f:
        lis = json.load(f)
        print(len(lis))

    f_name = 'face-forensics/train.json'
    with open(f_name, 'r') as f:
        lis += json.load(f)
        print(len(lis))
        # print(lis)

    f_name = 'face-forensics/val.json'
    with open(f_name, 'r') as f:
        lis += json.load(f)
        

    srcs,tgts = [],[]
    # with open('test-list.txt', 'w') as f:
    for src,tgt in lis:
        srcs.append(src)
        tgts.append(tgt)
            # print(src+'_'+tgt+' 0', file=f)
    print(len(lis))
    print(len(srcs),len(tgts))
    print(len(set(srcs)),len(set(tgts)))



def create_fake_label():
    f_name = 'face-forensics/test.json'
    with open(f_name, 'r') as f:
        lis = json.load(f)
        print(len(lis))

    f_name = 'face-forensics/train.json'
    with open(f_name, 'r') as f:
        lis = json.load(f)
        print(len(lis))
        # print(lis)

    f_name = 'face-forensics/val.json'
    with open(f_name, 'r') as f:
        lis = json.load(f)
        print(len(lis))

    with open('val-list.txt', 'w') as f:
        for src,tgt in lis:
            print(src+'_'+tgt+' 0', file=f)
            print(tgt+'_'+src+' 0', file=f)

def create_real_label():
    f_name = 'face-forensics/test.json'
    with open(f_name, 'r') as f:
        lis = json.load(f)
        print(len(lis))

    f_name = 'face-forensics/train.json'
    with open(f_name, 'r') as f:
        lis2 = json.load(f)
        print(len(lis2))
        # print(lis)

    f_name = 'face-forensics/val.json'
    with open(f_name, 'r') as f:
        lis3 = json.load(f)
        print(len(lis3))

    with open('test-list.txt', 'w') as f:
        for src,tgt in lis:
            print(src+' 1', file=f)
            print(tgt+' 1', file=f)

    with open('train-list.txt', 'w') as f:
        for src,tgt in lis2:
            print(src+' 1', file=f)
            print(tgt+' 1', file=f)

    with open('val-list.txt', 'w') as f:
        for src,tgt in lis3:
            print(src+' 1', file=f)
            print(tgt+' 1', file=f)

# import time
# from datetime import datetime
# def bprint(*content):
#     print(*content)
#     with open('./logs/train_log/'+'hape2.txt','a') as f:
#         print(*content,file=f)
# while True:
#     time.sleep(3)
#     bprint(datetime.now().strftime('%Y%m%d_%H%M%S'))
import os


def check_dfdc_preview():
    datapath = 'DFDC-preview/frames'
    record = []
    for root, dirs, files in os.walk(datapath, topdown=False):
        for dir in dirs:
            j_dir = os.path.join(root, dir)
            conts =  os.listdir(j_dir)
            if not conts:
                print(j_dir,'Empty!!')
                record.append(j_dir)
                continue
            if '.jpg' in conts[0]:
                if len(conts)<20:
                    print(j_dir, len(conts),'Too Small!!')
                    record.append(j_dir)
    return record


def check_dfdc_testset():
    datapath = 'DFDC-testset/frames'
    record = []
    for root, dirs, files in os.walk(datapath, topdown=False):
        for dir in dirs:
            j_dir = os.path.join(root, dir)
            conts =  os.listdir(j_dir)
            if not conts:
                print(j_dir,'Empty!!')
                record.append(j_dir)
                continue
            if '.jpg' in conts[0]:
                if len(conts)<20:
                    print(j_dir, len(conts),'Too Small!!')
                    record.append(j_dir)
    return record
record = check_dfdc_testset()
with open('dfdc-test_hard6.txt', 'w') as f:
    for rec in record:
        f.write(rec.split('/')[-1]+'\n')
# record = check_dfdc_preview()
# print(record)
# with open('dfdc-p_hard3.txt', 'w') as f:
#     for rec in record:
#         f.write(rec+'\n')