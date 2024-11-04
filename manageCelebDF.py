import json

def DFDC_preview_parse():
    # no validation set
    with open('DFDC-preview/dfdc_preview_set/dataset.json', 'r') as f:
        infos = json.load(f)

    train_r,train_f,test_r,test_f = [],[],[],[]

    for path, info in infos.items():
        path = path[:-4]
        if info['set']=="train":
            if info['label']=="real":
                train_r.append(path)
            else:
                train_f.append(path)
        if info['set']=="test":
            if info['label']=="real":
                test_r.append(path)
            else:
                test_f.append(path)

    with open('train-list.txt','w') as f:
        for t in train_r:
            f.write(t+' '+ '1'+'\n')
        for t in train_f:
            f.write(t+' '+ '0'+'\n')

    with open('test-list.txt','w') as f:
        for t in test_r:
            f.write(t+' '+ '1'+'\n')
        for t in test_f:
            f.write(t+' '+ '0'+'\n')

import csv
def DFDC_test_parse():
    f = csv.reader(open('DFDC-testset/labels.csv','r'))
    with open('test-list.txt','w') as f1:
        for file, label in f:
            if label=='0':
                f1.write(file[:-4]+' '+ '1\n')
            if label=='1':
                f1.write(file[:-4]+' '+ '0\n')
            
DFDC_preview_parse()
# DFDC_test_parse()