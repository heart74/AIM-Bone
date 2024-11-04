import os
import shutil

input_dirs = [
    'face-forensics/original_sequences/c23/frames',
    'face-forensics/original_sequences/c40/frames',
]

parent_dir = 'face-forensics/manipulated_sequences'
p_dirs = os.listdir(parent_dir)
for p_dir in p_dirs:
    input_dirs.append(os.path.join(parent_dir, p_dir, 'c23','frames'))
    input_dirs.append(os.path.join(parent_dir, p_dir, 'c40', 'frames'))


def find_half_finished(path = './face-forensics/original_sequences/c23/frames'):
    paths = os.listdir(path)
    for fi in paths:
        if 'json' not in fi and fi+'.json' not in paths:
            # shutil.rmtree(os.path.join(root, name))
            print(fi)

def count_folder(path = './face-forensics/original_sequences/c23/frames', writeTxt=False):
    from collections import Counter
    
    ctr = []
    count = 0
    print(path+':')
    if writeTxt:
        with open('strange_train.txt', 'w') as f:
            for fi in os.listdir(path):
                if 'json' not in fi and 'txt' not in fi:
                    n = len(os.listdir(os.path.join(path,fi)))
                    ctr.append(n)
                    if len(os.listdir(os.path.join(path,fi)))<100:
                        f.write(fi+' '+str(n)+'\n')
                        count+=1
    else:
        for fi in os.listdir(path):
            if 'json' not in fi and 'txt' not in fi:
                n = len(os.listdir(os.path.join(path,fi)))
                ctr.append(n)
                if len(os.listdir(os.path.join(path,fi)))<100:
                    # os.remove(os.path.join(root, name+'.json'))
                    # shutil.rmtree(os.path.join(root, name))
                    print(fi+' '+str(n))
                    count+=1
    ctr = Counter(ctr)
    # print(sorted(ctr.items()))
    print(count)

def count_1k(path = './face-forensics/original_sequences/c23/frames'):
    paths = os.listdir(path)
    json_count = 0
    for fi in paths:
        if 'json' in fi:
            json_count+=1
    if json_count==1000:
        print('Complete')
        return True
    else:
        print(json_count, len(paths))
        return False

for input_dir in input_dirs:
    # print(input_dir, os.path.exists(input_dir))
    if os.path.exists(input_dir):
        print(input_dir)
        print('checking integrality')
        count_1k(input_dir)
        print('checking half-finished:')
        find_half_finished(input_dir)
        print('checking empty folder')
        count_folder(path=input_dir)