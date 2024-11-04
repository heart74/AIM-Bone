from os import name
import matplotlib.pyplot as plt
import argparse
import pdb



# visualize and compare different log file(different operation) on different criteria

parser = argparse.ArgumentParser(description="visualize and plot the criteria in training(validating) logs")
# parser.add_argument('a', type=int, default=1)
parser.add_argument('--name', type=str, default='output')
parser.add_argument('-loss', action='store_true')
parser.add_argument('-EER', action='store_true')
parser.add_argument('-acc', action='store_true')
parser.add_argument('-AUC', action='store_true')
parser.add_argument('-AP', action='store_true')
parser.add_argument('-vEER', action='store_true')
parser.add_argument('-vacc', action='store_true')
parser.add_argument('-vAUC', action='store_true')
parser.add_argument('-vAP', action='store_true')
parser.add_argument('--log_paths', nargs='+', type=str, default=['./logs/train_log/0929_174925_AIMv2_baseFF++v1_HQ.txt'])
args = parser.parse_args()

criterias = []

if args.loss:
    criterias.append('loss')
if args.EER:
    criterias.append('EER')
if args.acc:
    criterias.append('acc')
if args.AUC:
    criterias.append('AUC')
if args.AP:
    criterias.append('AP')
if args.vEER:
    criterias.append('vEER')
if args.vacc:
    criterias.append('vacc')
if args.vAUC:
    criterias.append('vAUC')
if args.vAP:
    criterias.append('vAP')


plot_dic = {}

for i,path in enumerate(args.log_paths):
    for criteria in criterias:
        now_id = str(i+1)+'_'+criteria
        plot_dic[now_id] = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Val' in line or 'test loss' in line or 'val loss' in line:
                    idx = line.index(criteria)+2+len(criteria)
                    # print(line[idx: idx+6])
                    digit = float(line[idx: idx+6])
                    plot_dic[now_id].append(digit)

# pdb.set_trace()
# print(plot_dic)
import numpy as np
x = 0
avgs = []
for k,v in plot_dic.items():
    x = max(x,len(v))
    
    break
for k,v in plot_dic.items():
    xs = np.arange(0,x,x/len(v))
    if len(xs)!=len(v):
        xs = xs[:len(v)]
    plt.plot(xs, v, label=k)
    avgs.append(np.round(np.mean(v[-10:]),decimals=5))
plt.legend()
plt.savefig('./plots/performance/'+args.name+'.png',dpi=320)
print(avgs)
