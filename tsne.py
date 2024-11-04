from sklearn.manifold import TSNE
from datetime import datetime

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

date_str = datetime.now().strftime('%m%d_%H%M%S')

sns.set_theme(style="whitegrid")

parser = argparse.ArgumentParser(description='t-SNE to visualize classification results')

parser.add_argument('--json_paths', nargs='+', type=str, default=[''])
parser.add_argument('--json_folder', type=str, default='')
parser.add_argument('--comments', type=str, default='')

args = parser.parse_args()
namespace = vars(args).keys()

data, labels = [], []
paths = args.json_paths
if args.json_folder:
    paths = [os.path.join(args.json_folder, diri) for diri in os.listdir(args.json_folder)]
for file in sorted(paths):
    with open(file, 'r') as f:
        dic = json.load(f)
    data += dic['feats']
    labels += [dic['labels']]*len(dic['feats'])
tsne = TSNE(n_components=2)
tsne.fit_transform(data)
frame = pd.DataFrame(tsne.embedding_,columns=['x','y'])
frame['labels'] = labels
# planets = sns.load_dataset("planets")
# import pdb; pdb.set_trace()
# x,y = tsne.embedding_[:,0], tsne.embedding_[:,1]
fig_path = '/'.join(['.', 'plots/tSNE', date_str+'_'+args.comments+'.png'])
# fig = sns.relplot(data=frame,x='x',y='y', hue=frame['labels'], size=frame['labels'], palette=cmap)
fig = sns.relplot(data=frame,x='x',y='y', hue=frame['labels'],s=10)
# fig.ax.xaxis.grid(True, "minor", linewidth=.25)
# fig.ax.yaxis.grid(True, "minor", linewidth=.25)
# scatter_fig = fig.get_figure()
# scatter_fig.savefig(fig_path, dpi=400)
fig.ax.xaxis.grid(False)
fig.ax.yaxis.grid(False)
fig.despine(left=False, bottom=False, top=False, right=False)
fig.set(ylabel=None,xlabel=None,yticklabels=[],xticklabels=[])  # remove the y-axis label
# fig.set(yticklabels=[],xticklabels=[])
# fig.tick_params(left=False, bottom=False)
fig.savefig(fig_path, dpi=400)